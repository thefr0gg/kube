"""Minimal causal language-model training example for Kube datasets.

This script demonstrates the smallest practical end-to-end training loop for a
sharded `.kubed` dataset:

1. Open a dataset with `kube.open_dataset(...)`.
2. Stream fixed-length token windows with `iter_sequence_batches(...)`.
3. Feed those windows into a tiny decoder-only Transformer.
4. Optimize next-token cross-entropy loss.
5. Optionally save a model checkpoint and a Kube resume-state file.

Why this example exists:

- to show how `.kubed` maps onto a standard causal LM training loop
- to provide a starting point that is easy to read and modify
- to keep Kube-specific logic small and explicit
- to optionally bootstrap a tiny text corpus from Hugging Face for demos

Important assumptions:

- the dataset already contains token IDs, not raw text
- when training an existing dataset, you know the tokenizer vocabulary size and
  pass it with `--vocab-size`
- this is a compact example, not a production trainer

Typical usage:

    python examples/train_kube_lm.py corpora/wiki.kubed \
        --vocab-size 50257 \
        --sequence-length 256 \
        --batch-size 8 \
        --steps 200 \
        --checkpoint-out checkpoints/tiny-lm.pt \
        --resume-state-out checkpoints/tiny-lm-resume.json

Bootstrap a tiny dataset from Hugging Face and train on it:

    python examples/train_kube_lm.py data/wikitext-demo.kubed \
        --bootstrap-hf \
        --hf-dataset wikitext \
        --hf-config wikitext-2-raw-v1 \
        --hf-split train[:1%] \
        --hf-tokenizer gpt2 \
        --sequence-length 128 \
        --batch-size 4 \
        --steps 50
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path

import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from kube import create_sharded, open_dataset


class TinyCausalLM(nn.Module):
    """A very small decoder-only Transformer for next-token prediction.

    The model is intentionally simple:

    - token embeddings
    - learned position embeddings
    - a stack of Transformer encoder blocks used with a causal mask
    - a final language-model projection tied to the token embeddings

    This is enough to demonstrate how Kube sequence batches feed into a
    standard autoregressive language-model objective.
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        sequence_length: int,
        d_model: int,
        n_head: int,
        n_layer: int,
        dropout: float,
    ) -> None:
        """Build the tiny decoder-only network used by the example trainer."""
        super().__init__()
        self.sequence_length = int(sequence_length)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(sequence_length, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=n_layer)
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return token logits for each position in the input sequence."""
        batch_size, seq_len = input_ids.shape
        if seq_len > self.sequence_length:
            raise ValueError(
                f"sequence length {seq_len} exceeds model limit {self.sequence_length}"
            )
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=input_ids.device, dtype=torch.bool),
            diagonal=1,
        )
        x = self.blocks(x, mask=causal_mask)
        x = self.final_norm(x)
        return self.lm_head(x)


def parse_args() -> argparse.Namespace:
    """Define the command-line interface for the training example."""
    parser = argparse.ArgumentParser(
        description="Train a tiny causal language model on a .kubed dataset"
    )
    parser.add_argument(
        "dataset",
        type=Path,
        help="Path to an existing .kubed dataset, or the output path to create when bootstrapping from Hugging Face",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        help="Tokenizer vocabulary size for the embedding/output layer; required unless it can be inferred from Hugging Face bootstrap",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        help="Training sequence length; defaults to the dataset manifest default",
    )
    parser.add_argument("--stride", type=int, help="Stride between windows; defaults to sequence length")
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--steps", default=200, type=int)
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--weight-decay", default=0.01, type=float)
    parser.add_argument("--grad-clip", default=1.0, type=float)
    parser.add_argument("--d-model", default=256, type=int)
    parser.add_argument("--n-head", default=4, type=int)
    parser.add_argument("--n-layer", default=4, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--prefetch-shards", default=1, type=int)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a specific torch device")
    parser.add_argument("--seed", default=1337, type=int)
    parser.add_argument("--log-every", default=10, type=int)
    parser.add_argument("--checkpoint-out", type=Path, help="Optional .pt checkpoint path")
    parser.add_argument(
        "--resume-state-in",
        type=Path,
        help="Optional dataset resume-state JSON file created by a prior run",
    )
    parser.add_argument(
        "--resume-state-out",
        type=Path,
        help="Optional path to write the final dataset resume state JSON",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile the model with torch.compile() when available",
    )

    bootstrap = parser.add_argument_group("hugging-face bootstrap")
    bootstrap.add_argument(
        "--bootstrap-hf",
        action="store_true",
        help="Download a small text dataset from Hugging Face, convert it into a .kubed corpus, and then train on it",
    )
    bootstrap.add_argument("--hf-dataset", default="wikitext", help="Hugging Face dataset name")
    bootstrap.add_argument("--hf-config", default="wikitext-2-raw-v1", help="Optional Hugging Face dataset configuration")
    bootstrap.add_argument(
        "--hf-split",
        default="train[:1%]",
        help="Dataset split or split slice to load from Hugging Face",
    )
    bootstrap.add_argument(
        "--hf-text-column",
        default="text",
        help="Column in the Hugging Face dataset that contains raw text",
    )
    bootstrap.add_argument(
        "--hf-tokenizer",
        default="gpt2",
        help="Tokenizer name to download from Hugging Face for tokenization",
    )
    bootstrap.add_argument(
        "--hf-max-documents",
        type=int,
        help="Optional cap on the number of downloaded documents after the split is loaded",
    )
    bootstrap.add_argument("--bootstrap-shards", default=2, type=int, help="Number of shards for the generated .kubed dataset")
    bootstrap.add_argument(
        "--bootstrap-token-bytes",
        default=4,
        type=int,
        help="Bytes per token stored in each shard",
    )
    bootstrap.add_argument(
        "--bootstrap-tokens-per-shard",
        default=4096,
        type=int,
        help="Token capacity per shard for the generated dataset",
    )
    bootstrap.add_argument(
        "--bootstrap-overwrite",
        action="store_true",
        help="Delete the target .kubed directory before creating the Hugging Face demo dataset",
    )
    return parser.parse_args()


def choose_device(raw: str) -> str:
    """Resolve `auto` into the best available Torch device."""
    if raw != "auto":
        return raw
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters in the model."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def require_hf_datasets():
    """Import the Hugging Face datasets library with a clear error message."""
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Hugging Face bootstrap requires the optional 'datasets' dependency. "
            "Install it before using --bootstrap-hf."
        ) from exc
    return load_dataset


def require_hf_tokenizer():
    """Import Hugging Face AutoTokenizer with a clear error message."""
    try:
        from transformers import AutoTokenizer
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Hugging Face bootstrap requires the optional 'transformers' dependency. "
            "Install it before using --bootstrap-hf."
        ) from exc
    return AutoTokenizer


def coerce_text(value) -> str:
    """Convert a dataset field into a single training text string.

    Some Hugging Face text columns contain plain strings, while others may
    expose structured values such as lists of strings. For this example we
    collapse everything into one flat text field per row.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return " ".join(str(part) for part in value if part is not None)
    return str(value)


def bootstrap_hf_dataset(args: argparse.Namespace) -> tuple[Path, int]:
    """Create a tiny `.kubed` text corpus from a Hugging Face dataset.

    The bootstrap flow is:

    1. Download a small text split from the Hugging Face Hub.
    2. Tokenize each row with a Hugging Face tokenizer.
    3. Create a sharded `.kubed` dataset locally.
    4. Import the token-id rows into Kube.

    Returns the created dataset path and the inferred tokenizer vocabulary size.
    """
    load_dataset = require_hf_datasets()
    AutoTokenizer = require_hf_tokenizer()

    root = args.dataset
    if root.suffix.lower() != ".kubed":
        raise ValueError("bootstrap target path must end with .kubed")
    if root.exists():
        if not args.bootstrap_overwrite:
            raise FileExistsError(
                f"{root} already exists; pass --bootstrap-overwrite to recreate it"
            )
        shutil.rmtree(root)

    split_name = args.hf_split
    config_name = None if args.hf_config in {None, "", "-"} else args.hf_config
    if config_name is None:
        hf_dataset = load_dataset(args.hf_dataset, split=split_name)
    else:
        hf_dataset = load_dataset(args.hf_dataset, config_name, split=split_name)
    if args.hf_max_documents is not None:
        limit = min(len(hf_dataset), int(args.hf_max_documents))
        hf_dataset = hf_dataset.select(range(limit))

    if args.hf_text_column not in hf_dataset.column_names:
        raise ValueError(
            f"text column '{args.hf_text_column}' was not found in Hugging Face dataset columns "
            f"{hf_dataset.column_names}"
        )

    tokenizer = AutoTokenizer.from_pretrained(args.hf_tokenizer)
    eos_token_id = tokenizer.eos_token_id
    token_rows: list[list[int]] = []
    kept_documents = 0

    for row in hf_dataset:
        text = coerce_text(row[args.hf_text_column]).strip()
        if not text:
            continue
        encoded = tokenizer(text, add_special_tokens=False)
        input_ids = [int(token_id) for token_id in encoded["input_ids"]]
        if not input_ids and eos_token_id is None:
            continue
        token_rows.append(input_ids)
        kept_documents += 1

    if not token_rows:
        raise RuntimeError("bootstrap did not produce any tokenized documents")

    bootstrap_sequence_length = args.sequence_length or 128
    dataset = create_sharded(
        root,
        shards=args.bootstrap_shards,
        token_bytes=args.bootstrap_token_bytes,
        tokens_per_shard=args.bootstrap_tokens_per_shard,
        sequence_length=bootstrap_sequence_length,
        dataset_name=f"{args.hf_dataset.split('/')[-1]}-demo",
    )
    frame = pl.DataFrame({"input_ids": token_rows})
    dataset.import_token_frame(frame, tokens_column="input_ids", eos_token_id=eos_token_id)

    info = dataset.inspect()
    print(
        json.dumps(
            {
                "bootstrap": {
                    "dataset_path": str(root),
                    "hf_dataset": args.hf_dataset,
                    "hf_config": config_name,
                    "hf_split": split_name,
                    "hf_text_column": args.hf_text_column,
                    "hf_tokenizer": args.hf_tokenizer,
                    "kept_documents": kept_documents,
                    "tokenizer_vocab_size": len(tokenizer),
                    "eos_token_id": eos_token_id,
                },
                "kube_dataset": info,
            },
            indent=2,
        )
    )
    return root, int(len(tokenizer))


def maybe_save_checkpoint(
    path: Path | None,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    step: int,
    last_loss: float,
    resume_state: dict | None,
) -> None:
    """Persist a training checkpoint if an output path was provided.

    The checkpoint includes:

    - model weights
    - optimizer state
    - a compact copy of the training config
    - the latest Kube resume state, if available
    """
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": int(step),
        "last_loss": float(last_loss),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": {
            "dataset": str(args.dataset),
            "vocab_size": int(args.vocab_size),
            "sequence_length": int(args.sequence_length),
            "stride": int(args.stride),
            "batch_size": int(args.batch_size),
            "d_model": int(args.d_model),
            "n_head": int(args.n_head),
            "n_layer": int(args.n_layer),
            "dropout": float(args.dropout),
        },
        "resume_state": resume_state,
    }
    torch.save(payload, path)


def main() -> None:
    """Run a tiny training job on a `.kubed` dataset.

    The Kube-specific part is deliberately small: open the dataset, ask it for
    sequence batches, and convert those batches into Torch tensors. Everything
    else is a normal PyTorch training loop.
    """
    args = parse_args()
    torch.manual_seed(args.seed)

    dataset_path = args.dataset
    inferred_vocab_size = None
    if args.bootstrap_hf:
        dataset_path, inferred_vocab_size = bootstrap_hf_dataset(args)

    dataset = open_dataset(dataset_path, mode="r")
    dataset_info = dataset.inspect()

    if args.sequence_length is None:
        args.sequence_length = dataset.default_sequence_length
    if args.stride is None:
        args.stride = args.sequence_length

    if args.vocab_size is None:
        args.vocab_size = inferred_vocab_size
    if args.vocab_size is None:
        raise ValueError(
            "--vocab-size is required unless it can be inferred from Hugging Face bootstrap"
        )
    if args.vocab_size <= 0:
        raise ValueError("--vocab-size must be positive")
    if args.sequence_length <= 0:
        raise ValueError("--sequence-length must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.d_model <= 0 or args.n_head <= 0 or args.n_layer <= 0:
        raise ValueError("model dimensions must be positive")
    if args.d_model % args.n_head != 0:
        raise ValueError("--d-model must be divisible by --n-head")

    device = choose_device(args.device)
    model = TinyCausalLM(
        vocab_size=args.vocab_size,
        sequence_length=args.sequence_length,
        d_model=args.d_model,
        n_head=args.n_head,
        n_layer=args.n_layer,
        dropout=args.dropout,
    ).to(device)

    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    start_state = None
    if args.resume_state_in is not None:
        start_state = dataset.load_resume_state(args.resume_state_in)

    print(
        json.dumps(
            {
                "dataset": dataset_info,
                "dataset_path": str(dataset_path),
                "device": device,
                "vocab_size": args.vocab_size,
                "sequence_length": args.sequence_length,
                "stride": args.stride,
                "batch_size": args.batch_size,
                "steps": args.steps,
                "trainable_parameters": count_parameters(model),
            },
            indent=2,
        )
    )

    iterator = dataset.iter_sequence_batches(
        args.batch_size,
        args.sequence_length,
        stride=args.stride,
        start_state=start_state,
        prefetch_shards=args.prefetch_shards,
    )

    model.train()
    total_tokens = 0
    started_at = time.perf_counter()
    last_resume_state = start_state
    last_loss = float("nan")
    completed_steps = 0

    for step, batch in enumerate(iterator, start=1):
        if step > args.steps:
            break

        input_ids = torch.tensor(batch["input_ids"], dtype=torch.long, device=device)
        target_ids = torch.tensor(batch["target_ids"], dtype=torch.long, device=device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids)
        loss = F.cross_entropy(logits.reshape(-1, args.vocab_size), target_ids.reshape(-1))
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        total_tokens += int(input_ids.numel())
        last_resume_state = batch["resume_state"]
        last_loss = float(loss.item())
        completed_steps = step

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            elapsed = max(time.perf_counter() - started_at, 1e-9)
            tokens_per_second = total_tokens / elapsed
            perplexity = math.exp(min(last_loss, 20.0))
            print(
                f"step={step} loss={last_loss:.4f} ppl={perplexity:.2f} "
                f"tokens={total_tokens} tok/s={tokens_per_second:.0f}"
            )

        if step >= args.steps:
            break

    if completed_steps == 0:
        raise RuntimeError(
            "dataset did not yield any training batches; "
            "check sequence length, shard contents, and dataset size"
        )

    if args.resume_state_out is not None and last_resume_state is not None:
        dataset.save_resume_state(args.resume_state_out, last_resume_state)
        print(f"saved resume state to {args.resume_state_out}")

    maybe_save_checkpoint(
        args.checkpoint_out,
        model=model,
        optimizer=optimizer,
        args=args,
        step=completed_steps,
        last_loss=last_loss,
        resume_state=last_resume_state,
    )
    if args.checkpoint_out is not None:
        print(f"saved checkpoint to {args.checkpoint_out}")


if __name__ == "__main__":
    main()

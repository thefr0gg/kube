"""Standard metadata key constants for Kube manifests.

This module exists to prevent key-name typos and let tools introspect the
standard metadata vocabulary without reading raw JSON files.

These constants are used as keys in the ``metadata`` dict stored inside
``.kube`` file headers and ``.kubed`` dataset manifests.
"""

from __future__ import annotations

SPLIT = "split"
"""Dataset partition label: ``"train"``, ``"test"``, or ``"val"``."""

TOKENIZER_NAME = "tokenizer_name"
"""Human-readable tokenizer identifier (e.g. ``"gpt2"``)."""

TOKENIZER_PATH = "tokenizer_path"
"""File-system path or HuggingFace Hub ID of the tokenizer."""

SEQUENCE_LENGTH = "sequence_length"
"""Default token window length used when iterating sequences."""

FEATURE_SCHEMA = "feature_schema"
"""Schema mode string used when the kube was exported as a feature table."""

SOURCE_DATASET = "source_dataset"
"""Human-readable name or path of the upstream dataset before import."""

PIPELINE_VERSION = "pipeline_version"
"""Monotone version tag for the ingest pipeline that produced the file."""

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const holder = document.getElementById("canvas-holder");
const statusEl = document.getElementById("status");
const titleEl = document.getElementById("title");
const selectionEl = document.getElementById("selection");
const viewModeEl = document.getElementById("view-mode");
const sliceAxisEl = document.getElementById("slice-axis");
const sliceIndexEl = document.getElementById("slice-index");
const lodEl = document.getElementById("lod");

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a0a0a);

const camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.1, 2000);
camera.position.set(40, 40, 60);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);
holder.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

scene.add(new THREE.AmbientLight(0xffffff, 0.7));
const dirLight = new THREE.DirectionalLight(0xffffff, 0.6);
dirLight.position.set(1, 1, 1);
scene.add(dirLight);

let mesh = null;
let socket = null;
let selected = null;
let kubeState = null;

const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();

function apiUrl(path) {
  return path;
}

async function fetchState() {
  const res = await fetch(apiUrl("/api/state"));
  return res.json();
}

async function fetchMesh() {
  const res = await fetch(apiUrl("/api/mesh"));
  return res.json();
}

function buildMeshObject(data) {
  if (mesh) {
    scene.remove(mesh);
    mesh.geometry.dispose();
    mesh.material.dispose();
  }
  const geometry = new THREE.BufferGeometry();
  const positions = new Float32Array(data.positions);
  const colors = new Float32Array(data.colors.length);
  for (let i = 0; i < data.colors.length; i++) colors[i] = data.colors[i] / 255;
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  geometry.setIndex(data.indices);
  geometry.computeVertexNormals();

  const material = new THREE.MeshStandardMaterial({ vertexColors: true, side: THREE.DoubleSide });
  mesh = new THREE.Mesh(geometry, material);
  mesh.userData.facesPerQuad = true;
  scene.add(mesh);
  statusEl.textContent = `${data.vertex_count} verts`;
}

async function refreshMesh() {
  const data = await fetchMesh();
  buildMeshObject(data);
}

async function refreshState() {
  kubeState = await fetchState();
  titleEl.textContent = kubeState.file_label || "kube";
  viewModeEl.value = kubeState.view_mode;
  sliceAxisEl.value = String(kubeState.slice_axis ?? 0);
  sliceIndexEl.value = String(kubeState.slice_index ?? 0);
  lodEl.value = String(kubeState.lod ?? 1);
}

async function pushViewConfig() {
  const params = new URLSearchParams({
    view_mode: viewModeEl.value,
    slice_axis: sliceAxisEl.value,
    slice_index: sliceIndexEl.value,
    lod: lodEl.value,
  });
  await fetch(apiUrl("/api/view?" + params.toString()), { method: "POST" });
  await refreshMesh();
}

viewModeEl.addEventListener("change", pushViewConfig);
sliceAxisEl.addEventListener("change", pushViewConfig);
sliceIndexEl.addEventListener("change", pushViewConfig);
lodEl.addEventListener("change", pushViewConfig);

function connectSocket() {
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  socket = new WebSocket(`${proto}://${window.location.host}/ws/edit`);
  socket.addEventListener("message", (event) => {
    const msg = JSON.parse(event.data);
    if (msg.ok) {
      refreshMesh();
    } else {
      statusEl.textContent = `error: ${msg.error}`;
    }
  });
}

function renderSelectionPanel() {
  if (!selected) {
    selectionEl.innerHTML = '<div class="k">no selection</div>';
    return;
  }
  const { x, y, z, face, value, states } = selected;
  selectionEl.innerHTML = `
    <div><span class="k">cell</span> ${x}, ${y}, ${z}</div>
    <div><span class="k">face</span> ${face}</div>
    <div><span class="k">value</span> <input id="face-value" type="number" min="0" max="${states - 1}" value="${value}" /></div>
    <button id="apply-value">Apply</button>
  `;
  document.getElementById("apply-value").addEventListener("click", () => {
    const newValue = parseInt(document.getElementById("face-value").value, 10);
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ x, y, z, face, value: newValue }));
    }
  });
}

async function handlePick(event) {
  if (!mesh) return;
  const rect = renderer.domElement.getBoundingClientRect();
  pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(pointer, camera);
  const hits = raycaster.intersectObject(mesh, false);
  if (hits.length === 0) return;

  const hit = hits[0];
  const quadIndex = Math.floor(hit.faceIndex / 2);
  const posAttr = mesh.geometry.getAttribute("position");
  // Each quad is 4 contiguous vertices in the position buffer (see mesh.py's
  // vertex_offset += n * 4). Average them to get the face's center — but that
  // center sits half a cell out along the face normal (mesh.py's FACE_VERTS
  // are +-0.5 in-plane and a constant +-0.5 along the normal axis), so
  // subtract the normal offset to recover the cell's own center.
  const base = quadIndex * 4;
  const faceCenter = new THREE.Vector3();
  for (let i = 0; i < 4; i++) {
    faceCenter.x += posAttr.getX(base + i);
    faceCenter.y += posAttr.getY(base + i);
    faceCenter.z += posAttr.getZ(base + i);
  }
  faceCenter.multiplyScalar(0.25);

  const normal = hit.face.normal;
  const cellCenter = faceCenter.clone().addScaledVector(normal, -0.5);

  const shape = kubeState.shape;
  const x = Math.round(cellCenter.x + shape[0] / 2 - 0.5);
  const y = Math.round(cellCenter.y + shape[1] / 2 - 0.5);
  const z = Math.round(cellCenter.z + shape[2] / 2 - 0.5);
  const face = faceFromNormal(normal);

  try {
    const res = await fetch(apiUrl(`/api/cell?x=${x}&y=${y}&z=${z}`));
    const cell = await res.json();
    selected = { x, y, z, face, value: cell.faces[face], states: kubeState.states };
    renderSelectionPanel();
  } catch (err) {
    statusEl.textContent = `pick failed: ${err}`;
  }
}

function faceFromNormal(normal) {
  if (Math.abs(normal.y) > 0.9) return normal.y > 0 ? 0 : 1;
  if (Math.abs(normal.x) > 0.9) return normal.x < 0 ? 2 : 3;
  return normal.z > 0 ? 4 : 5;
}

renderer.domElement.addEventListener("dblclick", handlePick);

window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}

async function main() {
  await refreshState();
  await refreshMesh();
  connectSocket();
  animate();
  statusEl.textContent = "ready — double-click a face to select it";
}

main();

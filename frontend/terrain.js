import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const PLANE_SIZE = 200;
const GRID = 256;
const HEIGHT_SCALE = 45;

let renderer, scene, camera, controls;
let mesh = null;
let initialized = false;

export function initScene(canvas) {
  if (initialized) return;
  initialized = true;

  renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setClearColor(0x0b0d10);
  _resizeRenderer();

  scene = new THREE.Scene();
  scene.fog = new THREE.Fog(0x0b0d10, 400, 900);

  camera = new THREE.PerspectiveCamera(55, 1, 0.1, 2000);
  camera.position.set(260, 200, 260);
  camera.lookAt(0, 0, 0);

  const ambient = new THREE.AmbientLight(0xffffff, 0.35);
  scene.add(ambient);

  const sun = new THREE.DirectionalLight(0xffffff, 1.1);
  sun.position.set(100, 180, 60);
  scene.add(sun);

  controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.minDistance = 30;
  controls.maxDistance = 500;
  controls.maxPolarAngle = Math.PI * 0.49; // don't orbit underneath the terrain

  window.addEventListener("resize", _resizeRenderer);

  const tick = () => {
    requestAnimationFrame(tick);
    controls.update();
    renderer.render(scene, camera);
  };
  tick();
}

export async function renderTerrain(imageUrl) {
  const heights = await _loadHeights(imageUrl);

  const geometry = new THREE.PlaneGeometry(
    PLANE_SIZE, PLANE_SIZE, GRID - 1, GRID - 1,
  );
  geometry.rotateX(-Math.PI / 2); // lay flat on XZ plane

  const pos = geometry.attributes.position;
  for (let i = 0; i < pos.count; i++) {
    pos.setY(i, heights[i] * HEIGHT_SCALE);
  }
  pos.needsUpdate = true;
  geometry.computeVertexNormals();

  const material = new THREE.MeshStandardMaterial({
    color: 0xb8b8b8,
    roughness: 0.95,
    metalness: 0.0,
    flatShading: false,
  });

  if (mesh) {
    scene.remove(mesh);
    mesh.geometry.dispose();
    mesh.material.dispose();
  }
  mesh = new THREE.Mesh(geometry, material);
  scene.add(mesh);
}

async function _loadHeights(imageUrl) {
  const img = await _loadImage(imageUrl);

  const canvas = document.createElement("canvas");
  canvas.width = GRID;
  canvas.height = GRID;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(img, 0, 0, GRID, GRID);
  const { data } = ctx.getImageData(0, 0, GRID, GRID);

  const heights = new Float32Array(GRID * GRID);
  for (let i = 0; i < heights.length; i++) {
    heights[i] = data[i * 4] / 255; // red channel, 0..1
  }
  return heights;
}

function _loadImage(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = url;
  });
}

function _resizeRenderer() {
  if (!renderer) return;
  const canvas = renderer.domElement;
  const { clientWidth, clientHeight } = canvas;
  renderer.setSize(clientWidth, clientHeight, false);
  if (camera) {
    camera.aspect = clientWidth / clientHeight;
    camera.updateProjectionMatrix();
  }
}

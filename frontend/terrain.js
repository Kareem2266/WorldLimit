import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const PLANE_SIZE = 200;
const GRID = 256;
const HEIGHT_SCALE = 45;
const WATER_LEVEL = 0.10;      // fraction of HEIGHT_SCALE
const WATER_EXTENT = 1.0;      // 1.0 = exactly matches terrain
const BASE_DEPTH = 6;          // how far below Y=0 the skirt/floor sits

let renderer, scene, camera, controls;
let mesh = null;
let water = null;
let base = null;
let initialized = false;

// ---------- scene setup ----------

export function initScene(canvas) {
  if (initialized) return;
  initialized = true;

  renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setClearColor(0x0b0d10);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
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
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  const shadowHalf = PLANE_SIZE * 0.8;
  sun.shadow.camera.left = -shadowHalf;
  sun.shadow.camera.right = shadowHalf;
  sun.shadow.camera.top = shadowHalf;
  sun.shadow.camera.bottom = -shadowHalf;
  sun.shadow.camera.near = 10;
  sun.shadow.camera.far = 600;
  sun.shadow.bias = -0.0003;
  scene.add(sun);

  controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.minDistance = 30;
  controls.maxDistance = 500;
  controls.maxPolarAngle = Math.PI * 0.49;

  window.addEventListener("resize", _resizeRenderer);

  const tick = () => {
    requestAnimationFrame(tick);
    controls.update();
    renderer.render(scene, camera);
  };
  tick();
}

//  mesh build 

export async function renderTerrain(imageUrl, params = null) {
  const heights = await _loadHeights(imageUrl);

  const geometry = new THREE.PlaneGeometry(
    PLANE_SIZE, PLANE_SIZE, GRID - 1, GRID - 1,
  );
  geometry.rotateX(-Math.PI / 2);

  const pos = geometry.attributes.position;
  for (let i = 0; i < pos.count; i++) {
    pos.setY(i, heights[i] * HEIGHT_SCALE);
  }
  pos.needsUpdate = true;
  geometry.computeVertexNormals();

  // vertex colors driven by elevation + climate
  const palette = _pickPalette(params);
  const colors = new Float32Array(pos.count * 3);
  const c = new THREE.Color();
  for (let i = 0; i < pos.count; i++) {
    _colorForHeight(heights[i], palette, c);
    colors[i * 3 + 0] = c.r;
    colors[i * 3 + 1] = c.g;
    colors[i * 3 + 2] = c.b;
  }
  geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));

  const material = new THREE.MeshStandardMaterial({
    vertexColors: true,
    roughness: 0.92,
    metalness: 0.0,
    flatShading: false,
  });

  if (mesh) {
    scene.remove(mesh);
    mesh.geometry.dispose();
    mesh.material.dispose();
  }
  mesh = new THREE.Mesh(geometry, material);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  scene.add(mesh);

  _buildBase(params);
  _buildWater(params);
}

//  base (skirt + floor)

function _buildBase(params) {
  if (base) {
    scene.remove(base);
    base.traverse((o) => {
      if (o.geometry) o.geometry.dispose();
      if (o.material) o.material.dispose();
    });
    base = null;
  }

  const palette = _pickPalette(params);
  const col = palette.midland;
  const baseColor = new THREE.Color(col[0] * 0.55, col[1] * 0.50, col[2] * 0.45);

  const material = new THREE.MeshStandardMaterial({
    color: baseColor,
    roughness: 0.95,
    metalness: 0.0,
    side: THREE.DoubleSide,
  });

  const floorY = -BASE_DEPTH;
  base = new THREE.Group();

  const pos = mesh.geometry.attributes.position;
  const edges = _edgeIndexStrips();
  for (const edge of edges) {
    const wall = _buildWall(edge, pos, floorY, material);
    wall.castShadow = true;
    wall.receiveShadow = true;
    base.add(wall);
  }

  const floorGeom = new THREE.PlaneGeometry(PLANE_SIZE, PLANE_SIZE);
  floorGeom.rotateX(-Math.PI / 2);
  const floor = new THREE.Mesh(floorGeom, material);
  floor.position.y = floorY;
  floor.receiveShadow = true;
  base.add(floor);

  scene.add(base);
}

// Index strips for the four edges of the GRID×GRID terrain vertex grid.
function _edgeIndexStrips() {
  const top = [], bottom = [], left = [], right = [];
  for (let c = 0; c < GRID; c++) top.push(c);
  for (let c = 0; c < GRID; c++) bottom.push((GRID - 1) * GRID + c);
  for (let r = 0; r < GRID; r++) left.push(r * GRID);
  for (let r = 0; r < GRID; r++) right.push(r * GRID + (GRID - 1));
  return [top, bottom, left, right];
}

function _buildWall(indices, posAttr, floorY, material) {
  const N = indices.length;
  const verts = new Float32Array(N * 2 * 3);
  for (let i = 0; i < N; i++) {
    const idx = indices[i];
    const x = posAttr.getX(idx);
    const y = posAttr.getY(idx);
    const z = posAttr.getZ(idx);
    verts[i * 3 + 0] = x;
    verts[i * 3 + 1] = y;
    verts[i * 3 + 2] = z;
    verts[(N + i) * 3 + 0] = x;
    verts[(N + i) * 3 + 1] = floorY;
    verts[(N + i) * 3 + 2] = z;
  }
  const idx = [];
  for (let i = 0; i < N - 1; i++) {
    const t0 = i, t1 = i + 1, b0 = N + i, b1 = N + i + 1;
    idx.push(t0, b0, t1, t1, b0, b1);
  }
  const geom = new THREE.BufferGeometry();
  geom.setAttribute("position", new THREE.BufferAttribute(verts, 3));
  geom.setIndex(idx);
  geom.computeVertexNormals();
  return new THREE.Mesh(geom, material);
}

// water

function _buildWater(params) {
  if (water) {
    scene.remove(water);
    water.geometry.dispose();
    water.material.dispose();
    water = null;
  }

  // Skip water in very dry climates (desert).
  const bio12 = params ? params.bio12 : 800;
  if (bio12 < 200) return;

  const geometry = new THREE.PlaneGeometry(
    PLANE_SIZE * WATER_EXTENT,
    PLANE_SIZE * WATER_EXTENT,
  );
  geometry.rotateX(-Math.PI / 2);

  const material = new THREE.MeshStandardMaterial({
    color: 0x1a3a5a,
    roughness: 0.25,
    metalness: 0.2,
    transparent: true,
    opacity: 0.78,
    side: THREE.DoubleSide,
  });

  water = new THREE.Mesh(geometry, material);
  water.position.y = WATER_LEVEL * HEIGHT_SCALE;
  water.receiveShadow = true;
  scene.add(water);
}

// Biomes
// Pick a (lowland, midland) color pair from bio1 (temp) + bio12 (precip).
// Rock + snow are universal but the snow line shifts with temperature.
//
//   bio1  : annual mean temperature, °C
//   bio12 : annual precipitation, mm

function _pickPalette(params) {
  const bio1 = params ? params.bio1 : 10;
  const bio12 = params ? params.bio12 : 800;

  let lowland, midland;

  if (bio1 >= 20) {
    // hot
    if (bio12 < 300) {
      // desert
      lowland = [0.83, 0.69, 0.45];
      midland = [0.72, 0.47, 0.36];
    } else if (bio12 < 1200) {
      // savanna
      lowland = [0.66, 0.72, 0.34];
      midland = [0.48, 0.41, 0.23];
    } else {
      // jungle
      lowland = [0.17, 0.42, 0.23];
      midland = [0.24, 0.35, 0.20];
    }
  } else if (bio1 >= 8) {
    // warm / temperate
    if (bio12 < 400) {
      // dry grassland
      lowland = [0.62, 0.65, 0.30];
      midland = [0.46, 0.45, 0.24];
    } else {
      // forest
      lowland = [0.30, 0.55, 0.28];
      midland = [0.20, 0.36, 0.20];
    }
  } else if (bio1 >= -2) {
    // cool — boreal / conifer
    lowland = [0.22, 0.40, 0.22];
    midland = [0.26, 0.32, 0.24];
  } else {
    // cold — tundra
    lowland = [0.50, 0.48, 0.42];
    midland = [0.58, 0.55, 0.50];
  }

  // snow line (normalized 0..1) drops with temperature
  let snowLine;
  if (bio1 >= 20) snowLine = 1.3;           // effectively never
  else if (bio1 >= 10) snowLine = 0.92;
  else if (bio1 >= 0) snowLine = 0.72;
  else snowLine = 0.5;

  return {
    lowland,
    midland,
    rock: [0.42, 0.40, 0.38],
    snow: [0.94, 0.95, 0.97],
    snowLine,
  };
}

// Band-based interpolation: walk through breakpoints, lerp between the two
// bands bracketing `h`.
function _colorForHeight(h, p, out) {
  const bands = [
    { t: 0.00, c: p.lowland },
    { t: 0.35, c: p.midland },
    { t: Math.max(p.snowLine - 0.12, 0.5), c: p.rock },
    { t: p.snowLine, c: p.snow },
  ];

  for (let i = 1; i < bands.length; i++) {
    if (h <= bands[i].t) {
      const a = bands[i - 1];
      const b = bands[i];
      const span = Math.max(b.t - a.t, 1e-6);
      const t = Math.min(Math.max((h - a.t) / span, 0), 1);
      out.setRGB(
        a.c[0] + (b.c[0] - a.c[0]) * t,
        a.c[1] + (b.c[1] - a.c[1]) * t,
        a.c[2] + (b.c[2] - a.c[2]) * t,
      );
      return;
    }
  }
  out.setRGB(p.snow[0], p.snow[1], p.snow[2]);
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
    heights[i] = data[i * 4] / 255;
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

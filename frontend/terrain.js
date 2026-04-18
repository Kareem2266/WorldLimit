import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { mergeGeometries } from "three/addons/utils/BufferGeometryUtils.js";

const PLANE_SIZE = 200;
const GRID = 256;
const HEIGHT_SCALE_REF = 60;          // world-units used at REFERENCE_RANGE_M
const REFERENCE_RANGE_M = 700;        // real elevation range that maps to HEIGHT_SCALE_REF
const HEIGHT_SCALE_MIN = 20;          // never render flatter than this
const HEIGHT_SCALE_MAX = 110;         // cap dramatic vertical exaggeration
const WATER_LEVEL = 0.10;             // fraction of current height scale
const WATER_EXTENT = 1.0;             // 1.0 = exactly matches terrain
const BASE_DEPTH = 6;                 // how far below Y=0 the skirt/floor sits

let heightScale = HEIGHT_SCALE_REF;   // updated per-generate based on real elevation range

let renderer, scene, camera, controls;
let mesh = null;
let water = null;
let base = null;
let vegetation = null;
let initialized = false;

let _coniferGeom = null;
let _broadleafGeom = null;
let _vegMaterial = null;

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

export async function renderTerrain(imageUrl, params = null, minM = null, maxM = null) {
  const heights = await _loadHeights(imageUrl);
  heightScale = _computeHeightScale(minM, maxM);

  const geometry = new THREE.PlaneGeometry(
    PLANE_SIZE, PLANE_SIZE, GRID - 1, GRID - 1,
  );
  geometry.rotateX(-Math.PI / 2);

  const pos = geometry.attributes.position;
  for (let i = 0; i < pos.count; i++) {
    pos.setY(i, heights[i] * heightScale);
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
  _buildVegetation(params, heights);
}

//  vegetation 

function _buildVegetation(params, heights) {
  if (vegetation) {
    scene.remove(vegetation);
    vegetation.traverse((o) => {
      if (o.geometry) o.geometry.dispose();
    });
    vegetation = null;
  }

  const plan = _vegetationPlan(params);
  if (plan.attempts === 0) return;

  const palette = _pickPalette(params);
  const snowY = palette.snowLine * heightScale;
  const waterY = params && params.bio12 < 200 ? -Infinity : WATER_LEVEL * heightScale;
  const maxSlope = 0.08;

  const conifers = [];
  const broadleafs = [];
  const m = new THREE.Matrix4();
  const p = new THREE.Vector3();
  const q = new THREE.Quaternion();
  const e = new THREE.Euler();
  const s = new THREE.Vector3();

  for (let i = 0; i < plan.attempts; i++) {
    const col = Math.floor(Math.random() * (GRID - 2)) + 1;
    const row = Math.floor(Math.random() * (GRID - 2)) + 1;
    const idx = row * GRID + col;
    const h = heights[idx];
    const y = h * heightScale;

    if (y < waterY + 0.4) continue;
    if (y > snowY - 1.5) continue;

    const dx = Math.abs(heights[row * GRID + col + 1] - h);
    const dz = Math.abs(heights[(row + 1) * GRID + col] - h);
    if (Math.max(dx, dz) > maxSlope) continue;

    const x = -PLANE_SIZE / 2 + (col / (GRID - 1)) * PLANE_SIZE;
    const z = -PLANE_SIZE / 2 + (row / (GRID - 1)) * PLANE_SIZE;

    const r = Math.random();
    const pickConifer = r < plan.coniferRatio;
    const pickBroadleaf = !pickConifer && r < plan.coniferRatio + plan.broadleafRatio;
    if (!pickConifer && !pickBroadleaf) continue;

    const scl = 0.7 + Math.random() * 0.6;
    p.set(x, y - 0.1, z);
    e.set(0, Math.random() * Math.PI * 2, 0);
    q.setFromEuler(e);
    s.set(scl, scl, scl);
    m.compose(p, q, s);

    (pickConifer ? conifers : broadleafs).push(m.clone());
  }

  vegetation = new THREE.Group();
  const mat = _getVegMaterial();
  const coniferMesh = _instancedFromMatrices(_getConiferGeom(), mat, conifers);
  const broadleafMesh = _instancedFromMatrices(_getBroadleafGeom(), mat, broadleafs);
  if (coniferMesh) vegetation.add(coniferMesh);
  if (broadleafMesh) vegetation.add(broadleafMesh);
  scene.add(vegetation);
}

function _instancedFromMatrices(geom, material, matrices) {
  if (matrices.length === 0) return null;
  const im = new THREE.InstancedMesh(geom, material, matrices.length);
  for (let i = 0; i < matrices.length; i++) im.setMatrixAt(i, matrices[i]);
  im.instanceMatrix.needsUpdate = true;
  im.castShadow = true;
  im.receiveShadow = true;
  return im;
}

function _vegetationPlan(params) {
  if (!params) return { attempts: 0 };
  const { bio1, bio12 } = params;

  if (bio1 >= 20 && bio12 < 300) return { attempts: 0, coniferRatio: 0, broadleafRatio: 0 };      // desert
  if (bio1 >= 20 && bio12 < 1200) return { attempts: 150, coniferRatio: 0, broadleafRatio: 1 };   // savanna
  if (bio1 >= 20) return { attempts: 1200, coniferRatio: 0, broadleafRatio: 1 };                  // jungle
  if (bio1 >= 8 && bio12 < 400) return { attempts: 120, coniferRatio: 0.3, broadleafRatio: 0.7 }; // dry temperate
  if (bio1 >= 8) return { attempts: 900, coniferRatio: 0.5, broadleafRatio: 0.5 };                // temperate forest
  if (bio1 >= -2) return { attempts: 550, coniferRatio: 1, broadleafRatio: 0 };                   // boreal
  return { attempts: 0, coniferRatio: 0, broadleafRatio: 0 };                                     // tundra
}

function _getVegMaterial() {
  if (_vegMaterial) return _vegMaterial;
  _vegMaterial = new THREE.MeshStandardMaterial({
    vertexColors: true,
    roughness: 0.95,
    metalness: 0.0,
    flatShading: true,
  });
  return _vegMaterial;
}

function _getConiferGeom() {
  if (_coniferGeom) return _coniferGeom;
  const trunk = new THREE.CylinderGeometry(0.14, 0.22, 1.6, 6);
  trunk.translate(0, 0.8, 0);
  _paintGeom(trunk, [0.32, 0.20, 0.11]);
  const foliage = new THREE.ConeGeometry(1.1, 3.3, 8);
  foliage.translate(0, 2.7, 0);
  _paintGeom(foliage, [0.12, 0.36, 0.18]);
  _coniferGeom = mergeGeometries([trunk, foliage]);
  return _coniferGeom;
}

function _getBroadleafGeom() {
  if (_broadleafGeom) return _broadleafGeom;
  const trunk = new THREE.CylinderGeometry(0.17, 0.25, 1.8, 6);
  trunk.translate(0, 0.9, 0);
  _paintGeom(trunk, [0.36, 0.24, 0.14]);
  const foliage = new THREE.SphereGeometry(1.3, 6, 5);
  foliage.translate(0, 2.6, 0);
  _paintGeom(foliage, [0.22, 0.44, 0.20]);
  _broadleafGeom = mergeGeometries([trunk, foliage]);
  return _broadleafGeom;
}

function _paintGeom(geom, rgb) {
  const n = geom.attributes.position.count;
  const arr = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    arr[i * 3 + 0] = rgb[0];
    arr[i * 3 + 1] = rgb[1];
    arr[i * 3 + 2] = rgb[2];
  }
  geom.setAttribute("color", new THREE.BufferAttribute(arr, 3));
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
  water.position.y = WATER_LEVEL * heightScale;
  water.receiveShadow = true;
  scene.add(water);
}

function _computeHeightScale(minM, maxM) {
  if (minM == null || maxM == null) return HEIGHT_SCALE_REF;
  const range = Math.max(maxM - minM, 0);
  const raw = HEIGHT_SCALE_REF * (range / REFERENCE_RANGE_M);
  return Math.min(Math.max(raw, HEIGHT_SCALE_MIN), HEIGHT_SCALE_MAX);
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

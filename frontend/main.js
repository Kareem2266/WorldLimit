import { initScene, renderTerrain } from "/app/terrain.js";

const form = document.getElementById("form");
const promptInput = document.getElementById("prompt");
const submitBtn = document.getElementById("submit");
const exportBtn = document.getElementById("export");
const statusEl = document.getElementById("status");
const paramsEl = document.getElementById("params");
const canvas = document.getElementById("canvas");

initScene(canvas);

// Remember the last generated (prompt, seed) so Export stays in sync
// with whatever's on screen.
let lastGenerated = null;

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const prompt = promptInput.value.trim();
  if (!prompt) return;

  submitBtn.disabled = true;
  exportBtn.disabled = true;
  statusEl.classList.remove("error");
  statusEl.textContent = "generating... (~2s)";

  const t0 = performance.now();
  try {
    const res = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt }),
    });

    if (!res.ok) {
      const text = await res.text();
      throw new Error(`${res.status}: ${text}`);
    }

    const data = await res.json();
    const url = `${data.heightmap_url}?t=${Date.now()}`;
    await renderTerrain(url, data.params, data.heightmap_min_m, data.heightmap_max_m);

    const dt = ((performance.now() - t0) / 1000).toFixed(1);
    statusEl.textContent = `done in ${dt}s`;
    paramsEl.textContent = JSON.stringify(data.params, null, 2);
    paramsEl.classList.add("visible");

    lastGenerated = { prompt, seed: null };
    exportBtn.disabled = false;
  } catch (err) {
    statusEl.classList.add("error");
    statusEl.textContent = `error: ${err.message}`;
  } finally {
    submitBtn.disabled = false;
  }
});

exportBtn.addEventListener("click", async () => {
  if (!lastGenerated) return;

  exportBtn.disabled = true;
  submitBtn.disabled = true;
  statusEl.classList.remove("error");
  statusEl.textContent = "packaging ZIP (~3s)...";

  try {
    const res = await fetch("/api/export", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(lastGenerated),
    });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(`${res.status}: ${text}`);
    }

    const blob = await res.blob();
    const slug = res.headers.get("X-WorldLimit-Slug") || "world";
    const trees = res.headers.get("X-WorldLimit-Tree-Count") || "0";

    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `world_${slug}.zip`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(a.href);

    statusEl.textContent = `exported world_${slug}.zip (${trees} trees)`;
  } catch (err) {
    statusEl.classList.add("error");
    statusEl.textContent = `export failed: ${err.message}`;
  } finally {
    exportBtn.disabled = false;
    submitBtn.disabled = false;
  }
});

import { initScene, renderTerrain } from "/app/terrain.js";

const form = document.getElementById("form");
const promptInput = document.getElementById("prompt");
const submitBtn = document.getElementById("submit");
const statusEl = document.getElementById("status");
const paramsEl = document.getElementById("params");
const canvas = document.getElementById("canvas");

initScene(canvas);

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const prompt = promptInput.value.trim();
  if (!prompt) return;

  submitBtn.disabled = true;
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
  } catch (err) {
    statusEl.classList.add("error");
    statusEl.textContent = `error: ${err.message}`;
  } finally {
    submitBtn.disabled = false;
  }
});

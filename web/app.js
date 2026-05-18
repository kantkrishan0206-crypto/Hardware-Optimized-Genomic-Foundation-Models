const $ = (id) => document.getElementById(id);

function apiBase() {
  return $("apiBase").value.replace(/\/$/, "");
}

async function postJson(path, payload) {
  const response = await fetch(`${apiBase()}${path}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.detail || response.statusText);
  }
  return data;
}

function show(id, data) {
  $(id).textContent = JSON.stringify(data, null, 2);
}

async function checkHealth() {
  const badge = $("healthBadge");
  try {
    const response = await fetch(`${apiBase()}/health`);
    const data = await response.json();
    badge.textContent = `${data.status} · ${data.device}`;
    badge.classList.remove("error");
  } catch {
    badge.textContent = "offline";
    badge.classList.add("error");
  }
}

$("predictForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  show("predictOutput", { status: "running" });
  try {
    const data = await postJson("/predict", {
      sequence: $("predictSequence").value,
      max_length: 512,
    });
    show("predictOutput", data);
  } catch (error) {
    show("predictOutput", { error: error.message });
  }
});

$("variantForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  show("variantOutput", { status: "running" });
  try {
    const data = await postJson("/score_variant", {
      reference_sequence: $("referenceSequence").value,
      alternate_sequence: $("alternateSequence").value,
      max_length: 512,
    });
    show("variantOutput", data);
  } catch (error) {
    show("variantOutput", { error: error.message });
  }
});

$("memoryForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  show("memoryOutput", { status: "running" });
  try {
    const data = await postJson("/api/estimate-memory", {
      tokens: Number($("tokens").value),
      heads: Number($("heads").value),
      dim: Number($("dim").value),
      dtype_bytes: 2,
    });
    show("memoryOutput", data);
  } catch (error) {
    show("memoryOutput", { error: error.message });
  }
});

checkHealth();
setInterval(checkHealth, 15000);

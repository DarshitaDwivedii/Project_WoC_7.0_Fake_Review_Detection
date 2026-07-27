const API_URL = "http://localhost:8000/predict";

const btn = document.getElementById("checkBtn");
const textarea = document.getElementById("reviewText");
const resultDiv = document.getElementById("result");

btn.addEventListener("click", async () => {
  const text = textarea.value.trim();
  if (!text) return;

  btn.disabled = true;
  btn.textContent = "Checking...";
  resultDiv.style.display = "none";

  try {
    const res = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const data = await res.json();

    resultDiv.className = data.is_fake ? "fake" : "real";
    const pct = data.confidence != null ? ` (${Math.round(data.confidence * 100)}% confidence)` : "";
    resultDiv.textContent = data.is_fake
      ? `⚠ Likely Fake${pct}`
      : `✓ Likely Real${pct}`;
    resultDiv.style.display = "block";
  } catch (err) {
    resultDiv.className = "fake";
    resultDiv.textContent = "Could not reach backend. Is the API running on localhost:8000?";
    resultDiv.style.display = "block";
  } finally {
    btn.disabled = false;
    btn.textContent = "Check Review";
  }
});

// Finds review blocks on an Amazon product page, sends their text to the
// local backend for a prediction, and injects a badge next to each one.

const API_URL = "http://localhost:8000/predict";

// Only the stable review-body hook — do NOT also match nested spans inside
// it, or every word-span gets treated as its own "review" (many duplicate
// badges on one comment).
const REVIEW_SELECTOR = "[data-hook='review-body']";

function getReviewNodes() {
  return Array.from(document.querySelectorAll(REVIEW_SELECTOR)).filter(
    (el) => !el.dataset.frdChecked
  );
}

function makeBadge(result) {
  const badge = document.createElement("span");
  badge.className = `frd-badge ${result.is_fake ? "frd-fake" : "frd-real"}`;
  const pct = result.confidence != null ? ` (${Math.round(result.confidence * 100)}%)` : "";
  badge.textContent = result.is_fake ? `⚠ Likely Fake${pct}` : `✓ Likely Real${pct}`;
  return badge;
}

async function checkReview(node) {
  node.dataset.frdChecked = "true"; // mark BEFORE the async call so it's never picked up twice
  const text = node.innerText?.trim();
  if (!text || text.length < 5) return;

  try {
    const res = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    if (!res.ok) return;
    const result = await res.json();
    node.insertAdjacentElement("beforebegin", makeBadge(result));
  } catch (err) {
    // backend not running / CORS / network issue — fail silently on-page
    console.debug("Fake Review Detector: could not reach API", err);
  }
}

function scanPage() {
  getReviewNodes().forEach(checkReview);
}

scanPage();

// Amazon loads more reviews dynamically as you scroll/paginate. Debounce so
// a burst of DOM mutations (including our own badge inserts) triggers only
// one scan instead of one per mutation.
let debounceTimer = null;
const observer = new MutationObserver(() => {
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(scanPage, 300);
});
observer.observe(document.body, { childList: true, subtree: true });
// Minimal service worker — no persistent background logic needed currently.
// Reserved for future features (e.g. caching results, badge counts).
chrome.runtime.onInstalled.addListener(() => {
  console.log("Fake Review Detector installed.");
});

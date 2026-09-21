// Reserve a tab during the click, before viewer preparation outlasts browser activation.
export function desktopTarget(url = new URL(location.href)) {
  if (url.searchParams.get("desktop") === "browser") return "browser";
  const host = url.hostname.toLowerCase();
  return host === "localhost" || host === "[::1]" || /^127\./.test(host)
    ? "auto"
    : "browser";
}

export function reserveViewerTab() {
  const tab = window.open("/static/viewer-launch.html", "_blank");
  if (tab) tab.opener = null;
  return tab;
}

export function openPreparedViewer(url, tab, automatic = false) {
  if (tab) {
    // Closing or navigating away cancels automatic navigation, not the server job.
    if (!isLoadingTab(tab)) return false;
    tab.location.replace(url);
    return true;
  }
  if (automatic) {
    // Chat responses arrive after browser activation expires. Same-tab navigation
    // also lets a blocked popup complete without requiring another click.
    window.location.assign(url);
    return true;
  }
  return false;
}

export function closePendingViewer(tab) {
  if (isLoadingTab(tab)) tab.close();
}

function isLoadingTab(tab) {
  if (!tab || tab.closed) return false;
  try {
    return (
      tab.location.href === "about:blank" ||
      tab.location.href ===
        new URL("/static/viewer-launch.html", location.origin).href
    );
  } catch {
    // A tab the user has navigated elsewhere is no longer our loading page.
    return false;
  }
}

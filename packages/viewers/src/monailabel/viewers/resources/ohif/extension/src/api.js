export async function api(path, body, binary = false, extraHeaders = {}) {
  const raw = body instanceof Uint8Array;
  const response = await fetch("/api" + path, {
    credentials: "same-origin",
    method: body === undefined ? "GET" : "POST",
    headers:
      body === undefined
        ? {}
        : {
            ...extraHeaders,
            "Content-Type": raw
              ? "application/octet-stream"
              : "application/json",
          },
    body: body === undefined ? undefined : raw ? body : JSON.stringify(body),
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(
      response.status === 401
        ? "Sign in to MONAI Label in another tab, then reopen this viewer."
        : typeof error.detail === "string"
          ? error.detail
          : JSON.stringify(error.detail || response.status),
    );
  }
  return binary
    ? new Uint8Array(await response.arrayBuffer())
    : response.json();
}

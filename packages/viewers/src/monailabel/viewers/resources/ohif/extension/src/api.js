/*
Copyright (c) MONAI Consortium
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

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
        : response.status === 404 && path.endsWith("/spatial-inference")
          ? "Restart MONAI Label and reopen the viewer to enable Update. Your draft is preserved."
          : typeof error.detail === "string"
            ? error.detail
            : JSON.stringify(error.detail || response.status),
    );
  }
  return binary
    ? new Uint8Array(await response.arrayBuffer())
    : response.json();
}

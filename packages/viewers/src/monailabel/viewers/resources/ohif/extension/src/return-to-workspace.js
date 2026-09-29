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

import { randomId } from "./random-id.js";

// Reuse the launching workspace without granting the viewer access to window.opener.
// A copied link, closed workspace or blocked tab close falls back to this tab.
export async function returnToWorkspace(projectId, page) {
  const destination = new URL("/", location.origin);
  destination.searchParams.set("project", projectId);
  destination.searchParams.set("page", page);
  const workspace = new URLSearchParams(location.search).get("workspace");
  let returned = false;
  if (workspace && "BroadcastChannel" in window) {
    let channel;
    try {
      channel = new BroadcastChannel("monailabel-workspace-" + workspace);
      const requestId = randomId();
      returned = await new Promise((resolve) => {
        const timeout = setTimeout(() => resolve(false), 3000);
        channel.onmessage = ({ data }) => {
          if (data?.requestId !== requestId) return;
          clearTimeout(timeout);
          resolve(data.type === "returned");
        };
        channel.postMessage({ type: "return", requestId, projectId, page });
      });
    } catch {
      // Restricted browser storage must not prevent returning to the workspace.
      returned = false;
    } finally {
      channel?.close();
    }
  }
  if (returned) {
    window.close();
    // Browser-opened tabs may refuse close(); still leave the user in the workspace.
    setTimeout(() => location.replace(destination.href), 100);
  } else {
    location.assign(destination.href);
  }
}

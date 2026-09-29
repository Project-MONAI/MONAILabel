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

import { useSyncExternalStore } from "react";

// OHIF unmounts collapsed side panels. Keep one sample's editing/chat session
// alive for the lifetime of its viewer so hiding the assistant cannot lose work.
const sessions = new WeakMap();

export function panelSession(manager, assetId) {
  let session = sessions.get(manager);
  if (!session || session.assetId !== assetId) {
    const values = new Map();
    const listeners = new Set();
    session = {
      assetId,
      state: { current: { undo: [], redo: [], job: null, proposal: null } },
      subscribe(listener) {
        listeners.add(listener);
        return () => listeners.delete(listener);
      },
      get(key, initial) {
        if (!values.has(key)) values.set(key, initial);
        return values.get(key);
      },
      set(key, update) {
        values.set(
          key,
          typeof update === "function" ? update(values.get(key)) : update,
        );
        listeners.forEach((listener) => listener());
      },
    };
    sessions.set(manager, session);
  }
  return session;
}

export function usePanelState(session, key, initial) {
  const value = useSyncExternalStore(session.subscribe, () =>
    session.get(key, initial),
  );
  return [value, (update) => session.set(key, update)];
}

export function clearPanelSession(manager) {
  sessions.delete(manager);
}

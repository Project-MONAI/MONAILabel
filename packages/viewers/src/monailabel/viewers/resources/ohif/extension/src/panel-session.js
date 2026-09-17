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

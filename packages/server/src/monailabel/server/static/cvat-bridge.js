/* Native CVAT adapter, pinned and browser-tested with CVAT 2.76. */
(() => {
  Object.defineProperty(window, "cvat", {
    configurable: true,
    set(core) {
      core.config.backendAPI = "/cvat-api";
      Object.defineProperty(window, "cvat", {
        value: core,
        configurable: true,
      });
    },
  });
  document.addEventListener(
    "plugins.ready",
    () => {
      window.cvatUI.registerComponent(({ store, dispatch, actionCreators }) => {
        const state = () => store.getState().annotation;
        let lastSelected = null;
        const unsubscribe = store.subscribe(() => {
          const active = state().annotations.activatedStateID;
          if (active !== null) lastSelected = active;
        });
        const job = () => {
          const value = state().job.instance;
          if (!value)
            throw new Error("Wait for the CVAT video to finish loading.");
          return value;
        };
        let snapshotContent = null;
        let snapshotSignature = null;
        const snapshot = async () => {
          const content = JSON.stringify(await job().annotations.export());
          if (content !== snapshotContent) {
            // Bind a session-local token to the exact draft, including every
            // track/keyframe. getRandomValues also works on network HTTP origins.
            snapshotSignature = Array.from(
              crypto.getRandomValues(new Uint8Array(32)),
              (byte) => byte.toString(16).padStart(2, "0"),
            ).join("");
            snapshotContent = content;
          }
          return snapshotSignature;
        };
        const selected = () => {
          const a = state().annotations;
          const candidates = a.states.filter(
            (s) =>
              s.objectType === "track" &&
              ["rectangle", "polygon"].includes(s.shapeType) &&
              !s.outside &&
              !s.lock &&
              !s.rotation,
          );
          return (
            candidates.find(
              (s) => s.clientID === (a.activatedStateID ?? lastSelected),
            ) || (candidates.length === 1 ? candidates[0] : null)
          );
        };
        const refresh = () =>
          dispatch(
            actionCreators.changeFrameAsync(
              state().player.frame.number,
              false,
              undefined,
              true,
            ),
          );
        window.monaiVideo = {
          ready: () =>
            Boolean(
              state().job.instance &&
                state().player.frame.data &&
                !state().job.fetching &&
                !state().player.frame.fetching,
            ),
          frame: () => state().player.frame.number,
          hasSelection: () => Boolean(selected()),
          selection: () => {
            const s = selected();
            return s
              ? `${s.label.name} · track ${s.clientID}`
              : "Draw a rectangle using Track, or select an existing visible, unlocked rectangle or polygon track";
          },
          snapshot,
          async context() {
            if (!this.ready())
              throw new Error(
                "Wait for the selected video frame to finish loading.",
              );
            const s = selected();
            return {
              frame: this.frame(),
              client_id: s?.clientID ?? null,
              label: s?.label.id ?? null,
              box: s?.shapeType === "rectangle" ? [...s.points] : null,
              points: s?.shapeType === "polygon" ? [...s.points] : null,
              occluded: s?.occluded ?? false,
              draft_signature: await snapshot(),
            };
          },
          async save() {
            await job().annotations.save();
          },
          async clear(request, labelIDs) {
            const { clearAnnotations } = await import("/static/cvat-edits.js");
            if ((await snapshot()) !== request.draft_signature)
              throw new Error(
                "The CVAT draft changed. Retry the clear request to keep your edits.",
              );
            const count = await clearAnnotations(job(), request, labelIDs);
            await refresh();
            return count;
          },
          async history(operation, signature) {
            if (!["undo", "redo"].includes(operation))
              throw new Error("Unknown history operation.");
            if ((await snapshot()) !== signature)
              throw new Error(
                "The CVAT draft changed. Retry the history request.",
              );
            await job().actions[operation]();
            await refresh();
          },
          async apply(proposal, labelID) {
            const request = proposal.request;
            if ((await snapshot()) !== request.draft_signature)
              throw new Error(
                "The CVAT draft changed while tracking. Run tracking again to keep your edits.",
              );
            const changes = [];
            const last = proposal.keyframes.at(-1).frame;
            if (request.client_id === null) {
              if (!job().labels.some((label) => label.id === labelID))
                throw new Error(
                  "The proposed tool label is unavailable in this CVAT task.",
                );
              const keys = [...proposal.keyframes];
              if (last < job().stopFrame)
                keys.push({ ...keys.at(-1), frame: last + 1, outside: true });
              // One undoable native addition; existing objects and IDs are untouched.
              await job().annotations.commit(
                {
                  tracks: [
                    {
                      label_id: labelID,
                      frame: request.seed.frame,
                      group: 0,
                      source: "auto",
                      attributes: [],
                      elements: [],
                      shapes: keys.map((key) => ({
                        type: key.points ? "polygon" : "rectangle",
                        frame: key.frame,
                        points: [...(key.points || key.box)],
                        outside: key.outside,
                        occluded: key.occluded,
                        rotation: 0,
                        z_order: 0,
                        attributes: [],
                      })),
                    },
                  ],
                },
                {},
                request.seed.frame,
              );
              await refresh();
              return;
            }
            // Preserve interpolation beyond the requested range.
            const keys = [...proposal.keyframes];
            if (last < job().stopFrame) {
              const boundary = (
                await job().annotations.get(last + 1, true, [])
              ).find((s) => s.clientID === request.client_id);
              if (boundary && !boundary.keyframe)
                keys.push({
                  frame: last + 1,
                  ...(boundary.shapeType === "polygon"
                    ? { points: [...boundary.points] }
                    : { box: [...boundary.points] }),
                  outside: boundary.outside,
                  occluded: boundary.occluded,
                });
            }
            for (const key of keys) {
              const s = (await job().annotations.get(key.frame, true, [])).find(
                (s) => s.clientID === request.client_id,
              );
              if (
                !s ||
                s.lock ||
                s.objectType !== "track" ||
                s.shapeType !==
                  (request.output === "polygon" ? "polygon" : "rectangle")
              )
                throw new Error(
                  "The selected track is unavailable or locked. No proposal was applied.",
                );
              changes.push([s, key]);
            }
            // All operations below modify native objects; unrelated tracks and their IDs survive.
            for (const [s, key] of changes) {
              s.points = [...(key.points || key.box)];
              s.outside = key.outside;
              s.occluded = key.occluded;
              s.keyframe = true;
              await s.save();
            }
            await refresh();
          },
        };
        return {
          name: "MONAI Label video",
          destructor() {
            unsubscribe();
            delete window.monaiVideo;
          },
        };
      });
    },
    { once: true },
  );
})();

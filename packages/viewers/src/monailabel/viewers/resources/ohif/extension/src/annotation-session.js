import { api } from "./api";
import { panelSession } from "./panel-session";
import { MaskTransfer } from "./masks";
import { initializeAnnotation } from "./initialize";

/** Own loading at viewer scope, including when the assistant is closed on phones. */
export function startAnnotationSession(servicesManager, assetId) {
  const session = panelSession(servicesManager, assetId);
  const s = session.state.current;
  let failed = false,
    attempt = null;
  const fail = (error) => {
    if (s.closed) return;
    failed = true;
    session.set("ready", false);
    session.set("loadError", error.message);
  };
  const load = async () => {
    if (s.closed || failed || !s.transfer?.canPrepare()) return;
    if (!s.initialized)
      session.set(
        "loadingText",
        s.asset.annotation_id ? "Loading annotation…" : "Preparing annotation…",
      );
    try {
      if (!(await initializeAnnotation(s, api)) || s.closed) return;
      session.set("ready", true);
      if (!s.welcomed) {
        s.welcomed = true;
        session.set("messages", (items = []) => [
          ...items,
          s.asset.name +
            (s.asset.annotation_id
              ? " · Saved annotation loaded. Inspect it or make corrections."
              : " · Ready to annotate. Describe what you want to segment."),
        ]);
      }
    } catch (error) {
      fail(error);
    }
  };
  const connect = async () => {
    failed = false;
    session.set("loadError", "");
    try {
      if (!assetId)
        throw new Error("Open a sample from the MONAI Label workspace.");
      if (!s.asset) {
        const asset = await api("/assets/" + assetId);
        const [source, project, choices, permissions] = await Promise.all([
          api("/assets/" + assetId + "/dicom"),
          api("/projects/" + asset.project_id),
          api("/projects/" + asset.project_id + "/models"),
          api("/projects/" + asset.project_id + "/permissions"),
        ]);
        if (s.closed) return;
        Object.assign(s, {
          asset,
          source,
          project,
          transfer: new MaskTransfer(
            servicesManager.services,
            asset,
            source,
            project,
          ),
        });
        session.set("models", choices);
        session.set(
          "permission",
          permissions.roles.some((role) =>
            ["manager", "annotator"].includes(role),
          ),
        );
        session.set(
          "canReview",
          permissions.roles.some((role) =>
            ["manager", "reviewer"].includes(role),
          ),
        );
      }
      if (!s.initialized) session.set("loadingText", "Waiting for the image…");
      await load();
    } catch (error) {
      fail(error);
    }
  };
  session.retry = () => {
    if (!attempt)
      attempt = connect().finally(() => {
        attempt = null;
      });
    return attempt;
  };
  const { viewportGridService, cornerstoneViewportService, displaySetService } =
    servicesManager.services;
  const subscriptions = [
    viewportGridService.subscribe(
      viewportGridService.EVENTS.VIEWPORTS_READY,
      load,
    ),
    cornerstoneViewportService.subscribe(
      cornerstoneViewportService.EVENTS.VIEWPORT_DATA_CHANGED,
      load,
    ),
    displaySetService.subscribe(
      displaySetService.EVENTS.DISPLAY_SETS_ADDED,
      load,
    ),
  ];
  session.retry();
  return () => {
    s.closed = true;
    subscriptions.forEach(({ unsubscribe }) => unsubscribe());
  };
}

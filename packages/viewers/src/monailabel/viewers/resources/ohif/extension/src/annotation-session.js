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
        const preferred =
          session.get("selected", "") ||
          project.annotation_model_id ||
          Object.values(project.defaults || {})[0];
        session.set(
          "selected",
          choices.find((model) => model.id === preferred)?.id ||
            choices[0]?.id ||
            "",
        );
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

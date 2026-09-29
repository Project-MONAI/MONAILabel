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

import React, { useState } from "react";
import Panel from "./Panel";
import { startAnnotationSession } from "./annotation-session";
import { clearPanelSession } from "./panel-session";

let stopSession;

function Workspace({ servicesManager, ManualPanel }) {
  const [manual, setManual] = useState(false);
  return (
    <div style={{ height: "100%", display: "flex", flexDirection: "column" }}>
      <div
        className="monailabel-tabs"
        role="tablist"
        aria-label="Annotation tools"
      >
        <button
          role="tab"
          aria-selected={!manual}
          onClick={() => setManual(false)}
        >
          Assistant
        </button>
        <button
          role="tab"
          aria-selected={manual}
          onClick={() => setManual(true)}
        >
          Manual editing
        </button>
      </div>
      <div
        style={{ display: manual ? "none" : "block", flex: 1, minHeight: 0 }}
      >
        <Panel servicesManager={servicesManager} />
      </div>
      <div
        style={{
          display: manual ? "block" : "none",
          flex: 1,
          minHeight: 0,
          overflowY: "auto",
        }}
      >
        <ManualPanel />
      </div>
    </div>
  );
}
export default {
  id: "@monailabel/extension",
  onModeEnter({ servicesManager }) {
    stopSession?.();
    clearPanelSession(servicesManager);
    const assetId = new URLSearchParams(location.search).get("assetId");
    stopSession = startAnnotationSession(servicesManager, assetId);
  },
  onModeExit() {
    stopSession?.();
    stopSession = null;
  },
  getPanelModule({ servicesManager, extensionManager }) {
    const component = () => {
      const ManualPanel = extensionManager.getModuleEntry(
        "@ohif/extension-cornerstone.panelModule.panelSegmentationWithToolsLabelMap",
      ).component;
      return (
        <Workspace
          servicesManager={servicesManager}
          ManualPanel={ManualPanel}
        />
      );
    };
    return [
      {
        name: "assistant",
        iconName: "tab-segmentation",
        iconLabel: "Assistant",
        label: "Assistant",
        component,
      },
    ];
  },
};

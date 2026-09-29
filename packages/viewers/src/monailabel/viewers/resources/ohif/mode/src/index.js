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

import segmentationMode from "@ohif/mode-segmentation";

const id = "@monailabel/mode";
const extensionDependencies = {
  ...segmentationMode.extensionDependencies,
  "@monailabel/extension": "0.1.0",
};
export default {
  id,
  extensionDependencies,
  modeFactory(options) {
    const mode = segmentationMode.modeFactory(options);
    return {
      ...mode,
      id,
      routeName: "monailabel",
      displayName: "MONAI Label · Annotation",
      extensions: extensionDependencies,
      routes: mode.routes.map((route) => ({
        ...route,
        layoutTemplate(context) {
          const layout = route.layoutTemplate(context);
          return {
            ...layout,
            props: {
              ...layout.props,
              rightPanels: ["@monailabel/extension.panelModule.assistant"],
              rightPanelResizable: true,
              leftPanelClosed: window.innerWidth < 1200,
              rightPanelClosed: window.innerWidth < 700,
              rightPanelInitialExpandedWidth: Math.min(
                400,
                window.innerWidth - 80,
              ),
              rightPanelMinimumExpandedWidth: Math.min(
                300,
                window.innerWidth - 80,
              ),
            },
          };
        },
      })),
    };
  },
};

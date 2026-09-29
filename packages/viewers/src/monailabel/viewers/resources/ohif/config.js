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

const annotationAsset = new URLSearchParams(window.location.search).get(
  "assetId",
);
const dicomRoot = annotationAsset
  ? `/api/assets/${encodeURIComponent(annotationAsset)}/dicomweb`
  : "/api/dicomweb";

window.config = {
  name: "MONAI Label",
  routerBasename: "/ohif",
  extensions: [],
  modes: [],
  showStudyList: true,
  showWarningMessageForCrossOrigin: false,
  showCPUFallbackMessage: true,
  maxNumberOfWebWorkers: 3,
  showLoadingIndicator: true,
  // OHIF's default remembers only the current tab. Persist confirmation for
  // the lifetime of this browser's site data (a 100-year expiry).
  investigationalUseDialog: { option: "configure", days: 36500 },
  defaultDataSourceName: "monailabel",
  dataSources: [
    {
      namespace: "@ohif/extension-default.dataSourcesModule.dicomweb",
      sourceName: "monailabel",
      configuration: {
        friendlyName: "MONAI Label · Local DICOM",
        name: "monailabel",
        wadoUriRoot: dicomRoot,
        qidoRoot: dicomRoot,
        wadoRoot: dicomRoot,
        qidoSupportsIncludeField: true,
        imageRendering: "wadors",
        thumbnailRendering: "wadors",
        enableStudyLazyLoad: true,
        supportsFuzzyMatching: false,
        supportsWildcard: true,
        bulkDataURI: { enabled: true, relativeResolution: "studies" },
        singlepart: "bulkdata,video,pdf",
      },
    },
  ],
};

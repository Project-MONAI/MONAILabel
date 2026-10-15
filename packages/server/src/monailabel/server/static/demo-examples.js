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
import { button, escapeHTML } from "./ui.js";

export const DEMO_EXAMPLES = {
  radiology: {
    title: "Radiology · CT",
    load: "Load CT example",
    instructions: "Open the CT in Slicer or OHIF. Select Spleen and nnInteractive, add a positive point inside the spleen and negative points outside, then Update volume. SAM 2.1 offers Update slice instead. Inspect, correct and submit for review.",
  },
  mri: {
    title: "Radiology · MRI",
    load: "Load MRI example",
    instructions: "Open the public T2 prostate MRI in Slicer or OHIF. Ask: Segment the prostate using TotalSegmentator MRI. Inspect and correct the result before submitting for review. This is a workflow example, not a clinical validation or independent model benchmark.",
  },
  pathology: {
    title: "Pathology · QuPath",
    load: "Load pathology example",
    instructions: "Open CMU-1-Small-Region.svs in QuPath and select a small region. With GPT Astra selected, ask: Segment all nuclei in the selected region using GPT Astra. Inspect and correct the draft, then submit for review. SAM 2.1 remains available for point/box prompting. Do not request the entire slide at once.",
  },
  endoscopy: {
    title: "Endoscopy · CVAT",
    load: "Load endoscopy video",
    instructions: "Open the HyperKvasir video in CVAT and find a frame showing the snare. With GPT Astra selected, ask: Locate and segment the tool using GPT Astra. Track it for the next 20 frames. GPT supplies the initial segmentation; SAM 2.1 tracks it. Inspect every frame, correct the polygons and submit for review. Save your draft before closing the viewer.",
  },
};

export function preparedExamples(state) {
  if (!state.visitor?.restricted_roles || state.project?.id !== state.visitor.initial_project_id)
    return "";
  const busy = state.jobs.some((job) => job.kind === "dataset_import" && ["queued", "running"].includes(job.status));
  return `<section class="card" id="demo-examples"><h2>Try the prepared examples</h2><p>Each example loads once into your private project from the host's verified public-data cache. Your existing annotations are kept.</p>${Object.entries(DEMO_EXAMPLES).map(([id, example]) => `<details class="demo-example" data-example="${id}"><summary>${escapeHTML(example.title)}</summary><p>${escapeHTML(example.instructions)}</p>${busy ? '<p class="muted">Wait for the current sample to finish loading.</p>' : button(example.load, "demo-example", id)}</details>`).join("")}<p class="muted">Models must be prepared by the host. GPT annotation and chat are separate paid features; availability is controlled by the host.</p></section>`;
}

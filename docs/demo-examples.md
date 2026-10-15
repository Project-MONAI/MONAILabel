# Prepared visitor examples

Restricted demo visitors can use **Try the prepared examples** on Overview or Datasets. The Spleen CT remains the automatic starter. MRI, pathology and endoscopy are optional, fixed imports from the host's verified cache; visitors cannot choose a URL, destination project or import size. Each example is reused if present. Retries are bounded to three import jobs per example, and another sample cannot start while an import is running. No button clears existing annotations or extends a visit.

## Radiology: Slicer and OHIF

Open the prepared Spleen CT. Choose **nnInteractive** and **Spleen**, place a positive point inside the intended object, add negative points where needed, and choose **Update volume**. You can instead choose **SAM 2.1** for **Update slice**. Inspect and correct the mask before submitting it for review. Full-volume nnInteractive uses CT/MRI spatial prompts; it is not the pathology/video model.

Example viewer prompt:

```text
Start nnInteractive for spleen.
```

## Pathology: QuPath

When the host-approved GPT Astra demo connection is available, QuPath selects it initially without changing the project's radiology default. Select a small region and ask: **Segment all nuclei in the selected region using GPT Astra.** Other models remain selectable. Inspect every proposed object; this is not a validated whole-slide nuclei detector.

Choose **Load pathology example**, then open `CMU-1-Small-Region.svs` in QuPath. The bounded OpenSlide sample has no reference masks and is CC0. The starter adds a **Nucleus** target without changing existing label IDs. Zoom in, choose **SAM 2.1** and **Nucleus**, then add a positive point or tight box around one nucleus. Use **Update image**, or select a small region and use **Update region**. Add negative points to exclude nearby tissue, inspect and correct the result, save the draft, and submit for review.

```text
Start SAM 2.1 for nucleus.
```

SAM is prompted, one-object segmentation, not automatic whole-slide nuclei detection. Sample workflows and model execution checks do not establish clinical accuracy.

## Endoscopy: CVAT

When available, the host-approved GPT Astra demo connection is selected initially. The prompt hint is **Locate and segment the tool using GPT Astra. Track it for the next 20 frames.** GPT creates the initial segmentation and SAM 2.1 tracks it; this does not request a separate GPT annotation for every frame. Near the clip boundary, choose a range within the remaining frames.

Choose **Load endoscopy video**. Open the HyperKvasir snare clip in CVAT and select a frame showing the tool. Choose **SAM 2.1**, **Snare** and **New object**, place a point or box, then choose **Update frame**. Use **Track range** for a short 8–16 frame interval. Inspect every frame; correct the editable polygons, save the CVAT draft and submit the range for review. The original 71-second video is preserved; short tracking ranges limit interactive work. This is the public HyperKvasir research clip with CC BY-NC 4.0 terms, not patient data supplied by a visitor. Attribution and source checksums remain on its import job.

## Host preparation and GPT

### Completion and setup previews

QuPath and CVAT show a completion dialog only after a successful annotation submission or review decision. **Keep working** retains QuPath's current editable document; CVAT opens an editable task for the next revision after a consumed submission. **Return to workspace** leaves the viewer for Datasets or Reviews. QuPath uses its normal close handler and unsaved-work checks. A failed submission preserves the draft and does not show a success dialog. The redundant workspace **Submit saved tracks** shortcut is hidden for visitors; submission within CVAT remains available.

Restricted visitors can create VISTA3D, TotalSegmentator CT/MRI and U-Net setups in their own initial demo project using buttons or chat. For example: **Based on VISTA3D, create model "VISTA3D-Spleen" for spleen.** This creates setup metadata, not trained weights. U-Net previews use default 3D single-channel settings, even in a mixed-modality demo project. Visitors cannot supply arbitrary settings, weights or connections. Actual training stays blocked at HTTP, assistant, service and job boundaries, even with the preview environment switch disabled. General Annotator/Reviewer permissions outside this narrow demo capability do not change.

### Optional MRI

**Load MRI example** imports one cached Decathlon Prostate T2 image using the whole-gland template and adds a Prostate target without renumbering existing labels. CT and MRI imports are independently idempotent. Open the MRI in Slicer or OHIF and ask **Segment the prostate using TotalSegmentator MRI.** The public sample retains its source grouping, original geometry and CC BY-SA 4.0 attribution. It is a workflow demonstration, not independent validation of the pretrained model.

With the host's dataset/model cache configured, `uv run python scripts/demo_mri_acceptance.py --output /absolute/new-test-directory` imports CT and MRI and runs real MRI segmentation in a disposable visitor workspace. It makes no paid model calls and does not open the live workspace.

Prepare the pinned public samples, SAM 2.1 weights, nnInteractive worker and viewer assets before starting restricted mode. Missing caches fail closed without visitor downloads. The examples use local models and do not need an OpenAI key. GPU memory must be available for inference; cached models do not all need to remain resident.

GPT annotation and the conversation coordinator are separate configurations. `uv run python scripts/demo_openai_run.py --key-file /absolute/private/directory/key` checks an owner-only key file without contacting OpenAI or starting anything. For an approved test or deployment, place this loader **after** `scripts/demo_run.sh`, then use `--run` followed by the normal MONAI command with `--assistant openai --assistant-model gpt-6-luna`. The loader supplies `OPENAI_API_KEY` only to that process and its descendants. Keep the file on the server in a 0700 directory with 0600 permissions; never pass its value in chat, a browser, command arguments or a checked-in service file.

`uv run python scripts/demo_openai_smoke.py --output /absolute/private/new-test-directory` runs an offline Luna tool round-trip. Adding `--execute --key-file /absolute/private/directory/key` authorizes at most two real requests, capped at 1024 output tokens each, without retries. `--model` selects Luna, Sol or Astra explicitly. Use a fresh output directory only for a separately approved test. The exclusive marker prevents an accidental paid repeat. Only sanitized accounting and pass/fail metadata are reported.

`uv run python scripts/demo_gpt_annotation_smoke.py --output /absolute/private/new-annotation-test` checks the polygon adapter with a synthetic white circle, offline by default. The same `--execute --key-file` options permit one real annotation request using the exact Sol demo configuration; `--model demo-gpt-astra` explicitly selects the comparison. It never opens a running workspace or sends clinical data. A successful synthetic geometry check is not evidence of medical segmentation quality.

Loading a key alone does not enable visitor annotation. The separate host opt-in `MONAILABEL_DEMO_GPT=1`, together with `OPENAI_API_KEY`, adds **GPT-6.1 Sol · demo annotation** and **GPT-6 Astra · demo comparison** to existing projects at startup and to new visitors' projects. It permits those exact server-owned connections for every authenticated, unexpired visitor with project access, including future public registrants; it is not an account allowlist. Visitors still cannot register models, discover providers, upload arbitrary data or configure credentials. Disabling the flag or removing the key immediately blocks visitor GPT inference; existing proposals and annotations are preserved. No per-user spending budget is implemented by this opt-in. Expiry, concurrency, cancellation and request output limits remain active, but they do not cap the host's total API bill. Cancelling a local request cannot undo an upstream charge.

Use Luna for conversation and typed workflow tools. QuPath and CVAT initially select Astra when available; Sol remains selectable for comparison. Start with a single slice, image or video frame; annotating a full volume makes multiple paid requests. SAM 2.1 and nnInteractive remain the local segmentation choices and existing project defaults are not replaced. GPT returns experimental polygon drafts, not clinically validated segmentations; review source geometry and correct the output. The sign-in notice discloses that conversation context and selected annotation images are sent to OpenAI. Keep patient data, credentials and other sensitive information out of the demo. Approve a saved/closed-viewer deployment window before restarting the live application.

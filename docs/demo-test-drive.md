# Private demo test drive

This is a trusted-user, local-model test with optional timed visitor controls, not a hardened public service. Use public samples only. Do not upload patient data, add hosted model credentials, or expose the server publicly during this phase.

## Start as a first-time visitor

When the operator enables the private self-registration pilot, open the supplied private URL in a fresh incognito/private browser window. Do not use an administrator session for this test.

1. Choose **Create demo account**, enter a new username and a password of at least 12 characters, confirm it, and submit. Existing visitors use **Already have an account? Sign in** instead.
2. Check that **My demo** opens and the one-hour countdown is visible. You should not see the administrator's projects or another visitor's work.
3. Click **Load one sample CT**. This imports just one prepared public Spleen CT, without starting a dataset download. Open it in OHIF or Slicer from Datasets, choose **TotalSegmentator CT**, and click **Segment all 117 labels** without choosing a single target. Inspect several slices and organs. Then try **Delete the current segmentation and run TotalSegmentator CT on all labels.** Prepared results may be reused; not every structure is visible in every scan. To test another model, ask **Segment the spleen using VISTA3D.** Inspect and correct results before submitting them.
4. Open **Models → Training → Recorded examples** to inspect the prepared runs. These are not training jobs performed on your corrections.
5. Refresh, sign out, and sign back in: your original deadline and your project must remain. A second private browser profile can register independently. When all reservations are occupied, another visitor should receive a busy/retry message, not an invalid-password message or a partial account. Browser-desktop slots have a separate limit; closing a Slicer or QuPath session frees a desktop slot, not the visitor's one-hour reservation.
6. After expiry, the same account cannot sign in again. Work is retained privately for the host during this pilot, not automatically deleted or shown to the next visitor. Unfinished jobs/desktops still occupy capacity until they finish or the host closes them.

This checks the first-time visitor experience while access remains private. It does not establish safe public hosting, complete process isolation, automatic cleanup or a measured GPU concurrency limit.

## Test administrator-created invitations (optional)

When the operator has enabled the [private demo controls](demo-controls.md), use this checklist before the annotation examples below:

1. Sign in with the existing administrator account. Open **Team & roles → Create demo visitor** (not **Create user**), choose a fresh username/password, and select one hour. Administrators remain untimed; their banner displays the public-data reminder but no countdown. The separate pre-login recording notice remains.
2. Open the same private address in an incognito/private browser window and sign in as that visitor. The countdown starts at first sign-in. Refresh, then sign out and back in: the original deadline must remain unchanged. The visitor starts without access to the administrator's projects.
3. Create a separate test project. Inspect the dataset-template menu: only prepared datasets should appear. Ask for a known template absent from that menu, for example Decathlon Pancreas if the host has not prepared it. The request must not start a download; the assistant or API should explain the restriction and available choices. A completed Liver download may already be cached and therefore is not necessarily a negative test.
4. Run the three Spleen prompts in the radiology example below, one at a time. Inspect the result in OHIF, then try Slicer on another sample. Use small imports and one job at a time; comprehensive resource quotas and automatic cleanup are still pending.
5. In OHIF or Slicer, select **TotalSegmentator CT** and click **Segment all 117 labels**, leaving the single target empty. Inspect several organs; empty structures are expected outside the scan's field of view. Then try `Delete the current segmentation and run TotalSegmentator CT on all labels.` This specific replacement is one operation that preserves existing work until a successful result; arbitrary multi-action chaining remains unsupported. Shared prepared predictions may be reused for unchanged public scans; do not interpret their response time as fresh model inference speed.
6. Before the countdown ends, save or submit any work you want to inspect. With `MONAILABEL_DEMO_LIFECYCLE=1`, expiry refuses further API work/sign-in, cancels owned jobs, terminates isolated local inference, closes browser desktops and revokes managed CVAT bindings independently of whether a countdown is visible. Check the explicit limits for already-started CVAT decoding, CPU/provisioning stages and in-flight coordinator replies in [demo controls](demo-controls.md#expiry-and-an-explicit-fresh-demo). Cleanup failures retain capacity; this check alone is not evidence of public readiness.
7. In the administrator window, open `/api/demo-access/audit-status` to check that recording is enabled and the written-event count increases. Review dropped-event/write-error counters too. This endpoint shows telemetry health, not recorded prompts. Activity records remain in the host's private audit directory, and coverage inside native viewers is incomplete.

After the private trial, complete resource limits, bounded cleanup for remaining stages, model/download restrictions, viewer isolation and private reviewer validation before configuring network isolation, HTTPS and public admission. See the remaining gates in [private demo controls](demo-controls.md).

To test reset, use a disposable workspace with an administrator and disposable visitors. As the administrator, open **Team & roles → Start a fresh demo**, check the counts, and cancel first: nothing should change. Once all work and desktops have stopped, reopen it and type `START FRESH DEMO` to confirm. The administrator must stay signed in; old visitor logins must fail, and the same username may register again with a different visit ID. Shared caches, prepared models/examples and feedback must remain. Do not confirm this operation against existing visitor data without the host's explicit approval. Restarting the server by itself must preserve accounts.

## Start in the browser

Use your visitor account at the private URL supplied by the operator and wait for **Assistant ready**. Only the operator performs initial administrator setup. Nemotron interprets requests; separate local models create segmentations. Nemotron is not a replacement for a vision model that can automatically interpret arbitrary pathology or video frames.

Send prompts one at a time and wait for each job. Open a viewer from **Datasets** to annotate and from **Reviews** to review. Slicer and QuPath appear in browser tabs in this installation; allow popups. If needed, open `/datasets?desktop=browser`. The first CVAT launch creates its private services and may take longer than later launches.

## 1. Radiology: reach an editable annotation

In the workspace assistant:

```text
Create a project called "Private radiology test".

Import 3 Decathlon Spleen images without labels.

Annotate spleen in 1 image with VISTA3D; submit for review.
```

Check the job status and open the pending annotation in OHIF. Scroll through the volume, inspect the mask, make a small manual correction, and review it. This is the fastest test of the complete browser annotation path.

For a Slicer test, open an unannotated CT from Datasets. In its MONAI Label panel:

```text
Start nnInteractive for spleen.
```

Add a positive point inside the spleen and, if useful, a negative point outside it. Use **Update volume**. Move a point and update again; verify other labels remain unchanged. Submit only after checking the result, then open that submission through Reviews.

Try VISTA3D or TotalSegmentator CT for whole-volume anatomical segmentation, and MedSAM2 for prompted propagation. Do not use the README's GPT Astra correction prompt unless you later explicitly enable a suitable hosted annotation connection.

## 2. Pathology: edit one object in QuPath

Create a separate pathology project and import a small TNBC nuclei sample through Datasets. Open it in QuPath, choose **SAM 2.1**, and set the target label to Nucleus. Add points or a box around one nucleus, then choose **Update image** or **Update region**. Inspect the resulting object, correct it, save the QuPath draft, and submit for review.

Local SAM annotates one prompted object at a time. Automatic nuclei detection/classification through GPT, Claude or Gemini is not enabled by installing Nemotron. Keep that distinction when judging this demo.

## 3. Endoscopy: propagate an editable tool annotation

Create a separate video project and import **HyperKvasir tool tracking**. Open the clip in CVAT. Choose SAM 2.1, a target such as Snare, and **New object**. Draw a box or place points around the visible tool, then use **Update frame**. Track a short range, such as 16 frames, and inspect every resulting frame.

You can also select a manually drawn tool annotation and ask:

```text
Track the selected tool for 16 frames.
```

Save a draft, reopen it, and verify it is preserved. Submit for review only after correcting the track. Automatic “find the tool” requests that need a vision-language model are separate from local SAM propagation.

## 4. Explore training without starting a GPU job

When the operator enables [the training preview](demo-training.md), open **Models → Training**. The **nnU-Net training** and **VISTA3D fine-tuning** cards appear under **Recorded examples**, above Training setups. Inspect their summaries, then expand **View recorded log and run details** on either card. Each example must say that it was completed before your visit and did not use your corrections. An example the host has not prepared must report that clearly rather than show invented progress.

Create a training setup for your project, then choose **Preview training**. Expand the steps to see what a full installation would do. This preview neither validates training eligibility nor creates a snapshot or job. Ask the assistant to train, fine-tune or continue a model: it should explain the preview restriction, without starting training. Existing annotations and model weights must remain unchanged.

Normal installations without the preview flag still support live training. Test that separately with a small, reviewed dataset and an explicit resource budget. A run without independent evaluation data must not claim a held-out score; a short demo run does not establish improved clinical accuracy.

## Useful edge cases and feedback

- Refresh a viewer tab with a saved draft and reopen a submitted annotation.
- As a timed visitor, compare the countdown in the dashboard, OHIF assistant panel, Slicer/QuPath browser wrappers and managed CVAT page. They use one account deadline, not separate viewer timers. Reloading or opening another viewer must not grant another hour. Save or submit before expiry; the timer does not save drafts automatically.
- Change a point or mask while inference is running; a stale result should not replace the newer edit.
- Switch labels and models; existing annotations should survive.
- Ask for an unavailable hosted model; the assistant should explain the missing configuration rather than pretend to annotate.
- Open two browser desktops. Save before closing the last tab: it ends the desktop shortly afterward, and unsaved native state is not durable.

For each issue, record the exact prompt or button, viewer, public sample name, expected result, actual result, job ID and a screenshot. Include rough waiting time and the visit ID/timestamp when available; a reused username alone is not a unique identity. Do not include passwords, connection secrets or patient data. Physical tablet, voice, internet access, anonymous-user isolation and universal resource-termination bounds are separate tests, not implied by these private checks.

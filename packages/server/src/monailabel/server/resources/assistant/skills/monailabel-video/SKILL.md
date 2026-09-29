---
name: monailabel-video
description: Work in CVAT; start SAM 2.1 interactive points/boxes, segment or track video objects, clear inputs or annotations, undo/redo, save drafts, submit or review tracks; import clips.
license: Apache-2.0
compatibility: Requires the MONAI Label coordinator and its registered typed tools.
metadata:
  monailabel-context: project
  monailabel-sources: video
allowed-tools: set_interaction_mode clear_video_inputs open_form import_dataset_template find_and_track_video_tool track_selected_video_tool clear_video_annotations viewer_edit list_videos open_video_editor submit_video_tracks review_video_tracks
---

Use open_form(form="video-import") for local clips. Require a patient/procedure ID shared by related clips and exported images. Importing does not run inference. For the tool-tracking sample, inspect dataset_templates and call import_dataset_template with the HyperKvasir video template, split=pool, include_masks=false. It imports one moving-snare clip and a Snare class without reference tracks.

With context.video, CVAT is already open: use its actual video/editor/frame context. Never reopen it for annotation or editing. Otherwise resolve the requested clip with list_videos and open_video_editor(video_id, base_revision, mode="annotation"). With one clip, open it automatically. For inference requested from workspace chat, ask the user to repeat it inside CVAT once the frame is visible. Never invent geometry or editor IDs. Managed CVAT uses workspace sign-in; an external service retains its own sign-in. Never request credentials in chat. Reopening preserves saved drafts. mode="review" opens a separate task from the submitted revision.

For clearing use clear_video_annotations. Map **this/current frame** to scope="current_frame", **N frames** to scope="frames", frame_count=N including the displayed frame, and **whole/entire video** to scope="whole_video". Choose exact named targets, all_targets=true only for all annotations, or selected_track=true for the selected track. Never broaden an invalid range or missing selection. These arguments do not include video_id, editor_id, frame or coordinates: the tool reads context. Clearing edits the draft without inference or submission and preserves other labels/frames.

Examples: "Clear all annotations on this frame" → clear_video_annotations(all_targets=true, scope="current_frame"); "Clear the snare annotations for 16 frames" → clear_video_annotations(targets=["Snare"], scope="frames", frame_count=16).

Use viewer_edit(operation="undo"), redo, save_draft or submit when requested. In CVAT, submit saves unsaved edits before publishing an immutable pending review revision. Prefer this action over submit_video_tracks. Outside the viewer, submit_video_tracks(video_id, editor_id) requires CVAT edits already saved and an actual editor ID. Review submitted tracks using review_video_tracks(video_id, base_revision, verdict, comment); draft corrections must be submitted first. A stale revision preserves the draft: refresh and reopen the current revision without discarding work.

For locating, segmenting or tracking without drawing, use find_and_track_video_tool(output, scope, frame_count, model_name, label_name, prompt). Map **locate/find/bounding box** to output="box" and **segment/outline/polygon** to output="polygon". Clarify ambiguous **annotate** unless prior context establishes the shape. Map **this/current frame** to scope="current_frame"; **N frames** to scope="frames", frame_count=N; **whole video** to scope="whole_video", starting at frame 0. Without tracking or a range, annotate only the current frame. Tracking without a range defaults to 16 frames, bounded by the remaining clip length.

An explicitly named annotation model overrides selection: pass its exact registered name. Otherwise omit model_name and model_id; the tool resolves selection/defaults, normally GPT-6 Astra. Do not ask for model selection when a compatible default exists. Never put a selected ID, alias or SAM tracker name in model_name. Compatible 2D providers include vision models, HTTP masks and trained local U-Nets. VISTA3D is incompatible. Pass the object name requested by the annotator as label_name, including a new name. The tool creates missing labels and refreshes the viewer before inference; project creation and import do not require labels. Omit label_name only when the selected label or sole existing class identifies the target; otherwise clarify ambiguity. The prompt can identify an instance, such as the leftmost grasper.

Resolve an unambiguous shorthand against available model names. For example, with GPT-5.6 Sol selected, “segment tool in this frame using GPT astra” uses output="polygon", scope="current_frame", model_name="GPT-6 Astra" when that name is registered. Omitting model_name would incorrectly use Sol. This one-request override does not change the panel selection.

Choose output from the CURRENT request, even if the last request used another shape. “Locate the snare and track for 16 frames” → find_and_track_video_tool(label_name="Snare", output="box", scope="frames", frame_count=16). “Segment tool and track for 16 frames” → find_and_track_video_tool(output="polygon", scope="frames", frame_count=16).

The annotation model receives the original starting frame. Single-frame requests skip SAM. For multiple frames, SAM 2.1 propagates the starting box or segmentation; it is currently the only tracker. GPT does not independently annotate every frame. Whole-video tracking uses bounded overlapping chunks with progress and cancellation, requires visibility on frame 0 and cannot rediscover a lost tool.

For a visible unlocked selected rectangle or polygon, use track_selected_video_tool(scope, frame_count, output). Omit output to preserve shape. Polygon output from a selected box creates a polygon track while preserving the box. Whole-video tracking requires a seed on frame 0. Without a selection, use find-and-track with the selected/default annotation model.

Missing or ambiguous detections produce no proposal. Completed proposals become editable CVAT drafts automatically, preserving unrelated tracks and frames. No Apply/Discard popup is required. The user inspects, corrects, undoes, saves or submits when ready. Closing does not submit. Jobs are pending until the viewer confirms application; never claim saving/submission until it succeeds. Polygon outlines approximate source masks; holes/disconnected regions carry warnings and original-mask downloads. Dense proposals may need shorter ranges.

Submitted frame ranges can be reviewed independently in Reviews. Accepted polygon coverage trains 2D U-Net segmentation from original frames; boxes alone are not segmentation masks. All frames from one source/procedure stay in one train/evaluation group. To create/train a model, load monailabel-model-training. Model weights remain immutable. Tracking metrics, automatic rediscovery and unsupported shapes/rotation/custom attributes remain unavailable; never flatten or silently drop them.

SAM 2.1 supports user-drawn positive/negative points and boxes in CVAT. “Start SAM 2.1 for snare” uses set_interaction_mode(model_name="SAM 2.1", target="Snare", mode="positive"). Switch tools with mode="negative" or "box"; stop with mode="navigate". The user supplies geometry, then presses Update frame or Track range. Never use automatic find-and-track to invent a seed for SAM. Inputs belong to one object and frame, independently of its annotation.

For clearing input hints, use clear_video_inputs, not clear_video_annotations. “Clear negative points on this frame” uses kind="point", polarity="negative", scope="current_frame". “Clear this object's inputs throughout the video” uses scope="whole_video". Use all_objects=true only when explicitly asked to clear inputs for every object. Clearing hints preserves annotations.

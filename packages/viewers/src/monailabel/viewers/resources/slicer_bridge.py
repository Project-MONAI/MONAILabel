"""Run inside 3D Slicer with --python-script. Uses Slicer's bundled Python only.

Network work runs off the UI thread. Volume and segmentation operations run on the
Qt thread. The API uses NIfTI IJK array order; Slicer numpy arrays use KJI.
"""

import contextlib
import hashlib
import html
import importlib
import json
import os
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import qt
import slicer
import vtk

sys.path.insert(0, str(Path(__file__).parent))
import geometry  # noqa: E402
import spatial_hints  # noqa: E402
import volume_files  # noqa: E402

# Re-executing this bridge upgrades an open scene, including its geometry helpers.
geometry = importlib.reload(geometry)
volume_files = importlib.reload(volume_files)
spatial_hints = importlib.reload(spatial_hints)
merge_proposal, roi_geometry, source_slice = (
    geometry.merge_proposal,
    geometry.roi_geometry,
    geometry.source_slice,
)


class AnnotationDock:
    review_mode = False
    shared_filesystem = False
    loading = False

    def __init__(self, settings):
        self.url = settings["url"].rstrip("/")
        self.project_id = settings["project_id"]
        self.token = settings.get("token")
        self.initial_asset = settings.get("asset_id")
        self.review_mode = settings.get("mode") == "review"
        self.shared_filesystem = settings.get("shared_filesystem", False)
        self.can_annotate = False
        self.can_review = False
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.future = None
        self.callback = None
        self.job_id = None
        self.cancel_requested = False
        self.project = None
        self.asset = None
        self.volume = None
        self.segmentation = None
        self.context = {}
        self.conversation_id = None
        self.conversation_asset_id = None
        self.proposal_id = None
        self.region_nodes = []
        self.before_prediction = None
        self.temporary = tempfile.TemporaryDirectory(prefix="monailabel-slicer-")

        self.build_ui()
        self.timer = qt.QTimer()
        self.timer.setInterval(200)
        self.timer.timeout.connect(self.tick)
        self.timer.start()
        self.refresh()

    def build_ui(self):
        """Keep conversation space independent of the manual segmentation tools."""
        main = slicer.util.mainWindow()
        self.dock = qt.QDockWidget("MONAI Label · Annotation assistant", main)
        self.dock.setObjectName("MONAILabelAssistantDock")
        container = qt.QWidget()
        container.setMinimumWidth(340)
        self.layout = qt.QVBoxLayout(container)
        self.title = qt.QLabel("Connecting to project…")
        self.title.setWordWrap(True)
        self.layout.addWidget(self.title)

        toolbar = qt.QHBoxLayout()
        self.edit_button = qt.QPushButton("Manual editing")
        self.edit_button.setCheckable(True)
        toolbar.addWidget(self.edit_button)
        self.pop_button = qt.QPushButton("Pop out")
        self.pop_button.clicked.connect(self.toggle_floating)
        toolbar.addWidget(self.pop_button)
        self.layout.addLayout(toolbar)

        self.sample_options = qt.QPushButton("Sample and connection ▸")
        self.sample_options.setCheckable(True)
        self.layout.addWidget(self.sample_options)
        self.sample_panel = qt.QWidget()
        sample_layout = qt.QVBoxLayout(self.sample_panel)
        sample_layout.setContentsMargins(0, 0, 0, 0)
        self.assets = qt.QComboBox()
        sample_layout.addWidget(self.assets)
        self.load_button = qt.QPushButton("Open selected NIfTI")
        self.load_button.clicked.connect(self.open_selected)
        sample_layout.addWidget(self.load_button)
        self.refresh_button = qt.QPushButton("Refresh datasets and models")
        self.refresh_button.clicked.connect(self.refresh)
        sample_layout.addWidget(self.refresh_button)
        self.reload_button = qt.QPushButton("Reload current sample · discard local edits")
        self.reload_button.clicked.connect(self.open_selected)
        sample_layout.addWidget(self.reload_button)
        self.layout.addWidget(self.sample_panel)
        self.sample_panel.hide()
        self.sample_options.toggled.connect(self.sample_panel.setVisible)
        self.sample_name = qt.QLabel("No sample open")
        self.sample_name.setWordWrap(True)
        self.layout.addWidget(self.sample_name)

        selectors = qt.QFormLayout()
        self.models = qt.QComboBox()
        self.models.setToolTip("Used unless your prompt names another registered model")
        selectors.addRow("Model", self.models)
        self.views = qt.QComboBox()
        self.views.addItems(["Red", "Yellow", "Green"])
        self.views.setToolTip("View used by 'annotate this slice' and 'annotate all slices'")
        selectors.addRow("Slice view", self.views)
        self.spatial_hint = slicer.qMRMLNodeComboBox()
        self.spatial_hint.nodeTypes = [
            "vtkMRMLMarkupsROINode",
            "vtkMRMLMarkupsClosedCurveNode",
            "vtkMRMLMarkupsFiducialNode",
        ]
        self.spatial_hint.noneEnabled = True
        self.spatial_hint.addEnabled = False
        self.spatial_hint.removeEnabled = False
        self.spatial_hint.setMRMLScene(slicer.mrmlScene)
        self.spatial_hint.setToolTip(
            "SAM combines this target’s box and points on the current slice. Use chat or Markups; "
            "prefix an exclusion point's label with '-' or 'negative'."
        )
        selectors.addRow("SAM hint", self.spatial_hint)
        self.layout.addLayout(selectors)

        self.conversation = qt.QSplitter(qt.Qt.Vertical)
        self.conversation.setChildrenCollapsible(False)
        self.history = qt.QTextBrowser()
        self.history.setMinimumHeight(160)
        self.history.setStyleSheet("QTextBrowser { padding: 8px; font-size: 13px; }")
        self.conversation.addWidget(self.history)
        self.prompt = qt.QPlainTextEdit()
        self.prompt.setMinimumHeight(90)
        self.prompt.setStyleSheet("QPlainTextEdit { padding: 6px; font-size: 13px; }")
        self.conversation.addWidget(self.prompt)
        self.conversation.setStretchFactor(0, 4)
        self.conversation.setStretchFactor(1, 1)
        self.conversation.setSizes([440, 120])
        self.layout.addWidget(self.conversation, 1)
        self.shortcut = qt.QShortcut(qt.QKeySequence("Ctrl+Return"), self.prompt)
        self.shortcut.setContext(qt.Qt.WidgetShortcut)
        self.shortcut.activated.connect(self.send)
        self.layout.addWidget(
            qt.QLabel("Enter: new line · Ctrl+Enter: send · Drag divider to resize")
        )
        actions = qt.QHBoxLayout()
        self.send_button = qt.QPushButton("Send prompt")
        self.send_button.clicked.connect(self.send)
        actions.addWidget(self.send_button)
        self.cancel_button = qt.QPushButton("Cancel annotation")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel)
        actions.addWidget(self.cancel_button)
        self.status = qt.QLabel()
        self.status.setToolTip("Current operation")
        self.status.hide()
        actions.addWidget(self.status)
        self.layout.addLayout(actions)

        self.editor_dock = qt.QDockWidget("MONAI Label · Manual editing", main)
        self.editor_dock.setObjectName("MONAILabelEditorDock")
        self.editor = slicer.qMRMLSegmentEditorWidget()
        self.editor.setMRMLScene(slicer.mrmlScene)
        self.editor_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentEditorNode")
        self.editor.setMRMLSegmentEditorNode(self.editor_node)
        scroll = qt.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.editor)
        self.editor_dock.setWidget(scroll)
        main.addDockWidget(qt.Qt.LeftDockWidgetArea, self.editor_dock)
        self.editor_dock.hide()
        self.edit_button.toggled.connect(self.editor_dock.setVisible)
        self.editor_dock.visibilityChanged.connect(self.edit_button.setChecked)
        self.save_button = qt.QPushButton("Submit complete annotation for review")
        self.save_button.clicked.connect(self.save)
        self.layout.addWidget(self.save_button)
        self.review_comment = qt.QLineEdit()
        self.review_comment.setPlaceholderText("Review comment (optional)")
        self.review_comment.setObjectName("monailabel-review-comment")
        self.layout.addWidget(self.review_comment)
        review_row = qt.QHBoxLayout()
        self.good_button = qt.QPushButton("Good")
        self.bad_button = qt.QPushButton("Bad / needs changes")
        self.corrected_button = qt.QPushButton("Good with corrections")
        self.good_button.setObjectName("monailabel-review-good")
        self.bad_button.setObjectName("monailabel-review-bad")
        self.corrected_button.setObjectName("monailabel-review-corrected")
        self.good_button.clicked.connect(lambda: self.review_decision("accepted"))
        self.bad_button.clicked.connect(lambda: self.review_decision("changes_requested"))
        self.corrected_button.clicked.connect(lambda: self.review_decision("accepted"))
        for button in (self.good_button, self.bad_button, self.corrected_button):
            review_row.addWidget(button)
        self.layout.addLayout(review_row)

        self.dock.setWidget(container)
        main.addDockWidget(qt.Qt.RightDockWidgetArea, self.dock)
        main.resizeDocks([self.dock], [440], qt.Qt.Horizontal)
        self.dock.topLevelChanged.connect(
            lambda floating: self.pop_button.setText("Dock panel" if floating else "Pop out")
        )
        self.show_welcome()

    def show_welcome(self):
        """Give an empty conversation a starting point without replacing chat history."""
        self.prompt.setPlaceholderText(
            "Ask about your review…\nFor example: accept my corrected segmentation"
            if self.review_mode
            else "Describe your annotation…\nFor example: annotate spleen on this slice"
        )
        if self.history.toPlainText().strip():
            return
        if self.review_mode:
            guidance = (
                "<p>You’re in <b>review mode</b>. Inspect the segmentation and use "
                "<b>Manual editing</b> to make any corrections.</p>"
                "<p>When ready, choose <b>Good</b>, <b>Good with corrections</b>, "
                "or <b>Bad / needs changes</b> below.</p>"
                "<p><b>Try a review prompt</b></p>"
                "<ul><li>Accept my corrected segmentation</li>"
                "<li>Mark this annotation as needs changes</li></ul>"
                "<p>Your review is saved only when you submit a decision.</p>"
            )
        else:
            guidance = (
                "<p>Describe what you want to annotate. I can help create "
                "editable segmentations, bounding boxes and regions.</p>"
                "<p><b>Try a prompt</b></p>"
                "<ul><li>Annotate spleen on this slice</li>"
                "<li>Annotate liver on all slices</li>"
                "<li>Create a bounding box for spleen on this slice using GPT Astra</li></ul>"
                "<p>Segmentation uses the selected model. To locate anatomy without coordinates, "
                "name a "
                "vision model such as GPT Astra or Claude.</p>"
                "<p>Inspect and correct the result, then submit the complete "
                "annotation for review.</p>"
            )
        self.history.setHtml("<h3>Welcome to MONAI Label</h3>" + guidance + "<hr>")

    def append_message(self, role, message):
        """Append a plain chat paragraph without inheriting selected welcome formatting."""
        cursor = qt.QTextCursor(self.history.document)
        cursor.movePosition(qt.QTextCursor.End)
        paragraph = qt.QTextBlockFormat()
        paragraph.setTopMargin(8)
        cursor.insertBlock(paragraph, qt.QTextCharFormat())
        text = html.escape(message).replace("\n", "<br>")
        cursor.insertHtml(f"<b>{html.escape(role)}</b>: {text}")
        self.history.setTextCursor(cursor)
        self.history.ensureCursorVisible()

    def toggle_floating(self):
        floating = not self.dock.isFloating()
        self.dock.setFloating(floating)
        if floating:
            self.dock.resize(560, 800)
        self.dock.show()
        self.prompt.setFocus()

    def request(self, path, body=None, raw=False, headers=None):
        binary = isinstance(body, bytes)
        data = body if binary else json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(
            self.url + path,
            data=data,
            headers={
                "Content-Type": "application/octet-stream" if binary else "application/json",
                "Authorization": f"Bearer {self.token}",
                **(headers or {}),
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                result = response.read()
        except urllib.error.HTTPError as error:
            try:
                detail = json.loads(error.read()).get("detail", f"HTTP {error.code}")
            except ValueError:
                detail = f"HTTP {error.code}"
            raise RuntimeError(str(detail)) from error
        return result if raw else json.loads(result)

    def request_mask(self, path, shape):
        content = self.request(path + ".bin", raw=True)
        if len(content) != int(np.prod(shape)):
            raise RuntimeError("Returned mask dimensions do not match the source volume.")
        return np.frombuffer(content, dtype=np.uint8).reshape(shape)

    def report_error(self, error):
        text = str(error)
        self.append_message("Assistant · error", text)

    def cancel(self):
        if self.job_id:
            self.cancel_requested = True
            self.cancel_button.setEnabled(False)
            self.status.setText("Cancelling…")

    def run(self, work, callback):
        if self.future:
            return
        if not self.job_id:
            self.status.setText("Opening…" if self.loading else "Working…")
        self.set_busy(True)
        self.callback = callback
        self.future = self.executor.submit(work)

    def set_busy(self, busy):
        busy = busy or self.loading
        self.status.setVisible(busy)
        for widget in (self.load_button, self.send_button, self.refresh_button, self.reload_button):
            widget.setEnabled(not busy)
        self.save_button.setEnabled(not busy and self.can_annotate)
        self.save_button.setVisible(self.can_annotate and not self.review_mode)
        review_ready = (
            not busy and self.can_review and bool(self.asset and self.asset.get("annotation_id"))
        )
        for button in (self.good_button, self.bad_button, self.corrected_button):
            button.setEnabled(review_ready)
            button.setVisible(self.can_review)
        self.review_comment.setVisible(self.can_review)
        self.cancel_button.setEnabled(bool(self.job_id) and not self.cancel_requested)

    def tick(self):
        if self.future and not self.future.done():
            # Slicer's Qt loop otherwise holds the GIL between Python callbacks.
            # Yield briefly so background I/O can finish while the UI stays responsive.
            time.sleep(0.001)
        if self.future and self.future.done():
            future, callback = self.future, self.callback
            self.future = None
            try:
                callback(future.result())
            except Exception as error:
                self.loading = False
                self.job_id = None
                self.report_error(error)
            self.set_busy(bool(self.future or self.job_id))
        if self.job_id and self.future is None:
            if self.cancel_requested:
                self.cancel_requested = False
                self.run(
                    lambda: self.request(f"/api/jobs/{self.job_id}/cancel", {}), self.job_updated
                )
            else:
                self.run(lambda: self.request(f"/api/jobs/{self.job_id}"), self.job_updated)

    def refresh(self):
        if self.future or self.job_id:
            return
        self.run(
            lambda: (
                self.request(f"/api/projects/{self.project_id}"),
                self.request(f"/api/projects/{self.project_id}/assets"),
                self.request(f"/api/projects/{self.project_id}/models"),
                self.request(f"/api/projects/{self.project_id}/permissions"),
                self.request("/api/auth/me"),
            ),
            self.refreshed,
        )

    def refreshed(self, result):
        self.project, assets, models, permissions, user = result
        if self.segmentation and self.segmentation.GetScene():
            self.sync_labels(self.segmentation)
        self.can_annotate = bool(set(permissions["roles"]) & {"annotator", "manager"})
        self.can_review = bool(set(permissions["roles"]) & {"reviewer", "manager"})
        self.editor.setEnabled(self.can_annotate or self.can_review)
        mode = "Review" if self.review_mode else "Annotation"
        self.title.setText(f"{self.project['name']} · {mode} · {user['username']}")
        selected_asset = self.assets.currentData
        selected_model = self.models.currentData
        self.assets.clear()
        for asset in assets:
            if asset["kind"] == "volume3d":
                self.assets.addItem(f"{asset['name']} · {asset['split']}", asset["id"])
        self.models.clear()
        self.models.addItem("Project defaults", "")
        for model in models:
            self.models.addItem(model["name"], model["id"])
        asset_index = self.assets.findData(selected_asset)
        if asset_index >= 0:
            self.assets.setCurrentIndex(asset_index)
        model_index = self.models.findData(selected_model)
        if model_index >= 0:
            self.models.setCurrentIndex(model_index)
        if not models:
            self.append_message(
                "Assistant",
                "No annotation model is connected. Configure a model in the "
                "web console's Models page, set defaults, then refresh here.",
            )
        if self.initial_asset:
            index = self.assets.findData(self.initial_asset)
            self.initial_asset = None
            if index >= 0:
                self.assets.setCurrentIndex(index)
                self.open_selected()

    def open_selected(self):
        if self.future or self.job_id or self.loading:
            return
        identifier = self.assets.currentData
        if not identifier:
            return
        self.loading = True

        def load():
            files = (
                self.request(f"/api/assets/{identifier}/viewer-files")
                if self.shared_filesystem
                else {}
            )
            asset = files.get("asset") or self.request(f"/api/assets/{identifier}")
            directory = Path(self.temporary.name)
            path = volume_files.local_nifti(files.get("image_path"), directory)
            if path is None:
                # Asset identity also separates equal pixels with different image geometry.
                path = directory / f"{asset['id']}.nii"
                if not path.is_file():
                    content = self.request(f"/api/assets/{identifier}/image", raw=True)
                    path.write_bytes(content)
            mask = None
            if asset["annotation_id"]:
                if files.get("mask_path"):
                    with contextlib.suppress(OSError):
                        mask = np.load(files["mask_path"], mmap_mode="r", allow_pickle=False)
                if mask is None:
                    mask = self.request_mask(
                        f"/api/annotations/{asset['annotation_id']}/mask", asset["spatial_shape"]
                    )
                if mask.dtype != np.uint8 or tuple(mask.shape) != tuple(asset["spatial_shape"]):
                    raise RuntimeError("Saved mask does not match the source volume.")
            return asset, str(path), mask

        def loaded(result):
            try:
                self.opened(result)
            finally:
                self.loading = False

        self.run(load, loaded)

    def opened(self, result):
        asset, path, mask = result
        volume = None
        segmentation = None
        try:
            volume = slicer.util.loadVolume(path)
            if not volume:
                raise RuntimeError(
                    "Slicer could not load the NIfTI volume. Try reopening the sample."
                )
            name = volume_files.volume_name(asset["name"])
            volume.SetName(name)
            # Native Save must ask for a destination instead of overwriting our input cache.
            volume.GetStorageNode().SetFileName("")
            self.reverses_slices(volume, asset)
            segmentation = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
            segmentation.SetName(f"{name} segmentation")
            segmentation.CreateDefaultDisplayNodes()
            segmentation.SetReferenceImageGeometryParameterFromVolumeNode(volume)
            self.sync_labels(segmentation)
            if mask is not None:
                self.write_mask(mask, asset, volume, segmentation)
        except Exception:
            for node in (segmentation, volume):
                if node:
                    slicer.mrmlScene.RemoveNode(node)
            raise
        # Keep the previous case and its edits until the replacement is valid.
        for node in (*self.region_nodes, self.segmentation, self.volume):
            if node:
                slicer.mrmlScene.RemoveNode(node)
        self.region_nodes = []
        self.asset, self.volume, self.segmentation = asset, volume, segmentation
        self.editor.setSegmentationNode(self.segmentation)
        self.editor.setSourceVolumeNode(self.volume)
        self.editor.setUndoEnabled(True)
        self.editor.setMaximumNumberOfUndoStates(5)
        slicer.util.setSliceViewerLayers(background=self.volume)
        slicer.util.resetSliceViews()
        self.context["asset_id"] = self.asset["id"]
        self.proposal_id = None
        self.sample_name.setText(f"{name} · revision {self.asset['revision']}")

    @staticmethod
    def reverses_slices(volume, asset):
        if slicer.util.arrayFromVolume(volume).shape != tuple(reversed(asset["spatial_shape"])):
            raise RuntimeError("Loaded volume geometry differs from the source voxel grid.")
        matrix = vtk.vtkMatrix4x4()
        volume.GetIJKToRASMatrix(matrix)
        return geometry.slicer_reverses_slices(
            np.asarray(asset["affine"]),
            slicer.util.arrayFromVTKMatrix(matrix),
            asset["spatial_shape"],
        )

    def source_matrix(self, inverse=False):
        """Map prompt coordinates through the immutable source grid, not display IJK."""
        affine = np.asarray(self.asset["affine"])
        return slicer.util.vtkMatrixFromArray(np.linalg.inv(affine) if inverse else affine)

    def current_mask(self):
        if self.loading:
            raise RuntimeError("Wait for the image and annotation to finish loading.")
        if not self.asset or not self.volume or not self.segmentation:
            raise RuntimeError("Open a sample before annotating or submitting a review.")
        if not self.volume.GetScene() or not self.segmentation.GetScene():
            raise RuntimeError("The image or segmentation was removed. Reopen the sample.")
        if self.volume.GetParentTransformNode() or self.segmentation.GetParentTransformNode():
            raise RuntimeError(
                "Harden or remove transforms before saving to the source voxel grid."
            )
        reverse = self.reverses_slices(self.volume, self.asset)
        mask = np.zeros(self.asset["spatial_shape"], dtype=np.uint8)
        for label in self.project["labels"]:
            if label["id"] == 0:
                continue
            kji = slicer.util.arrayFromSegmentBinaryLabelmap(
                self.segmentation, str(label["id"]), self.volume
            )
            if reverse:
                kji = kji[::-1]
            selected = kji.transpose(2, 1, 0) > 0
            if np.any(selected & (mask != 0)):
                raise RuntimeError(
                    "This integration requires mutually exclusive segmentation labels."
                )
            mask[selected] = label["id"]
        return mask

    def sync_labels(self, node):
        """Keep project display colors and IDs consistent without replacing mask data."""
        segments = node.GetSegmentation()
        for label in self.project["labels"]:
            if not label["id"]:
                continue
            identifier = str(label["id"])
            color = tuple(int(label["color"][i : i + 2], 16) / 255 for i in (1, 3, 5))
            segment = segments.GetSegment(identifier)
            if segment:
                segment.SetColor(color)
            else:
                segments.AddEmptySegment(identifier, label["name"], color)

    def apply_mask(self, values):
        self.write_mask(values, self.asset, self.volume, self.segmentation)

    def write_mask(self, values, asset, volume, segmentation):
        mask = np.asarray(values, dtype=np.uint8)
        if tuple(mask.shape) != tuple(asset["spatial_shape"]):
            raise RuntimeError("Mask shape differs from the loaded volume.")
        if self.reverses_slices(volume, asset):
            mask = mask[:, :, ::-1]
        for label in self.project["labels"]:
            if label["id"]:
                slicer.util.updateSegmentBinaryLabelmapFromArray(
                    (mask == label["id"]).astype(np.uint8).transpose(2, 1, 0),
                    segmentation,
                    str(label["id"]),
                    volume,
                )
        segmentation.GetDisplayNode().SetVisibility(True)
        segmentation.GetDisplayNode().SetVisibility2D(True)
        segmentation.GetDisplayNode().SetOpacity2DFill(0.25)
        segmentation.GetDisplayNode().SetOpacity2DOutline(1.0)

    def send(self):
        if self.loading:
            self.append_message("Assistant", "Wait for the image to finish loading.")
            return
        if self.future or self.job_id:
            return
        try:
            self.send_prompt()
        except Exception as error:
            self.report_error(error)

    def send_prompt(self):
        message = self.prompt.toPlainText().strip()
        if not message:
            return
        self.append_message("You", message)
        if self.asset:
            self.before_prediction = hashlib.sha256(self.current_mask().tobytes()).hexdigest()
        context = dict(self.context)
        context["viewer_actions"] = (
            ["remove_regions", "roi", "box", "edit_spatial_prompts"]
            + (["clear_segments", "undo", "redo"] if self.can_annotate or self.can_review else [])
            + (["submit"] if self.can_annotate and not self.review_mode else [])
            + (["review_annotation"] if self.can_review else [])
        )
        if self.asset:
            context["base_revision"] = self.asset["revision"]
        if context.get("asset_id") != self.conversation_asset_id:
            self.conversation_id = None
            self.conversation_asset_id = context.get("asset_id")
        context.pop("slice", None)
        scope = self.current_slice()
        if scope:
            context["slice"] = scope
        context.pop("spatial_prompt", None)
        context["spatial_objects"] = []
        if self.volume and scope:
            context["spatial_objects"], _ = spatial_hints.inventory(self)
        self.before_spatial = context["spatial_objects"]
        self.before_slice = scope
        if (
            self.volume
            and self.spatial_hint.currentNode()
            and (
                not context["spatial_objects"]
                or self.spatial_hint.currentNode().IsA("vtkMRMLMarkupsROINode")
            )
        ):
            context["spatial_prompt"] = self.capture_spatial_hint()
        self.before_legacy_hint = context.get("spatial_prompt")
        self.before_hint_node = self.spatial_hint.currentNode()
        selected_model = self.models.currentData
        if selected_model:
            context["model_id"] = selected_model
        else:
            context.pop("model_id", None)
        defaults = self.project["defaults"]
        if defaults:
            context.setdefault("baseline_id", next(iter(defaults.values())))
        self.prompt.clear()
        self.run(
            lambda: self.request(
                f"/api/projects/{self.project_id}/assistant",
                {
                    "message": message,
                    "context": context,
                    "conversation_id": self.conversation_id,
                },
            ),
            self.replied,
        )

    def current_slice(self):
        if not self.volume or self.volume.GetParentTransformNode():
            return None
        matrix = self.source_matrix(inverse=True)
        view = slicer.app.layoutManager().sliceWidget(self.views.currentText).mrmlSliceNode()

        def array(m):
            return np.array([[m.GetElement(i, j) for j in range(4)] for i in range(4)])

        display = self.volume.GetDisplayNode()
        level, window = display.GetLevel(), display.GetWindow()
        with contextlib.suppress(ValueError):
            return source_slice(
                array(matrix),
                array(view.GetXYToRAS()),
                self.asset["spatial_shape"],
                [level - window / 2, level + window / 2],
            )
        return None

    def replied(self, reply):
        self.conversation_id = reply.get("conversation_id")
        if reply["data"].get("model_id"):
            index = self.models.findData(reply["data"]["model_id"])
            if index >= 0:
                self.models.setCurrentIndex(index)
        updated = reply["data"].get("project")
        if updated and updated["id"] == self.project["id"]:
            self.project = updated
            if self.segmentation:
                self.sync_labels(self.segmentation)
        if reply["data"].get("client_action") == "viewer_edit":
            action = reply["data"]
            before = self.checked_edit_mask(action)
            operation = action["operation"]
            if operation == "submit":
                self.save()
            elif operation in {"undo", "redo"}:
                getattr(self.editor, operation)()
                self.sync_labels(self.segmentation)
                changed = not np.array_equal(before, self.current_mask())
                self.proposal_id = None
                text = (
                    f"{'Undid' if operation == 'undo' else 'Redid'} the last segmentation edit."
                    if changed
                    else f"No segmentation change to {operation}."
                )
                self.append_message("Assistant", text)
            else:
                raise RuntimeError("This viewer operation is not supported.")
            return
        if reply["data"].get("client_action") == "edit_spatial_prompts":
            text = spatial_hints.apply(self, reply["data"])
            self.append_message("Assistant", reply["message"] + " " + text)
            return
        if reply["data"].get("client_action") == "clear_segments":
            text = self.clear_segments(reply["data"])
            self.append_message("Assistant", text)
            return
        if reply["data"].get("client_action") == "review_annotation":
            action = reply["data"]
            if (
                action["asset_id"] != self.asset["id"]
                or action["base_revision"] != self.asset["revision"]
            ):
                raise RuntimeError("Review action belongs to another sample or revision.")
            if self.before_prediction != hashlib.sha256(self.current_mask().tobytes()).hexdigest():
                raise RuntimeError(
                    "The segmentation changed while interpreting the review. Retry it."
                )
            self.review_decision(action["verdict"], action.get("comment", ""))
            return
        if reply["data"].get("client_action") == "remove_regions":
            text = self.remove_regions(reply["data"])
            self.append_message("Assistant", text)
            return
        self.append_message("Assistant", reply["message"])
        self.context.update(
            {
                k: v
                for k, v in reply["data"].items()
                if k in {"model_id", "snapshot_id", "evaluation_id"}
            }
        )
        self.job_id = reply.get("job_id")

    def checked_edit_mask(self, action):
        if not (self.can_annotate or self.can_review):
            raise RuntimeError("Annotation or review permission is required.")
        current = self.current_mask()
        if (
            action["project_id"] != self.project_id
            or action["asset_id"] != self.asset["id"]
            or action["base_revision"] != self.asset["revision"]
        ):
            raise RuntimeError("This edit belongs to another sample or revision. Reload and retry.")
        if self.before_prediction != hashlib.sha256(current.tobytes()).hexdigest():
            raise RuntimeError("The annotation changed during this request. Retry the prompt.")
        if self.editor.segmentationNode() != self.segmentation:
            raise RuntimeError(
                "Select this sample's segmentation in Manual editing before retrying."
            )
        return current

    def clear_segments(self, action):
        current = self.checked_edit_mask(action)
        if action.get("image_region") is not None:
            raise RuntimeError("Clearing a selected image region is not supported in Slicer.")
        ids = action["label_ids"]
        labels = [label for label in self.project["labels"] if label["id"] in ids and label["id"]]
        if not ids or {label["id"] for label in labels} != set(ids):
            raise RuntimeError("Clear request must name existing foreground labels.")
        scope = action.get("slice")
        updated = geometry.clear_labels(current, ids, scope)
        count = int(np.count_nonzero(current != updated))
        names = ", ".join(label["name"] for label in labels)
        where = "on the requested slice" if scope else "throughout the volume"
        if not count:
            return f"No {names} voxels to clear {where}."
        self.editor.saveStateForUndo()
        self.apply_mask(updated)
        self.proposal_id = None
        return (
            f"Cleared {names} {where} ({count:,} voxels). "
            'Say "undo" to restore. This is a local edit until you submit.'
        )

    def job_updated(self, job):
        self.status.setText(f"{job['progress']:.0%}" if job["status"] == "running" else "Queued…")
        self.status.setToolTip(job.get("progress_message", ""))
        if job["status"] in {"failed", "cancelled", "interrupted"}:
            self.job_id = None
            raise RuntimeError(job.get("error") or job["status"])
        if job["status"] != "succeeded":
            return
        self.job_id = None
        result = job["result"]
        self.context.update(
            {k: v for k, v in result.items() if k in {"model_id", "snapshot_id", "evaluation_id"}}
        )
        if "region_id" in result:
            identifier = result["region_id"]
            self.run(lambda: self.request(f"/api/regions/{identifier}"), self.region_proposed)
        elif "proposal_id" in result:
            identifier = result["proposal_id"]
            self.run(
                lambda: (
                    self.request(f"/api/proposals/{identifier}"),
                    self.request_mask(
                        f"/api/proposals/{identifier}/mask", self.asset["spatial_shape"]
                    ),
                ),
                self.proposed,
            )
        else:
            self.append_message("Assistant", f"{job['kind'].capitalize()} completed.")

    def region_proposed(self, region):
        if getattr(self, "before_slice", None):
            spatial_hints.check(self, self.before_spatial, self.before_slice)
        if (
            region["asset_id"] != self.asset["id"]
            or region["base_revision"] != self.asset["revision"]
        ):
            raise RuntimeError("Region belongs to another sample or revision.")
        if region.get("end_index") is not None:
            self.roi_proposed(region)
            return
        if not region["bounds"]:
            text = f"No {region['target']} was found on the selected slice. No box was added."
        else:
            low, high = region["bounds"]
            axes = [axis for axis in range(3) if axis != region["slice"]["axis"]]
            matrix = self.source_matrix()
            node = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLMarkupsClosedCurveNode", region["target"] + " · bounding box"
            )
            node.SetAttribute("MONAILabel.Target", region["target"])
            node.SetAttribute("MONAILabel.AssetID", region["asset_id"])
            node.SetAttribute("MONAILabel.ProjectID", self.project_id)
            node.SetAttribute("MONAILabel.RegionID", region["id"])
            node.SetCurveTypeToLinear()
            for a, b in ((0, 0), (1, 0), (1, 1), (0, 1)):
                corner = list(low)
                corner[axes[0]] = (low, high)[a][axes[0]]
                corner[axes[1]] = (low, high)[b][axes[1]]
                ras = matrix.MultiplyPoint(corner + [1])
                node.AddControlPoint(vtk.vtkVector3d(*ras[:3]))
            node.GetDisplayNode().SetSelectedColor(1.0, 0.78, 0.34)
            node.GetDisplayNode().SetPropertiesLabelVisibility(False)
            self.region_nodes.append(node)
            self.spatial_hint.setCurrentNode(node)
            text = (
                f"Added an editable {region['target']} box on the requested slice. "
                "Drag its corners to adjust it. Box edits stay in the Slicer scene; "
                "save the scene to retain them. They are not segmentation training labels."
            )
        self.append_message("Assistant", text)

    def roi_proposed(self, region):
        first, last = region["slice"]["index"] + 1, region["end_index"] + 1
        if not region["bounds"]:
            text = f"No {region['target']} was detected on slices {first}–{last}. No ROI was added."
        else:
            if self.volume.GetParentTransformNode():
                raise RuntimeError("Harden the volume transform before adding a source-grid ROI.")
            affine = np.asarray(self.asset["affine"])
            orientation, size = roi_geometry(affine, region["bounds"])
            transform = vtk.vtkMatrix4x4()
            for i in range(4):
                for j in range(4):
                    transform.SetElement(i, j, orientation[i, j])
            node = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLMarkupsROINode", f"{region['target']} · ROI {first}–{last}"
            )
            node.SetAndObserveObjectToNodeMatrix(transform)
            node.SetSize(*size)
            for key, value in {
                "Target": region["target"],
                "AssetID": region["asset_id"],
                "ProjectID": self.project_id,
                "RegionID": region["id"],
            }.items():
                node.SetAttribute("MONAILabel." + key, value)
            node.GetDisplayNode().SetSelectedColor(0.3, 0.85, 1.0)
            node.GetDisplayNode().SetPropertiesLabelVisibility(False)
            node.GetDisplayNode().SetHandlesInteractive(True)
            self.region_nodes.append(node)
            self.spatial_hint.setCurrentNode(node)
            text = (
                f"Added an editable {region['target']} ROI spanning slices {first}–{last} "
                f"(numbered from 1). Target detected on {region['detected_slices']} of "
                f"{last - first + 1} slices; width and height enclose all detections. "
                "Save the Slicer scene to retain ROI edits. ROIs are not segmentation labels."
            )
        self.append_message("Assistant", text)

    def capture_spatial_hint(self):
        node = self.spatial_hint.currentNode()
        if self.volume.GetParentTransformNode() or node.GetParentTransformNode():
            raise ValueError("Harden volume and markup transforms before using a SAM hint.")
        if node.GetAttribute("MONAILabel.AssetID") not in (None, self.asset["id"]):
            raise ValueError("Select a SAM hint belonging to the current sample.")
        matrix = self.source_matrix(inverse=True)
        if node.IsA("vtkMRMLMarkupsROINode"):
            transform = node.GetObjectToWorldMatrix()
            size = node.GetSize()
            points = [
                matrix.MultiplyPoint(
                    transform.MultiplyPoint([a * size[0] / 2, b * size[1] / 2, c * size[2] / 2, 1])
                )[:3]
                for a in (-1, 1)
                for b in (-1, 1)
                for c in (-1, 1)
            ]
        else:
            points = []
            for index in range(node.GetNumberOfControlPoints()):
                if node.GetNthControlPointPositionStatus(index) != node.PositionDefined:
                    continue
                position = [0.0, 0.0, 0.0]
                node.GetNthControlPointPositionWorld(index, position)
                points.append(list(matrix.MultiplyPoint(position + [1]))[:3])
        if not points:
            raise ValueError("Place a point or box before using SAM.")
        shape = np.array(self.asset["spatial_shape"])
        coords = np.clip(np.asarray(points), 0, shape - 1)
        if node.IsA("vtkMRMLMarkupsFiducialNode"):
            return {
                "points": [
                    {
                        "coordinates": point.tolist(),
                        "positive": not node.GetNthControlPointLabel(i)
                        .lower()
                        .startswith(("-", "negative")),
                    }
                    for i, point in enumerate(coords)
                ]
            }
        return {"box": [coords.min(axis=0).tolist(), coords.max(axis=0).tolist()]}

    def remove_regions(self, action):
        if not self.can_annotate:
            raise RuntimeError("Annotation permission is required to remove a box.")
        if (
            not self.asset
            or action["asset_id"] != self.asset["id"]
            or action["project_id"] != self.project_id
        ):
            raise RuntimeError("Box removal belongs to another sample or project.")
        target = " ".join(action["target"].casefold().split())
        self.region_nodes = [
            node for node in self.region_nodes if node.GetScene() == slicer.mrmlScene
        ]
        matches = []
        for node in self.region_nodes:
            if not node.IsA("vtkMRMLMarkupsClosedCurveNode"):
                continue
            # Bridge-owned nodes can identify their target through the generated name,
            # including Slicer's _1 suffixes.
            name = node.GetAttribute("MONAILabel.Target")
            if name is None:
                name = node.GetName().split(" · bounding box", 1)[0]
            if " ".join(name.casefold().split()) != target:
                continue
            if node.GetAttribute("MONAILabel.AssetID") not in (None, self.asset["id"]):
                continue
            if node.GetAttribute("MONAILabel.ProjectID") not in (None, self.project_id):
                continue
            if action.get("slice"):
                if self.volume.GetParentTransformNode():
                    raise RuntimeError(
                        "Harden the volume transform before removing boxes by slice."
                    )
                scope = action["slice"]
                matrix = self.source_matrix(inverse=True)
                on_plane = True
                for index in range(node.GetNumberOfControlPoints()):
                    ras = [0.0, 0.0, 0.0]
                    node.GetNthControlPointPositionWorld(index, ras)
                    ijk = matrix.MultiplyPoint(ras + [1.0])
                    if abs(ijk[scope["axis"]] - scope["index"]) > 0.5:
                        on_plane = False
                if not on_plane:
                    continue
            matches.append(node)
        if not matches:
            return f"No {action['target']} bounding box found in the requested sample/slice."
        if len(matches) > 1 and not action.get("all_matches"):
            return (
                f"Found {len(matches)} {action['target']} boxes. Specify 'in current slice' "
                "or say 'remove all bounding boxes for " + action["target"] + "'."
            )
        for node in matches:
            slicer.mrmlScene.RemoveNode(node)
            self.region_nodes.remove(node)
        noun = "box" if len(matches) == 1 else "boxes"
        return f"Removed {len(matches)} {action['target']} bounding {noun}."

    def proposed(self, result):
        proposal, mask = result
        if proposal.get("spatial_prompt") and getattr(self, "before_slice", None):
            spatial_hints.check(self, self.before_spatial, self.before_slice)
            if getattr(self, "before_legacy_hint", None) and (
                self.spatial_hint.currentNode() != self.before_hint_node
                or self.capture_spatial_hint() != self.before_legacy_hint
            ):
                raise RuntimeError("The selected SAM hint changed during inference. Retry it.")
        signature = hashlib.sha256(self.current_mask().tobytes()).hexdigest()
        if signature != self.before_prediction:
            raise RuntimeError(
                "Local edits changed during inference. The proposal was not applied."
            )
        if (
            proposal["asset_id"] != self.asset["id"]
            or proposal["base_revision"] != self.asset["revision"]
        ):
            raise RuntimeError("Proposal belongs to another volume or revision.")
        combined = merge_proposal(
            self.current_mask(),
            np.asarray(mask, dtype=np.uint8),
            proposal["label_ids"],
            None if proposal.get("all_slices") else proposal.get("slice"),
        )
        self.editor.saveStateForUndo()
        self.apply_mask(combined)
        self.proposal_id = proposal["id"]
        measured = mask
        if proposal.get("slice"):
            scope = proposal["slice"]
            measured = np.take(mask, scope["index"], axis=scope["axis"])
        selected = np.isin(measured, proposal["label_ids"])
        count = int(np.count_nonzero(selected))
        scope = "all slices" if proposal.get("all_slices") else "the requested region"
        text = (
            f"Proposal applied to {scope}: {count:,} labeled voxels. "
            "Inspect the colored overlay and correct it in Segment Editor before submission."
            if count
            else "The model returned no target on this image. No colored region was added; "
            "check the selected slice, window, and model."
        )
        self.append_message("Assistant", text)

    def review_decision(self, verdict, comment=None):
        if not self.can_review or not self.asset or not self.asset.get("annotation_id"):
            return
        try:
            note = self.review_comment.text if comment is None else comment
            identifier = self.asset["id"]
            if verdict == "accepted":
                metadata = {
                    "base_revision": self.asset["revision"],
                    "covered_labels": [label["id"] for label in self.project["labels"]],
                    "comment": note,
                }
                payload = self.current_mask().tobytes()
                self.run(
                    lambda: self.request(
                        f"/api/assets/{identifier}/review-complete",
                        payload,
                        headers={"X-MONAILABEL-REVIEW": json.dumps(metadata)},
                    ),
                    lambda annotation: self.reviewed(annotation, True),
                )
            else:
                annotation_id = self.asset["annotation_id"]
                self.run(
                    lambda: self.request(
                        f"/api/annotations/{annotation_id}/decision",
                        {"verdict": "changes_requested", "comment": note},
                    ),
                    lambda decision: self.reviewed(decision, False),
                )
        except Exception as error:
            self.report_error(error)

    def reviewed(self, result, accepted):
        if accepted:
            self.asset["revision"] = result["revision"]
            self.asset["annotation_id"] = result["id"]
            self.proposal_id = None
        text = (
            f"Review saved: {'Good' if accepted else 'Needs changes'} · "
            f"revision {result['revision']}."
        )
        if not accepted:
            text += " Segmentation edits are not included in this decision."
        self.review_comment.clear()
        self.submission_complete("Review submitted", text)

    def save(self):
        if self.review_mode:
            self.report_error("Use Good or Needs changes to submit your review and corrections.")
            return
        if not self.asset:
            return
        try:
            params = {
                "base_revision": self.asset["revision"],
                "covered_labels": [label["id"] for label in self.project["labels"]],
            }
            if self.proposal_id:
                params["proposal_id"] = self.proposal_id
            payload = self.current_mask().tobytes()
            query = urllib.parse.urlencode(params, doseq=True)
            identifier = self.asset["id"]
            self.run(
                lambda: self.request(f"/api/assets/{identifier}/review-mask?{query}", payload),
                self.saved,
            )
        except Exception as error:
            self.report_error(error)

    def saved(self, annotation):
        self.asset["revision"] = annotation["revision"]
        self.asset["annotation_id"] = annotation["id"]
        self.proposal_id = None
        self.submission_complete(
            "Annotation submitted",
            f"Revision {annotation['revision']} saved. Awaiting reviewer acceptance.",
        )

    def submission_complete(self, title, text):
        """Offer to exit only after the server has persisted the submission."""
        self.append_message("Assistant", text)
        dialog = qt.QMessageBox(slicer.util.mainWindow())
        dialog.setWindowTitle(title)
        dialog.setIcon(qt.QMessageBox.Information)
        dialog.setTextFormat(qt.Qt.PlainText)
        dialog.setText(text)
        dialog.setInformativeText(
            "Exit 3D Slicer and return to the web workspace?\n\n"
            "Other local edits, boxes and ROIs are not saved by exiting. "
            "Choose Keep working to save those in your scene first."
        )
        exit_button = dialog.addButton("Exit Slicer", qt.QMessageBox.AcceptRole)
        keep_button = dialog.addButton("Keep working", qt.QMessageBox.RejectRole)
        dialog.setDefaultButton(exit_button)
        dialog.setEscapeButton(keep_button)
        self.timer.stop()
        exit_requested = False
        try:
            dialog.exec()
            exit_requested = dialog.clickedButton() == exit_button
        finally:
            if not exit_requested:
                self.timer.start()
            dialog.deleteLater()
        if exit_requested:
            slicer.util.exit()


def start():
    existing = getattr(slicer, "monailabelAssistant", None)
    if existing is not None:
        if existing.future or existing.job_id or getattr(existing, "loading", False):
            raise RuntimeError(
                "Wait for the current assistant operation before updating the bridge."
            )
        existing.__class__ = AnnotationDock
        existing.editor.setUndoEnabled(True)
        existing.editor.setMaximumNumberOfUndoStates(5)
        existing.show_welcome()
        text = "Assistant updated in place. Your current volume, boxes, and edits are retained."
        # Move the old footer into the action row when updating an existing session.
        existing.layout.removeWidget(existing.status)
        for index in range(existing.layout.count()):
            row = existing.layout.itemAt(index).layout()
            if row and row.indexOf(existing.send_button) >= 0:
                row.addWidget(existing.status)
                break
        existing.status.setWordWrap(False)
        existing.status.clear()
        existing.status.hide()
        existing.append_message("Assistant", text)
        return
    config = Path(os.environ["MONAILABEL_VIEWER_SESSION"])
    # Retain the dock and timer for the lifetime of this Slicer process.
    slicer.monailabelAssistant = AnnotationDock(json.loads(config.read_text()))
    config.unlink(missing_ok=True)


start()

"""Run inside a disposable Slicer process with --python-script, without a server."""

import ast
import hashlib
from pathlib import Path

import numpy as np
import slicer


def run():
    root = Path(__file__).resolve().parents[2]
    bridge = root / "packages/viewers/src/monailabel/viewers/resources/slicer_bridge.py"
    module = ast.parse(bridge.read_text(), filename=str(bridge))
    # Load the installed adapter without its application-launch entry point.
    assert isinstance(module.body[-1], ast.Expr)
    assert isinstance(module.body[-1].value, ast.Call)
    assert module.body[-1].value.func.id == "start"
    module.body.pop()
    namespace = {"__file__": str(bridge)}
    exec(compile(module, str(bridge), "exec"), namespace)

    try:
        dock = namespace["AnnotationDock"].__new__(namespace["AnnotationDock"])
        dock.project_id = "disposable"
        dock.asset = {
            "id": "sample",
            "revision": 0,
            "spatial_shape": [16, 18, 12],
            "affine": np.eye(4).tolist(),
        }
        dock.project = {
            "labels": [
                {"id": 0, "name": "Background", "color": "#000000"},
                {"id": 1, "name": "Spleen", "color": "#9d6ca2"},
                {"id": 2, "name": "Liver", "color": "#dd8265"},
            ]
        }
        dock.can_annotate, dock.can_review = True, False
        dock.volume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        slicer.util.updateVolumeFromArray(dock.volume, np.zeros((12, 18, 16), np.float32))
        dock.segmentation = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        dock.segmentation.CreateDefaultDisplayNodes()
        dock.segmentation.SetReferenceImageGeometryParameterFromVolumeNode(dock.volume)
        dock.sync_labels(dock.segmentation)
        dock.editor = slicer.qMRMLSegmentEditorWidget()
        dock.editor.setMRMLScene(slicer.mrmlScene)
        editor_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentEditorNode")
        dock.editor.setMRMLSegmentEditorNode(editor_node)
        dock.editor.setSegmentationNode(dock.segmentation)
        dock.editor.setSourceVolumeNode(dock.volume)
        dock.editor.setUndoEnabled(True)
        original = np.zeros((16, 18, 12), np.uint8)
        original[2:6, 3:10, 3:9] = 1
        original[9:14, 4:14, 2:10] = 2
        dock.apply_mask(original)
        for axis in (0, 1, 2):
            selected = {"axis": axis, "index": 4}
            dock.before_prediction = hashlib.sha256(original.tobytes()).hexdigest()
            dock.clear_segments(
                {
                    "project_id": dock.project_id,
                    "asset_id": dock.asset["id"],
                    "base_revision": 0,
                    "label_ids": [1],
                    "slice": selected,
                }
            )
            expected = original.copy()
            plane = [slice(None)] * 3
            plane[axis] = selected["index"]
            view = expected[tuple(plane)]
            view[view == 1] = 0
            np.testing.assert_array_equal(dock.current_mask(), expected)
            dock.editor.undo()
            np.testing.assert_array_equal(dock.current_mask(), original)
            dock.editor.redo()
            np.testing.assert_array_equal(dock.current_mask(), expected)
            dock.editor.undo()
        print("SLICER_SCOPED_CLEAR_AND_NATIVE_UNDO_OK", flush=True)
        slicer.util.exit(0)
    except Exception:
        import traceback

        traceback.print_exc()
        slicer.util.exit(1)


run()

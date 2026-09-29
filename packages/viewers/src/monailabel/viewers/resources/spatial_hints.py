# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Slicer-native spatial hints. Loaded in Slicer's Python; no server or model dependencies."""

import numpy as np
import qt
import slicer
import vtk


def observe_selection(node):
    """Use native markup interaction to choose a box without an extra selector."""
    if getattr(node, "_monailabel_selection_observers", None):
        return

    def select(caller, event):
        slicer.modules.markups.logic().SetActiveListID(node)

    node._monailabel_selection_observers = [
        node.AddObserver(node.PointStartInteractionEvent, select),
        node.GetDisplayNode().AddObserver(node.GetDisplayNode().ActionEvent, select),
    ]


def show_target(dock, target=None):
    """Keep every label's hints, showing the active label while drawing."""
    for node in dock.region_nodes:
        if node.GetScene() and node.GetDisplayNode():
            label = node.GetAttribute("MONAILabel.Target") or ""
            node.GetDisplayNode().SetVisibility(
                target is None or label.casefold() == target.casefold()
            )


def inventory(dock):
    if dock.volume.GetParentTransformNode():
        raise ValueError("Harden the volume transform before editing spatial hints.")
    selected = dock.selected_hint()
    nodes = [n for n in dock.region_nodes if n.GetScene() == slicer.mrmlScene]
    if selected and selected not in nodes:
        nodes.append(selected)
    matrix = dock.source_matrix(inverse=True)
    objects, handles = [], {}
    for node in nodes:
        if node.GetAttribute("MONAILabel.AssetID") not in (None, dock.asset["id"]):
            continue
        if node.GetAttribute("MONAILabel.ProjectID") not in (None, dock.project_id):
            continue
        if not (
            node.IsA("vtkMRMLMarkupsFiducialNode") or node.IsA("vtkMRMLMarkupsClosedCurveNode")
        ):
            continue
        if node.GetParentTransformNode():
            raise ValueError("Harden markup transforms before editing spatial hints.")
        points = []
        for i in range(node.GetNumberOfControlPoints()):
            if node.GetNthControlPointPositionStatus(i) != node.PositionDefined:
                continue
            world = [0.0, 0.0, 0.0]
            node.GetNthControlPointPositionWorld(i, world)
            points.append((i, [round(v, 5) for v in matrix.MultiplyPoint(world + [1])[:3]]))
        target = node.GetAttribute("MONAILabel.Target") or ""
        base = node.GetAttribute("MONAILabel.HintID") or node.GetID()
        if node.IsA("vtkMRMLMarkupsFiducialNode"):
            for i, point in points:
                identifier = base + ":" + node.GetNthControlPointID(i)
                objects.append(
                    dict(
                        id=identifier,
                        target=target,
                        kind="point",
                        coordinates=[point],
                        positive=not node.GetNthControlPointLabel(i)
                        .lower()
                        .startswith(("-", "negative")),
                        selected=node == selected,
                    )
                )
                handles[identifier] = (node, i)
        elif points:
            coords = np.asarray([p for _, p in points])
            objects.append(
                dict(
                    id=base,
                    target=target,
                    kind="box",
                    coordinates=[coords.min(axis=0).tolist(), coords.max(axis=0).tolist()],
                    positive=True,
                    selected=node == selected,
                )
            )
            handles[base] = (node, None)
    return sorted(objects, key=lambda item: item["id"]), handles


def same_slice(a, b):
    return a and b and a["axis"] == b["axis"] and a["index"] == b["index"]


def check(dock, expected, scope):
    if not same_slice(scope, dock.current_slice()):
        raise RuntimeError("The source slice changed during the request. Retry it.")
    current, handles = inventory(dock)
    if current != expected:
        raise RuntimeError("spatial hints or their selection changed during the request. Retry it.")
    return handles


def write(dock, item, scope, handle=None):
    node, index = handle if handle else (None, None)
    point = item["kind"] == "point"
    if node is None:
        node = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLMarkupsFiducialNode" if point else "vtkMRMLMarkupsClosedCurveNode",
            (item["target"] or "Prompt")
            + (
                " · positive point"
                if point and item["positive"]
                else " · negative point"
                if point
                else " · bounding box"
            ),
        )
        for key, value in {
            "HintID": item["id"],
            "Target": item["target"],
            "AssetID": dock.asset["id"],
            "ProjectID": dock.project_id,
        }.items():
            node.SetAttribute("MONAILabel." + key, value)
        dock.region_nodes.append(node)
    matrix = dock.source_matrix()
    if point:
        world = matrix.MultiplyPoint(item["coordinates"][0] + [1])[:3]
        if index is None:
            index = node.AddControlPoint(vtk.vtkVector3d(*world))
        else:
            node.SetNthControlPointPositionWorld(index, *world)
        node.SetNthControlPointLabel(
            index, ("positive" if item["positive"] else "negative") + " " + item["target"]
        )
    else:
        node.RemoveAllControlPoints()
        node.SetCurveTypeToLinear()
        axes = [a for a in range(3) if a != scope["axis"]]
        low, high = item["coordinates"]
        for a, b in ((0, 0), (1, 0), (1, 1), (0, 1)):
            p = list(low)
            p[axes[0]], p[axes[1]] = (low, high)[a][axes[0]], (low, high)[b][axes[1]]
            node.AddControlPoint(vtk.vtkVector3d(*matrix.MultiplyPoint(p + [1])[:3]))
    display = node.GetDisplayNode()
    color = (
        (0.3, 0.9, 0.4)
        if point and item["positive"]
        else (1.0, 0.3, 0.3)
        if point
        else (1.0, 0.78, 0.34)
    )
    display.SetSelectedColor(*color)
    display.SetColor(*color)
    display.SetPropertiesLabelVisibility(False)
    display.SetPointLabelsVisibility(False)
    return node


def apply(dock, action):
    dock.checked_edit_mask(action)
    handles = check(dock, action["expected"], action["slice"])
    cancel_placement(dock)
    # Remove control points backwards so list indices remain valid.
    removed = sorted((handles[i] for i in action["remove"]), key=lambda h: h[1] or 0, reverse=True)
    for node, index in removed:
        if index is None or node.GetNumberOfControlPoints() == 1:
            slicer.mrmlScene.RemoveNode(node)
        else:
            node.RemoveNthControlPoint(index)
    for item in action["upsert"]:
        write(dock, item, action["slice"], handles.get(item["id"]))
    dock.region_nodes = [n for n in dock.region_nodes if n.GetScene() == slicer.mrmlScene]
    show_target(dock, dock.interaction_target if dock.hint_panel.visible else None)
    return (
        f"Updated spatial hints ({len(action['upsert'])} added/adjusted, "
        f"{len(action['remove'])} removed). Drag them to refine; segmentation unchanged."
    )


def cancel_placement(dock):
    dock.interaction_mode = "navigate"
    for button in getattr(dock, "hint_buttons", {}).values():
        button.setChecked(False)
    pending = getattr(dock, "hint_placement", None)
    if pending is None:
        return
    node, observer, kind = pending
    dock.hint_placement = None
    node.RemoveObserver(observer)
    interaction = slicer.app.applicationLogic().GetInteractionNode()
    interaction.SetPlaceModePersistence(False)
    interaction.SetCurrentInteractionMode(interaction.ViewTransform)
    if node.GetScene() and (kind == "box" or not node.GetNumberOfDefinedControlPoints()):
        slicer.mrmlScene.RemoveNode(node)


def place(dock, kind, positive=True):
    """Native point placement persists across slices; two corners create a slice box."""
    if dock.asset is None or dock.volume is None:
        raise ValueError("Open an image before adding a spatial prompt.")
    if dock.future or dock.job_id or dock.loading:
        raise ValueError("Wait for the current request before changing prompts.")
    if not (dock.can_annotate or dock.can_review):
        raise ValueError("Annotation permission is required to add spatial prompts.")
    scope = dock.current_slice()
    if not scope:
        raise ValueError("Select a source slice first.")
    cancel_placement(dock)
    dock.configure_interaction()
    specification = getattr(dock, "hint_specification", None)
    input_kind = "box" if kind == "box" else "positive_point" if positive else "negative_point"
    if not specification or input_kind not in specification["inputs"]:
        raise ValueError("The selected model does not accept this input.")
    target = getattr(dock, "interaction_target", "")
    if not target:
        raise ValueError("Choose or type a target label before placing hints.")
    dock.interaction_mode = "box" if kind == "box" else "positive" if positive else "negative"
    show_target(dock, target)
    dock.hint_panel.show()
    for key, button in dock.hint_buttons.items():
        button.setChecked(key == input_kind)
    asset_id = dock.asset["id"]
    name = "Box corners" if kind == "box" else "Positive points" if positive else "Negative points"
    node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", name)
    if kind == "box":
        node.SetMaximumNumberOfControlPoints(2)
    else:
        dock.region_nodes.append(node)
    for key, value in {"AssetID": asset_id, "ProjectID": dock.project_id, "Target": target}.items():
        node.SetAttribute("MONAILabel." + key, value)
    node.SetControlPointLabelFormat(("positive" if positive else "negative") + " %d")
    display = node.GetDisplayNode()
    color = (0.3, 0.9, 0.4) if positive else (1, 0.3, 0.3)
    display.SetSelectedColor(*color)
    display.SetColor(*color)
    display.SetPropertiesLabelVisibility(False)
    display.SetPointLabelsVisibility(False)
    dock.select_hint(node)

    def placed(caller, event):
        if kind == "point":
            return  # Each native point keeps its polarity; validate geometry before inference.
        if node.GetNumberOfDefinedControlPoints() < 2:
            return
        try:
            if dock.asset["id"] != asset_id or not same_slice(scope, dock.current_slice()):
                raise ValueError("The image or slice changed. Place the box again.")
            matrix = dock.source_matrix(inverse=True)
            points = []
            for index in range(2):
                world = [0.0, 0.0, 0.0]
                node.GetNthControlPointPositionWorld(index, world)
                point = list(matrix.MultiplyPoint(world + [1])[:3])
                if abs(point[scope["axis"]] - scope["index"]) > 0.5:
                    raise ValueError("Place the box in the selected slice view.")
                point[scope["axis"]] = scope["index"]
                if any(
                    v < 0 or v > n - 1
                    for v, n in zip(point, dock.asset["spatial_shape"], strict=True)
                ):
                    raise ValueError("Place the box inside the source image.")
                points.append(point)
            coordinates = [np.min(points, axis=0).tolist(), np.max(points, axis=0).tolist()]
            if any(coordinates[0][a] >= coordinates[1][a] for a in range(3) if a != scope["axis"]):
                raise ValueError("Draw a box with nonzero width and height.")
            item = dict(
                id=node.GetID(), target=target, kind="box", coordinates=coordinates, positive=True
            )
            cancel_placement(dock)
            result = write(dock, item, scope)
            dock.select_hint(result)
        except Exception as exc:
            cancel_placement(dock)
            slicer.util.errorDisplay(str(exc))

    # Markups still processes the native mouse event when PointPositionDefined fires.
    # Finish box conversion on the next Qt turn before removing its temporary node.
    def schedule(caller, event):
        if kind == "box" and node.GetNumberOfDefinedControlPoints() == 2:
            qt.QTimer.singleShot(0, lambda: placed(caller, event) if node.GetScene() else None)

    observer = node.AddObserver(node.PointPositionDefinedEvent, schedule)
    dock.hint_placement = (node, observer, kind)
    slicer.modules.markups.logic().SetActiveListID(node)
    interaction = slicer.app.applicationLogic().GetInteractionNode()
    interaction.SetPlaceModePersistence(True)
    interaction.SetCurrentInteractionMode(interaction.Place)

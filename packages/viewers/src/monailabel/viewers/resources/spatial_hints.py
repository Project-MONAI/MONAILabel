"""Slicer-native SAM hints. Loaded in Slicer's Python; no server or model dependencies."""

import numpy as np
import slicer
import vtk


def inventory(dock):
    if dock.volume.GetParentTransformNode():
        raise ValueError("Harden the volume transform before editing SAM hints.")
    selected = dock.spatial_hint.currentNode()
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
            raise ValueError("Harden markup transforms before editing SAM hints.")
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
        raise RuntimeError("SAM hints or their selection changed during the request. Retry it.")
    return handles


def write(dock, item, scope, handle=None):
    node, index = handle if handle else (None, None)
    point = item["kind"] == "point"
    if node is None:
        node = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLMarkupsFiducialNode" if point else "vtkMRMLMarkupsClosedCurveNode",
            (item["target"] or "SAM")
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
    return node


def apply(dock, action):
    dock.checked_edit_mask(action)
    handles = check(dock, action["expected"], action["slice"])
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
    return (
        f"Updated SAM hints ({len(action['upsert'])} added/adjusted, "
        f"{len(action['remove'])} removed). Drag them to refine; segmentation unchanged."
    )

import {
  annotation,
  ToolGroupManager,
  Enums,
  ProbeTool,
  RectangleROITool,
  utilities,
} from "@cornerstonejs/tools";
import { mat4, vec3 } from "gl-matrix";

function viewportFor(services) {
  return services.cornerstoneViewportService.getCornerstoneViewport(
    services.viewportGridService.getActiveViewportId(),
  );
}

export function drawHint(services, tool) {
  const viewport = viewportFor(services);
  if (!viewport) throw new Error("Select the source viewport first.");
  const group = ToolGroupManager.getToolGroupForViewport(
    viewport.id,
    viewport.renderingEngineId,
  );
  if (!group) throw new Error("The selected viewport has no annotation tools.");
  const active = group.getActivePrimaryMouseButtonTool();
  if (active) group.setToolPassive(active);
  if (!group.hasTool(tool)) group.addTool(tool);
  group.setToolActive(tool, {
    bindings: [{ mouseButton: Enums.MouseBindings.Primary }],
  });
}

// A stable source-coordinate snapshot also guards edits made while chat/inference is pending.
export function spatialObjects(services, asset) {
  const viewport = viewportFor(services);
  if (!viewport) throw new Error("Select the source viewport first.");
  const matrix = sourceMatrix(asset);
  const inverse = mat4.invert(mat4.create(), matrix);
  if (!inverse) throw new Error("Invalid source affine.");
  const selected = new Set(annotation.selection.getAnnotationsSelected());
  return ["RectangleROI", "Probe"]
    .flatMap((tool) =>
      (annotation.state.getAnnotations(tool, viewport.element) || [])
        .filter(
          (item) =>
            (!item.data.monailabelAsset ||
              item.data.monailabelAsset === asset.id) &&
            annotation.visibility.isAnnotationVisible(item.annotationUID) &&
            item.data.handles?.points?.length,
        )
        .map((item) => {
          const points = item.data.handles.points.map((p) =>
            Array.from(
              vec3.transformMat4(vec3.create(), p, inverse),
              (v) => Math.round(v * 1e5) / 1e5,
            ),
          );
          const box = tool === "RectangleROI";
          return {
            id: item.annotationUID,
            target: item.data.monailabelTarget || "",
            kind: box ? "box" : "point",
            coordinates: box
              ? [Math.min, Math.max].map((fn) =>
                  [0, 1, 2].map((axis) => fn(...points.map((p) => p[axis]))),
                )
              : [points[0]],
            positive: box || !/^(negative|-)/i.test(item.data.label || ""),
            selected: selected.has(item.annotationUID),
          };
        }),
    )
    .sort((a, b) => a.id.localeCompare(b.id));
}

function sourceMatrix(asset) {
  const matrix = mat4.fromValues(...asset.affine.flat());
  mat4.transpose(matrix, matrix);
  return mat4.multiply(
    matrix,
    mat4.fromScaling(mat4.create(), [-1, -1, 1]),
    matrix,
  );
}

export function checkSpatial(services, asset, expected, scope, currentScope) {
  if (scope.axis !== currentScope.axis || scope.index !== currentScope.index)
    throw new Error("The source slice changed during the request. Retry it.");
  if (
    JSON.stringify(spatialObjects(services, asset)) !== JSON.stringify(expected)
  )
    throw new Error(
      "SAM hints or their selection changed during the request. Retry it.",
    );
}

function writeSpatial(services, asset, item, scope) {
  const viewport = viewportFor(services);
  const group = ToolGroupManager.getToolGroupForViewport(
    viewport.id,
    viewport.renderingEngineId,
  );
  const Tool = item.kind === "point" ? ProbeTool : RectangleROITool;
  if (!group.hasTool(Tool.toolName)) group.addTool(Tool.toolName);
  // Passive permits native dragging without taking over the user's primary tool.
  if (group.getToolInstance(Tool.toolName).mode === Enums.ToolModes.Disabled)
    group.setToolPassive(Tool.toolName);
  const matrix = sourceMatrix(asset);
  let points = item.coordinates;
  if (item.kind === "box") {
    const axes = [0, 1, 2].filter((a) => a !== scope.axis),
      [low, high] = points;
    points = [
      [0, 0],
      [1, 0],
      [0, 1],
      [1, 1],
    ].map(([a, b]) => {
      const p = [...low];
      p[axes[0]] = [low, high][a][axes[0]];
      p[axes[1]] = [low, high][b][axes[1]];
      return p;
    });
  }
  const world = points.map((p) =>
    Array.from(vec3.transformMat4(vec3.create(), p, matrix)),
  );
  let native = annotation.state.getAnnotation(item.id);
  const fresh = !native;
  if (fresh)
    native = Tool.createAnnotationForViewport(viewport, {
      data: {
        handles: {
          points: world,
          activeHandleIndex: null,
          textBox: {
            hasMoved: false,
            worldPosition: [0, 0, 0],
            worldBoundingBox: {
              topLeft: [0, 0, 0],
              topRight: [0, 0, 0],
              bottomLeft: [0, 0, 0],
              bottomRight: [0, 0, 0],
            },
          },
        },
        cachedStats: {},
      },
    });
  native.annotationUID = item.id;
  native.data.handles.points = world;
  native.data.cachedStats = {};
  native.data.label =
    (item.kind === "point"
      ? item.positive
        ? "positive "
        : "negative "
      : "box ") + item.target;
  native.data.monailabelAsset = asset.id;
  native.data.monailabelTarget = item.target;
  native.invalidated = true;
  if (fresh) annotation.state.addAnnotation(native, viewport.element);
  const color =
    item.kind === "box" ? "#ffc757" : item.positive ? "#4de666" : "#ff4d4d";
  annotation.config.style.setAnnotationStyles(item.id, {
    color,
    colorHighlighted: color,
    colorSelected: color,
  });
}

export function applySpatial(services, asset, projectId, action, scope) {
  if (
    action.project_id !== projectId ||
    action.asset_id !== asset.id ||
    action.base_revision !== asset.revision
  )
    throw new Error("SAM edit belongs to another sample or revision.");
  checkSpatial(services, asset, action.expected, action.slice, scope);
  action.remove.forEach((id) => annotation.state.removeAnnotation(id));
  action.upsert.forEach((item) =>
    writeSpatial(services, asset, item, action.slice),
  );
  utilities.triggerAnnotationRenderForViewportIds([viewportFor(services).id]);
}

export function applyRegion(
  services,
  asset,
  projectId,
  region,
  expected,
  scope,
  currentScope,
) {
  if (
    region.project_id !== projectId ||
    region.asset_id !== asset.id ||
    region.base_revision !== asset.revision ||
    region.end_index != null
  )
    throw new Error(
      "Box belongs to another sample/revision or has unsupported 3D scope.",
    );
  checkSpatial(services, asset, expected, scope, currentScope);
  if (!region.bounds.length) return false;
  writeSpatial(
    services,
    asset,
    {
      id: region.id,
      target: region.target,
      kind: "box",
      coordinates: region.bounds,
      positive: true,
    },
    region.slice,
  );
  utilities.triggerAnnotationRenderForViewportIds([viewportFor(services).id]);
  return true;
}

import { cache } from "@cornerstonejs/core";
import { segmentation } from "@cornerstonejs/tools";
import { mat4, vec3 } from "gl-matrix";

export class MaskTransfer {
  constructor(services, asset, source, project) {
    Object.assign(this, { services, asset, source, project });
    this.id = "monailabel:" + asset.id;
  }
  sourceViewports() {
    const { viewportGridService, cornerstoneViewportService } = this.services;
    return [...viewportGridService.getState().viewports.keys()].filter((id) => {
      const viewport = cornerstoneViewportService.getCornerstoneViewport(id);
      const image = viewport?.getImageIds?.()?.[0];
      const uid =
        image && decodeURIComponent(image).match(/\/instances\/([0-9.]+)/)?.[1];
      return uid && this.source.instance_uids.includes(uid);
    });
  }
  canPrepare() {
    return (
      this.sourceViewports().length > 0 &&
      this.services.displaySetService
        .getActiveDisplaySets()
        .some(
          (d) =>
            d.SeriesInstanceUID === this.source.series_uid &&
            d.imageIds?.length,
        )
    );
  }
  syncProject(project) {
    const service = this.services.segmentationService;
    const current = service.getSegmentation(this.id);
    for (const label of project.labels.filter((l) => l.id)) {
      const color = [
        ...label.color
          .slice(1)
          .match(/../g)
          .map((c) => parseInt(c, 16)),
        255,
      ];
      if (current && !current.segments?.[label.id]) {
        service.addSegment(this.id, {
          segmentIndex: label.id,
          label: label.name,
          color,
        });
      }
      for (const id of service.getViewportIdsWithSegmentation(this.id))
        service.setSegmentColor(id, this.id, label.id, color);
    }
    this.project = project;
  }
  async prepare() {
    const { displaySetService, segmentationService } = this.services;
    const ds = displaySetService
      .getActiveDisplaySets()
      .find((d) => d.SeriesInstanceUID === this.source.series_uid);
    if (!ds?.imageIds?.length || !this.sourceViewports().length)
      throw new Error("Load the source DICOM series in a viewport first.");
    if (!segmentationService.getSegmentation(this.id)) {
      const segments = Object.fromEntries(
        this.project.labels
          .filter((l) => l.id)
          .map((l) => [l.id, { label: l.name }]),
      );
      await segmentationService.createLabelmapForDisplaySet(ds, {
        segmentationId: this.id,
        label: "MONAI Label",
        segments,
      });
    }
    if (!this.images) {
      const data = segmentationService.getSegmentation(this.id)
        .representationData.Labelmap;
      this.images = [...data.imageIds];
      this.references = [...data.referencedImageIds];
      this.indices = this.references.map((reference) => {
        const uid = decodeURIComponent(reference).match(
          /\/instances\/([0-9.]+)/,
        )?.[1];
        const index = this.source.instance_uids.indexOf(uid);
        if (index < 0)
          throw new Error("The viewport contains a different DICOM series.");
        return index;
      });
      if (new Set(this.indices).size !== this.source.instance_uids.length)
        throw new Error("Incomplete source series.");
    }
    for (const viewportId of this.sourceViewports()) {
      if (
        !segmentationService
          .getViewportIdsWithSegmentation(this.id)
          .includes(viewportId)
      )
        await segmentationService.addSegmentationRepresentation(viewportId, {
          segmentationId: this.id,
        });
      segmentationService.setActiveSegmentation(viewportId, this.id);
    }
    this.syncProject(this.project);
  }
  read() {
    if (!this.images) throw new Error("Initialize the annotation layer first.");
    const [width, height, depth] = this.asset.spatial_shape;
    const mask = new Uint8Array(width * height * depth);
    this.images.forEach((id, frame) => {
      const pixels = cache.getImage(id).voxelManager.getScalarData();
      if (pixels.length !== width * height)
        throw new Error("Segmentation frame dimensions changed.");
      const k = this.indices[frame];
      for (let j = 0; j < height; j++)
        for (let i = 0; i < width; i++)
          mask[(i * height + j) * depth + k] = pixels[j * width + i];
    });
    return mask;
  }
  write(mask) {
    const [width, height, depth] = this.asset.spatial_shape;
    if (mask.length !== width * height * depth)
      throw new Error(
        "Returned mask dimensions differ from the source volume.",
      );
    this.images.forEach((id, frame) => {
      const image = cache.getImage(id);
      const pixels = new Uint8Array(width * height);
      const k = this.indices[frame];
      for (let j = 0; j < height; j++)
        for (let i = 0; i < width; i++)
          pixels[j * width + i] = mask[(i * height + j) * depth + k];
      image.voxelManager.getScalarData().set(pixels);
    });
    segmentation.triggerSegmentationEvents.triggerSegmentationDataModified(
      this.id,
    );
    this.services.cornerstoneViewportService.getRenderingEngine()?.render();
  }
  scope() {
    const id = this.services.viewportGridService.getActiveViewportId();
    const viewport =
      this.services.cornerstoneViewportService.getCornerstoneViewport(id);
    if (!viewport) throw new Error("Select a source viewport first.");
    const displayed =
      viewport.getCurrentImageId?.() || viewport.getImageIds?.()?.[0];
    const uid =
      displayed &&
      decodeURIComponent(displayed).match(/\/instances\/([0-9.]+)/)?.[1];
    if (!uid || !this.source.instance_uids.includes(uid))
      throw new Error("Select the connected source series before annotation.");
    const camera = viewport.getCamera();
    const affine = mat4.fromValues(...this.asset.affine.flat());
    mat4.transpose(affine, affine);
    const lps = mat4.fromScaling(mat4.create(), [-1, -1, 1]);
    mat4.multiply(affine, lps, affine);
    const inverse = mat4.invert(mat4.create(), affine);
    const ijk = vec3.transformMat4(vec3.create(), camera.focalPoint, inverse);
    const directions = [0, 1, 2].map((i) =>
      vec3.normalize(vec3.create(), affine.slice(i * 4, i * 4 + 3)),
    );
    const chooseAxis = (vector) => {
      const scores = directions.map((d) => vec3.dot(d, vector));
      const axis = scores.reduce(
        (best, score, i) =>
          Math.abs(score) > Math.abs(scores[best]) ? i : best,
        0,
      );
      if (Math.abs(scores[axis]) < 0.9999)
        throw new Error(
          "Align the viewport with the source voxel axes before annotation.",
        );
      return [axis, scores[axis] < 0];
    };
    const [axis] = chooseAxis(camera.viewPlaneNormal);
    const index = Math.round(ijk[axis]);
    if (index < 0 || index >= this.asset.spatial_shape[axis])
      throw new Error("Viewport is outside the source volume.");
    const right = vec3.cross(
      vec3.create(),
      camera.viewUp,
      camera.viewPlaneNormal,
    );
    const down = vec3.negate(vec3.create(), camera.viewUp);
    const [columnAxis, flipColumns] = chooseAxis(right),
      [rowAxis, flipRows] = chooseAxis(down);
    const sourceAxes = [0, 1, 2].filter((a) => a !== axis);
    if (
      rowAxis === columnAxis ||
      !sourceAxes.includes(rowAxis) ||
      !sourceAxes.includes(columnAxis)
    )
      throw new Error("Viewport axes do not match the source grid.");
    const voi = viewport.getProperties().voiRange;
    if (!voi || voi.upper <= voi.lower)
      throw new Error("Set a valid intensity window first.");
    return {
      axis,
      index,
      window: [voi.lower, voi.upper],
      orientation: {
        transpose: rowAxis !== sourceAxes[0],
        flip_rows: flipRows,
        flip_columns: flipColumns,
      },
    };
  }
}

export function mergeMask(current, prediction, proposal, shape) {
  if (current.length !== prediction.length)
    throw new Error("Mask dimensions differ.");
  const result = current.slice(),
    labels = new Set(proposal.label_ids);
  const [width, height, depth] = shape;
  for (let i = 0; i < width; i++)
    for (let j = 0; j < height; j++)
      for (let k = 0; k < depth; k++) {
        if (
          proposal.slice &&
          [i, j, k][proposal.slice.axis] !== proposal.slice.index
        )
          continue;
        const n = (i * height + j) * depth + k;
        if (labels.has(result[n])) result[n] = 0;
        if (labels.has(prediction[n])) {
          if (result[n])
            throw new Error(
              "Prediction overlaps another locally edited structure.",
            );
          result[n] = prediction[n];
        }
      }
  return result;
}

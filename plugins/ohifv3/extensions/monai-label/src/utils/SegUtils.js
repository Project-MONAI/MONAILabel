function currentSegmentsInfo(segmentationService) {
  const info = {};
  const indices = new Set();

  const segmentations = segmentationService.getSegmentations();
  if (segmentations && segmentations.length) {
    const segmentation = segmentations[0];
    const { segments } = segmentation;
    for (const segment of segments) {
      if (segment) {
        info[segment.label] = {
          segmentIndex: segment.segmentIndex,
          color: segment.color,
        };
        indices.add(segment.segmentIndex);
      }
    }
  }
  return { info, indices };
}


export {
  currentSegmentsInfo,
};
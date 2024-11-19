function currentSegmentsInfo(segmentationService) {
  const info = {};
  const indices = new Set();

  const segmentations = segmentationService.getSegmentations();
  if (segmentations && Object.keys(segmentations).length > 0) {
    const segmentation = segmentations['0'];
    const { segments } = segmentation.config;
    for (const segmentIndex of Object.keys(segments)) {
      const segment = segments[segmentIndex];
      info[segment.label] = {
        segmentIndex: segment.segmentIndex,
        color: segment.color,
      };
      indices.add(segment.segmentIndex);
    }
  }
  return { info, indices };
}

export { currentSegmentsInfo };

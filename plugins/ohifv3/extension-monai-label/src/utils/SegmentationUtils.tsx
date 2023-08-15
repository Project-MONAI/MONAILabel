
function getImageIdsForDisplaySet(
  studies,
  StudyInstanceUID,
  SeriesInstanceUID
) {

  const study = studies.find(
    study => study.StudyInstanceUID === StudyInstanceUID
  );

  const displaySets = study.displaySets.filter(displaySet => {
    return displaySet.SeriesInstanceUID === SeriesInstanceUID;
  });

  if (displaySets.length > 1) {
    console.warn(
      'More than one display set with the same SeriesInstanceUID. This is not supported yet...'
    );
    // TODO -> We could make check the instance list and see if any match?
    // Do we split the segmentation into two cornerstoneTools segmentations if there are images in both series?
    // ^ Will that even happen?
  }

  const referencedDisplaySet = displaySets[0];
  return referencedDisplaySet.images.map(image => image.getImageId());
}

export {
  getImageIdsForDisplaySet,
};

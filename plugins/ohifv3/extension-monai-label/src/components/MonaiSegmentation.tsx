import React from 'react';
import { SegmentationGroup } from '@ohif/ui';

function MonaiSegmentation({ servicesManager, segmentation }) {
  const { segmentationService } = servicesManager.services;
  const getToolGroupIds = (segmentationId) => {
    const toolGroupIds =
      segmentationService.getToolGroupIdsWithSegmentation(segmentationId);

    return toolGroupIds;
  };

  const onToggleSegmentVisibility = (segmentationId, segmentIndex) => {
    const segmentation = segmentationService.getSegmentation(segmentationId);
    const segmentInfo = segmentation.segments[segmentIndex];
    const isVisible = !segmentInfo.isVisible;
    const toolGroupIds = getToolGroupIds(segmentationId);

    // Todo: right now we apply the visibility to all tool groups
    toolGroupIds.forEach((toolGroupId) => {
      segmentationService.setSegmentVisibility(
        segmentationId,
        segmentIndex,
        isVisible,
        toolGroupId
      );
    });
  };

  const onToggleSegmentationVisibility = (segmentationId) => {
    segmentationService.toggleSegmentationVisibility(segmentationId);
  };

  return (
    <SegmentationGroup
      key={segmentation.id}
      label={segmentation.label}
      id={segmentation.id}
      segmentCount={segmentation.segments.length}
      isVisible={segmentation.isVisible}
      isActive={segmentation.isActive}
      showAddSegment={false}
      segments={segmentation.segments}
      activeSegmentIndex={segmentation.activeSegmentIndex}
      onToggleSegmentVisibility={onToggleSegmentVisibility}
      onToggleSegmentationVisibility={onToggleSegmentationVisibility}
      showSegmentDelete={false}
      // onSegmentClick={onSegmentationClick}
      // isMinimized={segmentation.isMinimized}
      // onSegmentColorClick={onClickSegmentColor}
      // onSegmentationClick={onSegmentationClick}
      // onClickSegmentColor={onClickSegmentColor}
      // onSegmentationEdit={onSegmentationEdit}
      // onSegmentDelete={onSegmentDelete}
      // onToggleMinimizeSegmentation={onToggleMinimizeSegmentation}
      // onSegmentationConfigChange={onSegmentationConfigChange}
      // onSegmentationDelete={onSegmentationDelete}
      // onSegmentEdit={onSegmentEdit}
    />
  );
}

export default MonaiSegmentation;

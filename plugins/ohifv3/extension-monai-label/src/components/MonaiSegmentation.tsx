import React from 'react';
import { SegmentationGroup } from '@ohif/ui';

function MonaiSegmentation({ servicesManager, segmentation }) {
  const { segmentationService, toolGroupService } = servicesManager.services;
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

  const onSegmentClick = (segmentationId, segmentIndex) => {
    segmentationService.setActiveSegmentForSegmentation(
      segmentationId,
      segmentIndex
    );

    const toolGroupIds = getToolGroupIds(segmentationId);

    toolGroupIds.forEach((toolGroupId) => {
      // const toolGroupId =
      segmentationService.setActiveSegmentationForToolGroup(
        segmentationId,
        toolGroupId
      );
      segmentationService.jumpToSegmentCenter(
        segmentationId,
        segmentIndex,
        toolGroupId
      );
    });
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
      onSegmentClick={onSegmentClick}
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

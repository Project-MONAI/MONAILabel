/*
Copyright (c) MONAI Consortium
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

import cornerstoneTools from 'cornerstone-tools';
import cornerstone from 'cornerstone-core';

const { setters, getters } = cornerstoneTools.getModule('segmentation');

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

/**
 * Gets an array of LabelMap.
 * Each LabelMap is an array of segments.
 *
 * Note that this LabelMap we have here is different from cornerstone's.
 *
 * @param element
 */
function getLabelMaps(element) {
  let labelmaps = [];
  if (!element) {
    console.warn('element is empty... weird...');
    return labelmaps;
  }

  const segmentationModule = cornerstoneTools.getModule('segmentation');
  const { labelmaps3D } = getters.labelmaps3D(element);
  if (!labelmaps3D) {
    console.debug('LabelMap3D is empty.. so zero segments');
    return labelmaps;
  }
  console.debug(labelmaps3D);

  for (let i = 0; i < labelmaps3D.length; i++) {
    let segments = [];
    const labelmap3D = labelmaps3D[i];

    // TODO:: which one is standard metadata.data[] or metadata[] ???
    const metadata =
      labelmap3D && labelmap3D.metadata && labelmap3D.metadata.data
        ? labelmap3D.metadata.data
        : null;
    const colorLutTable =
      segmentationModule.state.colorLutTables[labelmap3D.colorLUTIndex];
    console.debug('Labelmap3D Index = ' + i);
    console.debug('labelmap3D.colorLUTIndex = ' + labelmap3D.colorLUTIndex);
    console.debug(labelmap3D);
    console.debug(colorLutTable);

    if (!metadata) {
      console.warn('Missing Meta Data for Label; so ignore');
    } else {
      for (let j = 1; j < metadata.length; j++) {
        const meta = metadata[j];
        if (!meta) {
          continue;
        }

        console.debug('SegmentNumber = ' + meta.SegmentNumber);
        console.debug(meta);

        const id = i + '+' + meta.SegmentNumber;
        const color = colorLutTable[meta.SegmentNumber];
        const segmentItem = {
          id: id,
          labelmapIndex: i,
          segmentIndex: meta.SegmentNumber,
          color: color,
          meta: meta,
          name: meta.SegmentLabel,
          description: meta.SegmentDescription,
        };
        segments.push(segmentItem);
      }
    }
    labelmaps.push(segments);
  }

  return labelmaps;
}

function flattenLabelmaps(labelmaps) {
  return [].concat.apply([], labelmaps);
}

function getFirstSegmentId(element) {
  const labelmaps = getLabelMaps(element);
  const segments = flattenLabelmaps(labelmaps);
  console.debug(segments);
  return segments && segments.length ? segments[0].id : null;
}

/**
 * Creates a segment.
 *
 * @param {Object} element A cornerstone element
 * @param {string} label Name of the segment
 * @param description
 * @param color
 * @param {boolean} newLabelMap Whether to put this segment in a new labelmap3D or not
 * @param labelMeta
 * @returns {{labelmapIndex: *, segmentIndex: number}}
 */
function createSegment(
  element,
  label,
  description = '',
  color = null,
  newLabelMap = false,
  labelMeta = null
) {
  labelMeta = labelMeta
    ? labelMeta
    : {
        SegmentedPropertyCategoryCodeSequence: {
          CodeValue: 'T-D0050',
          CodingSchemeDesignator: 'SRT',
          CodeMeaning: 'Tissue',
        },
        SegmentNumber: 1,
        SegmentLabel: label ? label : 'label-0-1',
        SegmentDescription: description,
        SegmentAlgorithmType: 'SEMIAUTOMATIC',
        SegmentAlgorithmName: 'MONAI',
        SegmentedPropertyTypeCodeSequence: {
          CodeValue: 'T-D0050',
          CodingSchemeDesignator: 'SRT',
          CodeMeaning: 'Tissue',
        },
      };

  if (newLabelMap) {
    const labelmaps = getLabelMaps(element);
    let nextLabelmapIndex = labelmaps ? labelmaps.length : 0; // Reuse First Empty LabelMap
    for (let i = 0; i < labelmaps.length; i++) {
      if (!labelmaps[i] || !labelmaps[i].length) {
        nextLabelmapIndex = i;
        break;
      }
    }

    console.debug('Next LabelmapIndex: ' + nextLabelmapIndex);
    setters.activeLabelmapIndex(element, nextLabelmapIndex);
  }

  // this labelmap2D function will create a labelmap3D if a labelmap does
  // not yet exist. it will also generate a labelmap2D for the currentImageIndex
  // if it does not yet exist.
  // refer to: https://github.com/cornerstonejs/cornerstoneTools/blob/master/src/store/modules/segmentationModule/getLabelmap2D.js
  const { labelmap3D, activeLabelmapIndex } = getters.labelmap2D(element);
  console.debug('activeLabelmapIndex: ' + activeLabelmapIndex);

  // Add new colorLUT if required for new labelmapIndex
  const { state } = cornerstoneTools.getModule('segmentation');
  console.debug(state.colorLutTables);
  if (state.colorLutTables.length <= activeLabelmapIndex) {
    console.debug('Adding new Color LUT Table for: ' + activeLabelmapIndex);
    setters.colorLUT(activeLabelmapIndex);
    console.debug(state.colorLutTables);
    labelmap3D.colorLUTIndex = activeLabelmapIndex;
  }

  console.debug('labelmap3D.colorLUTIndex = ' + labelmap3D.colorLUTIndex);

  // TODO:: which one is standard metadata.data[] or metadata[] ???
  if (!labelmap3D.metadata || !labelmap3D.metadata.data) {
    labelmap3D.metadata = { data: [undefined] };
  }

  const { metadata } = labelmap3D;
  let nextSegmentId = 1;
  for (let i = 1; i < metadata.data.length; i++) {
    if (nextSegmentId === metadata.data[i].SegmentNumber) {
      nextSegmentId++;
    } else {
      break;
    }
  }
  console.debug(
    'Next Segment: ' + nextSegmentId + '; LabelMap: ' + activeLabelmapIndex
  );

  labelMeta.SegmentNumber = nextSegmentId;
  labelMeta.SegmentLabel = label
    ? label
    : 'label_' + activeLabelmapIndex + '-' + nextSegmentId;

  if (nextSegmentId === metadata.data.length) {
    metadata.data.push(labelMeta);
  } else {
    metadata.data.splice(nextSegmentId, 0, labelMeta);
  }
  setters.activeSegmentIndex(element, nextSegmentId);

  if (color) {
    const segmentationModule = cornerstoneTools.getModule('segmentation');
    const colorLutTable =
      segmentationModule.state.colorLutTables[labelmap3D.colorLUTIndex];

    colorLutTable[nextSegmentId][0] = color.r;
    colorLutTable[nextSegmentId][1] = color.g;
    colorLutTable[nextSegmentId][2] = color.b;
  }

  return {
    id: activeLabelmapIndex + '+' + nextSegmentId,
    labelmapIndex: activeLabelmapIndex,
    segmentIndex: nextSegmentId,
  };
}

function getSegmentInfo(element, labelmapIndex, segmentIndex) {
  var name = '';
  var description = '';
  var color = '';

  const labelmap3D = getters.labelmap3D(element, labelmapIndex);
  if (!labelmap3D) {
    console.warn('Missing Label; so ignore');
    return { name, description, color };
  }

  const metadata = labelmap3D.metadata.data
    ? labelmap3D.metadata.data
    : labelmap3D.metadata;
  if (!metadata) {
    console.warn('Missing Meta; so ignore');
    return { name, description, color };
  }

  name = metadata[segmentIndex].SegmentLabel;
  description = metadata[segmentIndex].SegmentDescription;

  const segmentationModule = cornerstoneTools.getModule('segmentation');
  const colorLutTable =
    segmentationModule.state.colorLutTables[labelmap3D.colorLUTIndex];
  color = colorLutTable[segmentIndex];

  return { name, description, color };
}

function updateSegment(
  element,
  labelmapIndex,
  segmentIndex,
  buffer,
  numberOfFrames,
  operation,
  slice = -1
) {
  const labelmap3D = getters.labelmap3D(element, labelmapIndex);
  if (!labelmap3D) {
    console.warn('Missing Label; so ignore');
    return;
  }

  const metadata = labelmap3D.metadata.data
    ? labelmap3D.metadata.data
    : labelmap3D.metadata;
  if (!metadata) {
    console.warn('Missing Meta; so ignore');
    return;
  }

  // Segments on LabelMap
  const segmentsOnLabelmap = metadata
    .filter(x => x && x.SegmentNumber)
    .map(x => x.SegmentNumber);
  segmentsOnLabelmap.unshift(0);
  console.debug(segmentsOnLabelmap);

  const segmentOffset = segmentIndex - 1;
  console.debug('labelmapIndex: ' + labelmapIndex);
  console.debug('segmentIndex: ' + segmentIndex);
  console.debug('segmentOffset: ' + segmentOffset);

  const labelmaps2D = labelmap3D.labelmaps2D;
  const slicelengthInBytes = buffer.byteLength / numberOfFrames;

  console.debug('labelmap2d length:' + labelmaps2D.length);
  if (!labelmaps2D.length || labelmaps2D.length < 1) {
    console.debug('First time update...');
    operation = undefined;
  }

  // Update Buffer (2D/3D)
  let srcBuffer = labelmap3D.buffer;
  let useSourceBuffer = false;
  for (let i = 0; i < numberOfFrames; i++) {
    if (slice >= 0 && i !== slice) {
      // do only one slice (in case of 3D Volume but 2D result e.g. Deeprow2D)
      continue;
    }

    // no segments in this slice
    if (
      !labelmaps2D[i] ||
      !labelmaps2D[i].segmentsOnLabelmap ||
      !labelmaps2D[i].segmentsOnLabelmap.length
    ) {
      operation = 'override';
    }

    const sliceOffset = slicelengthInBytes * i;
    const sliceLength = slicelengthInBytes / 2;

    let pixelData = new Uint16Array(buffer, sliceOffset, sliceLength);
    let srcPixelData = new Uint16Array(srcBuffer, sliceOffset, sliceLength);

    if (operation === 'overlap' || operation === 'override') {
      useSourceBuffer = true;
    }

    for (let j = 0; j < pixelData.length; j++) {
      if (operation === 'overlap') {
        if (pixelData[j] > 0) {
          srcPixelData[j] = pixelData[j] + segmentOffset;
        }
      } else if (operation === 'override') {
        if (srcPixelData[j] === segmentIndex) {
          srcPixelData[j] = 0;
        }
        if (pixelData[j] > 0) {
          srcPixelData[j] = pixelData[j] + segmentOffset;
        }
      } else {
        if (pixelData[j] > 0) {
          pixelData[j] = pixelData[j] + segmentOffset;
        }
      }
    }

    pixelData = useSourceBuffer ? srcPixelData : pixelData;
    labelmaps2D[i] = { pixelData, segmentsOnLabelmap };
  }

  labelmap3D.buffer = useSourceBuffer ? srcBuffer : buffer;
  cornerstone.updateImage(element);
}

function updateSegmentMeta(
  element,
  labelmapIndex,
  segmentIndex,
  label = undefined,
  desc = undefined,
  color = undefined
) {
  const labelmap3D = getters.labelmap3D(element, labelmapIndex);
  if (!labelmap3D) {
    console.warn('Missing Label; so ignore');
    return;
  }

  const metadata = labelmap3D.metadata.data
    ? labelmap3D.metadata.data
    : labelmap3D.metadata;
  if (!metadata) {
    console.warn('Missing Meta; so ignore');
    return;
  }

  if (label) {
    metadata[segmentIndex].SegmentLabel = label;
  }
  if (desc) {
    metadata[segmentIndex].SegmentDescription = desc;
  }

  if (color) {
    const segmentationModule = cornerstoneTools.getModule('segmentation');
    const colorLutTable =
      segmentationModule.state.colorLutTables[labelmap3D.colorLUTIndex];

    colorLutTable[segmentIndex][0] = color.r;
    colorLutTable[segmentIndex][1] = color.g;
    colorLutTable[segmentIndex][2] = color.b;
  }
}

function deleteSegment(element, labelmapIndex, segmentIndex) {
  console.debug(
    'calling delete segment with ' + labelmapIndex + ' and ' + segmentIndex
  );
  if (!element || !segmentIndex) {
    return;
  }

  const labelmap3D = getters.labelmap3D(element, labelmapIndex);
  if (!labelmap3D) {
    console.warn('Missing Label; so ignore');
    return;
  }

  // TODO:: which one is standard metadata.data[] or metadata[] ???
  if (labelmap3D.metadata && labelmap3D.metadata.data) {
    let newData = [undefined];
    for (let i = 1; i < labelmap3D.metadata.data.length; i++) {
      const meta = labelmap3D.metadata.data[i];
      if (segmentIndex !== meta.SegmentNumber) {
        newData.push(meta);
      }
    }
    labelmap3D.metadata.data = newData;
  }

  // remove segments mapping
  const labelmaps2D = labelmap3D.labelmaps2D;
  for (let i = 0; i < labelmaps2D.length; i++) {
    const labelmap2D = labelmaps2D[i];
    if (labelmap2D && labelmap2D.segmentsOnLabelmap.includes(segmentIndex)) {
      const indexOfSegment = labelmap2D.segmentsOnLabelmap.indexOf(
        segmentIndex
      );
      labelmap2D.segmentsOnLabelmap.splice(indexOfSegment, 1);
    }
  }

  // cleanup buffer
  let z = new Uint16Array(labelmap3D.buffer);
  for (let i = 0; i < z.length; i++) {
    if (z[i] === segmentIndex) {
      z[i] = 0;
    }
  }
  cornerstone.updateImage(element);
}

function clearSegment(element, labelmapIndex, segmentIndex) {
  console.debug(
    'calling clear segment with ' + labelmapIndex + ' and ' + segmentIndex
  );
  if (!element || !segmentIndex) {
    return;
  }

  const labelmap3D = getters.labelmap3D(element, labelmapIndex);
  if (!labelmap3D) {
    console.warn('Missing Label; so ignore');
    return;
  }

  // cleanup buffer
  let z = new Uint16Array(labelmap3D.buffer);
  for (let i = 0; i < z.length; i++) {
    if (z[i] === segmentIndex) {
      z[i] = 0;
    }
  }
  cornerstone.updateImage(element);
}

export {
  getImageIdsForDisplaySet,
  getLabelMaps,
  flattenLabelmaps,
  createSegment,
  clearSegment,
  getSegmentInfo,
  updateSegment,
  deleteSegment,
  updateSegmentMeta,
  getFirstSegmentId,
};

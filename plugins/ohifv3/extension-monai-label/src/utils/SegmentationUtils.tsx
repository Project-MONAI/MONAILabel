import {
  getLabelColor,
} from './GenericUtils';


function createSegmentMetadata(
  label,
  segmentId,
  description = '',
  newLabelMap = false,
) {

  const labelMeta = {
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
    console.debug('Logic to create a new segment');
  }

  const color = getLabelColor(label)

  const rgbColor = [];
  for (let key in color) {
    rgbColor.push(color[key]);
    }

  rgbColor.push(255);

  return {
    id: '0+' + segmentId,
    color: rgbColor,
    labelmapIndex: 0,
    name: label,
    segmentIndex: segmentId,
    description: description,
    meta: labelMeta,
  };
}

export {
  createSegmentMetadata,
};

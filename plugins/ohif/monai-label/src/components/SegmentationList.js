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

import React, { Component } from 'react';
import PropTypes from 'prop-types';

import { Icon } from '@ohif/ui';
import { UIModalService, UINotificationService } from '@ohif/core';
import cornerstone from 'cornerstone-core';
import cornerstoneTools from 'cornerstone-tools';

import './SegmentationList.styl';

import {
  createSegment,
  clearSegment,
  deleteSegment,
  flattenLabelmaps,
  getFirstSegmentId,
  getLabelMaps,
  getSegmentInfo,
  updateSegment,
  updateSegmentMeta,
} from '../utils/SegmentationUtils';
import SegmentationLabelForm from './SegmentationLabelForm';
import {
  hexToRgb,
  randomName,
  randomRGB,
  rgbToHex,
  getLabelColor,
} from '../utils/GenericUtils';
import SegmentationReader from '../utils/SegmentationReader';

const ColoredCircle = ({ color }) => {
  return (
    <span
      className="segColor"
      style={{ backgroundColor: `rgba(${color.join(',')})` }}
    />
  );
};

ColoredCircle.propTypes = {
  color: PropTypes.array.isRequired,
};

export default class SegmentationList extends Component {
  static propTypes = {
    viewConstants: PropTypes.any,
    onSegmentCreated: PropTypes.func,
    onSegmentUpdated: PropTypes.func,
    onSegmentDeleted: PropTypes.func,
    onSegmentSelected: PropTypes.func,
  };

  constructor(props) {
    super(props);

    this.notification = UINotificationService.create({});
    this.uiModelService = UIModalService.create({});
    const { element } = this.props.viewConstants;
    const labelmaps = getLabelMaps(element);
    const segments = flattenLabelmaps(labelmaps);

    this.state = {
      element: element,
      segments: segments,
      selectedSegmentId: segments && segments.length ? segments[0].id : null,
      header: null,
    };
  }

  getSelectedActiveIndex = () => {
    const id = this.state.selectedSegmentId;
    if (id) {
      let index = id.split('+').map(Number);
      return { id, labelmapIndex: index[0], segmentIndex: index[1] };
    }
    return null;
  };

  getIndexByName = name => {
    console.info(this.state.segments);
    for (let i = 0; i < this.state.segments.length; i++) {
      if (this.state.segments[i].meta.SegmentLabel == name) {
        let id = this.state.segments[i].id;
        let index = id.split('+').map(Number);
        return { id, labelmapIndex: index[0], segmentIndex: index[1] };
      }
    }
    return null;
  };

  getNameByIndex = index => {
    const id = index.id;
    for (let i = 0; i < this.state.segments.length; i++) {
      if (this.state.segments[i].id == id) {
        return this.state.segments[i].meta.SegmentLabel;
      }
    }
    return null;
  };

  onAddSegment = (
    name,
    description,
    color,
    selectActive = true,
    newLabelMap = false
  ) => {
    this.uiModelService.hide();

    const { element } = this.props.viewConstants;
    const { id } = createSegment(
      element,
      name,
      description,
      hexToRgb(color),
      newLabelMap
    );
    this.refreshSegTable(id, selectActive);

    if (this.props.onSegmentCreated) {
      this.props.onSegmentCreated(id);
    }
  };

  onUpdateSegment = (name, description, color) => {
    this.uiModelService.hide();

    const { element } = this.props.viewConstants;
    const activeIndex = this.getSelectedActiveIndex();
    updateSegmentMeta(
      element,
      activeIndex.labelmapIndex,
      activeIndex.segmentIndex,
      name,
      description,
      hexToRgb(color)
    );
    this.refreshSegTable(activeIndex.id);

    if (this.props.onSegmentUpdated) {
      this.props.onSegmentUpdated(activeIndex.id);
    }
  };

  onSelectSegment = evt => {
    let id = evt.currentTarget.value;
    this.setState({ selectedSegmentId: id });
  };

  onClickAddSegment = () => {
    this.uiModelService.show({
      content: SegmentationLabelForm,
      title: 'Add New Label',
      contentProps: {
        name: randomName(),
        description: '',
        color: randomRGB(),
        onSubmit: this.onAddSegment,
      },
      customClassName: 'segmentationLabelForm',
      shouldCloseOnEsc: true,
    });
  };

  onClickEditSegment = () => {
    const { element } = this.props.viewConstants;
    const activeIndex = this.getSelectedActiveIndex();
    const { name, description, color } = getSegmentInfo(
      element,
      activeIndex.labelmapIndex,
      activeIndex.segmentIndex
    );

    this.uiModelService.show({
      content: SegmentationLabelForm,
      title: 'Edit Label',
      contentProps: {
        name: name,
        description: description,
        color: rgbToHex(
          Math.floor(color[0]),
          Math.floor(color[1]),
          Math.floor(color[2])
        ),
        onSubmit: this.onUpdateSegment,
      },
      customClassName: 'segmentationLabelForm',
      shouldCloseOnEsc: true,
    });
  };

  onUpdateLabelOrDesc = (id, evt, label) => {
    const { element } = this.props.viewConstants;
    let index = id.split('+').map(Number);
    const labelmapIndex = index[0];
    const segmentIndex = index[1];

    updateSegmentMeta(
      element,
      labelmapIndex,
      segmentIndex,
      label ? evt.currentTarget.textContent : undefined,
      label ? undefined : evt.currentTarget.textContent
    );
  };

  onClickDeleteSegment = () => {
    const { element } = this.props.viewConstants;
    const activeIndex = this.getSelectedActiveIndex();

    deleteSegment(element, activeIndex.labelmapIndex, activeIndex.segmentIndex);
    this.setState({ selectedSegmentId: null });
    this.refreshSegTable(null);

    if (this.props.onSegmentDeleted) {
      this.props.onSegmentDeleted(activeIndex.id);
    }
  };

  onDeleteSegmentByName = name => {
    this.onDeleteSegmentByIndex(this.getIndexByName(name));
  };

  onDeleteSegmentByIndex = selectedIndex => {
    const { element } = this.props.viewConstants;
    console.info(selectedIndex);

    if (selectedIndex) {
      deleteSegment(
        element,
        selectedIndex.labelmapIndex,
        selectedIndex.segmentIndex
      );
      this.setState({ selectedSegmentId: null });
      this.refreshSegTable(null);

      if (this.props.onSegmentDeleted) {
        this.props.onSegmentDeleted(selectedIndex.id);
      }
    } else {
      console.info(
        'onDeleteSegmentByIndex: segment ' +
          selectedIndex +
          ' not found, skipping...'
      );
    }
  };

  onClearSegmentByName = name => {
    this.onClearSegmentByIndex(this.getIndexByName(name));
  };

  onClearSegmentByIndex = selectedIndex => {
    const { element } = this.props.viewConstants;
    console.info(selectedIndex);

    if (selectedIndex) {
      clearSegment(
        element,
        selectedIndex.labelmapIndex,
        selectedIndex.segmentIndex
      );
    } else {
      console.info(
        'onClearSegmentByIndex: segment ' +
          selectedIndex +
          ' not found, skipping...'
      );
    }
  };

  onClickExportSegments = () => {
    const { getters } = cornerstoneTools.getModule('segmentation');
    const { labelmaps3D } = getters.labelmaps3D(
      this.props.viewConstants.element
    );
    if (!labelmaps3D) {
      console.info('LabelMap3D is empty.. so zero segments');
      return;
    }

    this.notification.show({
      title: 'MONAI Label',
      message: 'Preparing the labelmap to download as .nrrd',
      type: 'info',
      duration: 5000,
    });

    for (let i = 0; i < labelmaps3D.length; i++) {
      const labelmap3D = labelmaps3D[i];
      if (!labelmap3D) {
        console.warn('Missing Label; so ignore');
        continue;
      }

      const metadata = labelmap3D.metadata.data
        ? labelmap3D.metadata.data
        : labelmap3D.metadata;
      if (!metadata || !metadata.length) {
        console.warn('Missing Meta; so ignore');
        continue;
      }

      SegmentationReader.serializeNrrdCompressed(
        this.state.header,
        labelmap3D.buffer,
        'segment-' + i + '.nrrd'
      );
    }
  };

  refreshSegTable = (id, selectActive = true) => {
    const { element } = this.props.viewConstants;

    const labelmaps = getLabelMaps(element);
    const segments = flattenLabelmaps(labelmaps);
    if (!segments.length) {
      id = undefined;
    } else if (!id) {
      id = segments[segments.length - 1].id;
    }

    // do not select if index is from a scribbles volume
    // scribbles volumes exist in table
    // but wont be shown or selectable through the table ui
    for (let i = 0; i < segments.length; i++) {
      if (
        segments[i].meta.SegmentLabel.includes('scribbles') &&
        segments[i].id == id
      ) {
        id = null;
        break;
      }
    }

    if (id && selectActive) {
      this.setState({ segments: segments, selectedSegmentId: id });
    } else {
      this.setState({ segments: segments });
    }
  };

  setActiveSegment = () => {
    const { element } = this.props.viewConstants;
    const activeIndex = this.getSelectedActiveIndex();

    const { setters } = cornerstoneTools.getModule('segmentation');
    const { labelmapIndex, segmentIndex } = activeIndex;

    setters.activeLabelmapIndex(element, labelmapIndex);
    setters.activeSegmentIndex(element, segmentIndex);

    // Refresh
    cornerstone.updateImage(element);

    if (this.props.onSegmentSelected) {
      this.props.onSegmentSelected(activeIndex.id);
    }
  };

  updateView = async (
    response,
    labels,
    operation,
    slice,
    overlap,
    selectedIndex
  ) => {
    const { element, numberOfFrames } = this.props.viewConstants;
    if (!selectedIndex) {
      selectedIndex = this.getSelectedActiveIndex();
    }
    const { header, image } = SegmentationReader.parseNrrdData(response.data);
    this.setState({ header: header });
    console.debug(selectedIndex);

    if (labels) {
      let i = 0;
      for (var label in labels) {
        if (Array.isArray(labels)) {
          label = labels[label];
        }

        if (label === 'background') {
          console.debug('Ignore Background...');
          continue;
        }

        const resp = createSegment(
          element,
          label,
          '',
          getLabelColor(label),
          i === 0 ? !overlap : false
        );
        if (i === 0) {
          selectedIndex = resp;
        }
        i++;

        if (this.state.selectedSegmentId) {
          this.refreshSegTable();
        } else {
          this.refreshSegTable(selectedIndex.id);
        }
      }
    }

    if (!operation && overlap) {
      operation = 'overlap';
    }

    updateSegment(
      element,
      selectedIndex.labelmapIndex,
      selectedIndex.segmentIndex,
      image,
      numberOfFrames,
      operation,
      slice
    );
  };

  componentDidUpdate(prevProps, prevState, snapshot) {
    if (prevState.selectedSegmentId !== this.state.selectedSegmentId) {
      if (this.state.selectedSegmentId) {
        this.setActiveSegment();
      }
    }
  }

  render() {
    const segmentId = this.state.selectedSegmentId
      ? this.state.selectedSegmentId
      : getFirstSegmentId(this.props.viewConstants.element);
    console.debug('render seg list: ' + segmentId);

    return (
      <div className="segmentationList">
        <table width="100%">
          <tbody>
            <tr>
              <td>
                <button
                  className="segButton"
                  onClick={this.onClickAddSegment}
                  title="Add Segment"
                >
                  <Icon name="plus" width="12px" height="12px" />
                </button>
                &nbsp;
                <button
                  className="segButton"
                  onClick={this.onClickEditSegment}
                  title="Edit Selected Segment"
                  disabled={!segmentId}
                >
                  <Icon name="edit" width="12px" height="12px" />
                </button>
                &nbsp;
                <button
                  className="segButton"
                  onClick={this.onClickDeleteSegment}
                  title="Delete Selected Segment"
                  disabled={!segmentId}
                >
                  <Icon name="trash" width="12px" height="12px" />
                </button>
              </td>
              <td align="right">
                <button
                  className="segButton"
                  onClick={this.onClickExportSegments}
                  title={'Download Segments'}
                  disabled={!this.state.header}
                >
                  <Icon name="save" width="12px" height="12px" />
                </button>
              </td>
            </tr>
          </tbody>
        </table>

        <div className="segSection">
          <table className="segTable">
            <thead>
              <tr>
                <th width="2%">#</th>
                <th width="8%">Color</th>
                <th width="60%">Name</th>
                <th width="30%">Desc</th>
              </tr>
            </thead>
            <tbody>
              {this.state.segments
                .filter(
                  // do not display anything related to scribbles
                  function(seg) {
                    return !seg.meta.SegmentLabel.includes('scribbles');
                  }
                )
                .map(seg => (
                  <tr key={seg.id}>
                    <td>
                      <input
                        type="radio"
                        name="segitem"
                        value={seg.id}
                        checked={seg.id === this.state.selectedSegmentId}
                        onChange={this.onSelectSegment}
                      />
                    </td>
                    <td>
                      <ColoredCircle color={seg.color} />
                    </td>
                    <td
                      className="segEdit"
                      contentEditable="true"
                      suppressContentEditableWarning="true"
                      onKeyUp={evt =>
                        this.onUpdateLabelOrDesc(seg.id, evt, true)
                      }
                    >
                      {seg.meta.SegmentLabel}
                    </td>
                    <td
                      contentEditable="true"
                      suppressContentEditableWarning="true"
                      onKeyUp={evt =>
                        this.onUpdateLabelOrDesc(seg.id, evt, false)
                      }
                    >
                      {seg.meta.SegmentDescription}
                    </td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      </div>
    );
  }
}

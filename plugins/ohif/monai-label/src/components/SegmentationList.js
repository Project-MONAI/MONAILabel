import React, { Component } from 'react';
import PropTypes from 'prop-types';

import { Icon } from '@ohif/ui';
import { UIModalService } from '@ohif/core';
import cornerstone from 'cornerstone-core';
import cornerstoneTools from 'cornerstone-tools';

import './SegmentationList.styl';

import {
  createSegment,
  deleteSegment,
  flattenLabelmaps,
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

    this.uiModelService = UIModalService.create({});
    const { element } = this.props.viewConstants;
    const labelmaps = getLabelMaps(element);
    const segments = flattenLabelmaps(labelmaps);

    this.state = {
      element: element,
      segments: segments,
      selectedSegmentId: segments && segments.length ? segments[0].id : null,
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

  onAddSegment = (name, description, color) => {
    this.uiModelService.hide();

    const { element } = this.props.viewConstants;
    const { id } = createSegment(element, name, description, hexToRgb(color));
    this.refreshSegTable(id);

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
    this.refreshSegTable(null);

    if (this.props.onSegmentDeleted) {
      this.props.onSegmentDeleted(activeIndex.id);
    }
  };

  refreshSegTable = id => {
    const { element } = this.props.viewConstants;

    const labelmaps = getLabelMaps(element);
    const segments = flattenLabelmaps(labelmaps);
    if (!segments.length) {
      id = undefined;
    } else if (!id) {
      id = segments[segments.length - 1].id;
    }

    if (id) {
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

  updateView = async (response, labels, operation, slice, overlap) => {
    const { element, numberOfFrames } = this.props.viewConstants;
    let activeIndex = this.getSelectedActiveIndex();
    const { image } = SegmentationReader.parseNrrdData(response.data);

    if (labels) {
      for (let i = 0; i < labels.length; i++) {
        const resp = createSegment(
          element,
          labels[i],
          '',
          getLabelColor(labels[i]),
          i === 0 ? !overlap : false
        );
        if (i === 0) {
          activeIndex = resp;
        }

        if (this.state.selectedSegmentId) {
          this.refreshSegTable();
        } else {
          this.refreshSegTable(activeIndex.id);
        }
      }
    }

    if (!operation && overlap) {
      operation = 'overlap';
    }

    updateSegment(
      element,
      activeIndex.labelmapIndex,
      activeIndex.segmentIndex,
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
                  disabled={!this.state.selectedSegmentId}
                >
                  <Icon name="edit" width="12px" height="12px" />
                </button>
                &nbsp;
                <button
                  className="segButton"
                  onClick={this.onClickDeleteSegment}
                  title="Delete Selected Segment"
                  disabled={!this.state.selectedSegmentId}
                >
                  <Icon name="trash" width="12px" height="12px" />
                </button>
              </td>
              <td align="right">
                <button
                  className="segButton"
                  onClick={this.onClickExportSegments}
                  title={'Download Segments'}
                  disabled
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
              {this.state.segments.map(seg => (
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
                    onKeyUp={evt => this.onUpdateLabelOrDesc(seg.id, evt, true)}
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

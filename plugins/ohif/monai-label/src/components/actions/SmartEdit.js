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

import React from 'react';

import './SmartEdit.styl';
import cornerstone from 'cornerstone-core';
import cornerstoneTools from 'cornerstone-tools';
import ModelSelector from '../ModelSelector';
import BaseTab from './BaseTab';
import { getFirstSegmentId } from '../../utils/SegmentationUtils';

export default class SmartEdit extends BaseTab {
  constructor(props) {
    super(props);

    this.modelSelector = React.createRef();

    this.state = {
      segmentId: null,
      currentPoint: null,
      deepgrowPoints: new Map(),
      currentEvent: null,
      currentModel: null,
    };
  }

  onSelectModel = model => {
    this.setState({ currentModel: model });
  };

  onDeepgrow = async () => {
    const { info, viewConstants } = this.props;
    const image = viewConstants.SeriesInstanceUID;
    const model = this.modelSelector.current.currentModel();
    const segmentId = this.state.segmentId
      ? this.state.segmentId
      : getFirstSegmentId(viewConstants.element);
    if (segmentId && !this.state.segmentId) {
      this.onSegmentSelected(segmentId);
    }

    const is3D = info.models[model].dimension === 3;
    if (!segmentId) {
      this.notification.show({
        title: 'MONAI Label',
        message: 'Please create/select a label first',
        type: 'warning',
      });
      return;
    }

    const points = this.state.deepgrowPoints.get(segmentId);
    if (!points.length) {
      return;
    }

    const currentPoint = points[points.length - 1];
    const foreground = points
      .filter(p => (is3D || p.z === currentPoint.z) && !p.data.ctrlKey)
      .map(p => [p.x, p.y, p.z]);
    const background = points
      .filter(p => (is3D || p.z === currentPoint.z) && p.data.ctrlKey)
      .map(p => [p.x, p.y, p.z]);

    const config = this.props.onOptionsConfig();
    const params =
      config && config.infer && config.infer[model] ? config.infer[model] : {};

    const cursor = viewConstants.element.style.cursor;
    viewConstants.element.style.cursor = 'wait';
    const response = await this.props
      .client()
      .deepgrow(model, image, foreground, background, params);
    viewConstants.element.style.cursor = cursor;

    if (response.status !== 200) {
      this.notification.show({
        title: 'MONAI Label',
        message: 'Failed to Run Deepgrow',
        type: 'error',
        duration: 3000,
      });
    } else {
      await this.props.updateView(
        response,
        null,
        'override',
        is3D ? -1 : currentPoint.z
      );
    }
  };

  deepgrowClickEventHandler = async evt => {
    if (!evt || !evt.detail) {
      console.info('Not a valid event; So Ignore');
      return;
    }

    const segmentId = this.state.segmentId
      ? this.state.segmentId
      : getFirstSegmentId(this.props.viewConstants.element);
    if (segmentId && !this.state.segmentId) {
      this.onSegmentSelected(segmentId);
    }

    if (!segmentId) {
      this.notification.show({
        title: 'MONAI Label',
        message: 'Please create/select a label first',
        type: 'warning',
      });
      return;
    }

    let points = this.state.deepgrowPoints.get(segmentId);
    if (!points) {
      points = [];
      this.state.deepgrowPoints.set(segmentId, points);
    }

    const pointData = this.getPointData(evt);
    points.push(pointData);
    await this.onDeepgrow(this.state.model);
  };

  getPointData = evt => {
    const { x, y, imageId } = evt.detail;
    const z = this.props.viewConstants.imageIdsToIndex.get(imageId);

    console.debug('X: ' + x + '; Y: ' + y + '; Z: ' + z);
    return { x, y, z, data: evt.detail, imageId };
  };

  onSegmentDeleted = id => {
    this.clearPoints(id);
    this.setState({ segmentId: null });
  };
  onSegmentSelected = id => {
    this.initPoints(id);
    this.setState({ segmentId: id });
  };

  initPoints = id => {
    const pointsAll = this.state.deepgrowPoints;
    const segmentId = !id ? this.state.segmentId : id;
    if (!segmentId) {
      return;
    }

    const points = pointsAll.get(segmentId);
    if (!points) {
      return;
    }

    const { element } = this.props.viewConstants;
    for (let i = 0; i < points.length; i++) {
      const enabledElement = cornerstone.getEnabledElement(element);
      const oldImageId = enabledElement.image.imageId;

      for (let i = 0; i < points.length; ++i) {
        let { imageId, data } = points[i];
        enabledElement.image.imageId = imageId;
        cornerstoneTools.addToolState(element, 'DeepgrowProbe', data);
      }
      enabledElement.image.imageId = oldImageId;
    }

    // Refresh
    cornerstone.updateImage(element);
  };

  clearPoints = id => {
    const pointsAll = this.state.deepgrowPoints;
    const segmentId = !id ? this.state.segmentId : id;
    if (!segmentId) {
      return;
    }

    const points = pointsAll.get(segmentId);
    if (!points) {
      return;
    }

    const { element } = this.props.viewConstants;
    const enabledElement = cornerstone.getEnabledElement(element);
    const oldImageId = enabledElement.image.imageId;

    for (let i = 0; i < points.length; ++i) {
      let { imageId } = points[i];
      enabledElement.image.imageId = imageId;
      cornerstoneTools.clearToolState(element, 'DeepgrowProbe');
    }
    enabledElement.image.imageId = oldImageId;
    cornerstone.updateImage(element);
    pointsAll.delete(segmentId);
  };

  onSelectActionTab = evt => {
    this.props.onSelectActionTab(evt.currentTarget.value);
  };

  onEnterActionTab = () => {
    cornerstoneTools.setToolActive('DeepgrowProbe', { mouseButtonMask: 1 });
    this.addEventListeners(
      'monailabel_deepgrow_probe_event',
      this.deepgrowClickEventHandler
    );
  };
  onLeaveActionTab = () => {
    cornerstoneTools.setToolDisabled('DeepgrowProbe', {});
    this.removeEventListeners();
  };

  addEventListeners = (eventName, handler) => {
    this.removeEventListeners();

    const { element } = this.props.viewConstants;
    element.addEventListener(eventName, handler);
    this.setState({ currentEvent: { name: eventName, handler: handler } });
  };

  removeEventListeners = () => {
    if (!this.state.currentEvent) {
      return;
    }

    const { element } = this.props.viewConstants;
    const { currentEvent } = this.state;

    element.removeEventListener(currentEvent.name, currentEvent.handler);
    this.setState({ currentEvent: null });
  };

  render() {
    let models = [];
    if (this.props.info && this.props.info.models) {
      for (let [name, model] of Object.entries(this.props.info.models)) {
        if (model.type === 'deepgrow' || model.type === 'deepedit') {
          models.push(name);
        }
      }
    }

    return (
      <div className="tab">
        <input
          type="radio"
          name="rd"
          id={this.tabId}
          className="tab-switch"
          value="smartedit"
          onClick={this.onSelectActionTab}
        />
        <label htmlFor={this.tabId} className="tab-label">
          SmartEdit
        </label>
        <div className="tab-content">
          <ModelSelector
            ref={this.modelSelector}
            name="smartedit"
            title="SmartEdit"
            models={models}
            currentModel={this.state.currentModel}
            onClick={this.onDeepgrow}
            onSelectModel={this.onSelectModel}
            usage={
              <div style={{ fontSize: 'smaller' }}>
                <p>
                  Create a label and annotate <b>any organ</b>. &nbsp;Use{' '}
                  <i>Ctrl + Click</i> to add{' '}
                  <b>
                    <i>background</i>
                  </b>{' '}
                  points.
                </p>
                <a href="#" onClick={() => this.clearPoints()}>
                  Clear Points
                </a>
              </div>
            }
          />
        </div>
      </div>
    );
  }
}

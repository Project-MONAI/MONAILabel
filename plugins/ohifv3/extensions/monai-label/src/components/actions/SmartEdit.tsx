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
import ModelSelector from '../ModelSelector';
import BaseTab from './BaseTab';
import * as cornerstoneTools from '@cornerstonejs/tools';
import { vec3 } from 'gl-matrix';
/* import { getFirstSegmentId } from '../../utils/SegmentationUtils'; */

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

  componentDidMount() {
    const { segmentationService, toolGroupService, viewportGridService } =
      this.props.servicesManager.services;

    const added = segmentationService.EVENTS.SEGMENTATION_ADDED;
    const updated = segmentationService.EVENTS.SEGMENTATION_UPDATED;
    const removed = segmentationService.EVENTS.SEGMENTATION_REMOVED;
    const subscriptions = [];

    [added, updated, removed].forEach((evt) => {
      const { unsubscribe } = segmentationService.subscribe(evt, () => {
        const segmentations = segmentationService.getSegmentations();

        if (!segmentations?.length) {
          return;
        }

        // get the first segmentation Todo: fix this to be active
        const segmentation = segmentations[0];
        const { segments, activeSegmentIndex } = segmentation;

        const selectedSegment = segments[activeSegmentIndex];

        const color = selectedSegment.color;

        // get the active viewport toolGroup
        const { viewports, activeViewportId } =
          viewportGridService.getState();
        const viewport = viewports.get(activeViewportId);
        const { viewportOptions } = viewport;
        const toolGroupId = viewportOptions.toolGroupId;

        toolGroupService.setToolConfiguration(toolGroupId, 'ProbeMONAILabel', {
          customColor: `rgb(${color[0]}, ${color[1]}, ${color[2]})`,
        });
      });
      subscriptions.push(unsubscribe);
    });

    this.unsubscribe = () => {
      subscriptions.forEach((unsubscribe) => unsubscribe());
    };
  }

  componentWillUnmount() {
    this.unsubscribe();
  }

  onSelectModel = (model) => {
    this.setState({ currentModel: model });
  };

  onDeepgrow = async () => {
    const { segmentationService, cornerstoneViewportService, viewportGridService } =
      this.props.servicesManager.services;
    const { info, viewConstants } = this.props;
    const image = viewConstants.SeriesInstanceUID;
    const model = this.modelSelector.current.currentModel();

    const activeSegment = segmentationService.getActiveSegment();
    const segmentId = activeSegment.label;

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

    /* const points = this.state.deepgrowPoints.get(segmentId); */

    // Getting the clicks in IJK format

    const { activeViewportId } = viewportGridService.getState();
    const viewPort = cornerstoneViewportService.getCornerstoneViewport(activeViewportId);

    const pts = cornerstoneTools.annotation.state.getAnnotations(
      'ProbeMONAILabel',
      viewPort.element
    );

    const pointsWorld = pts.map((pt) => pt.data.handles.points[0]);
    const { imageData } = viewPort.getImageData();
    const ijk = vec3.fromValues(0, 0, 0);

    // Rounding is not working
    /* const pointsIJK = pointsWorld.map((world) =>
      Math.round(imageData.worldToIndex(world, ijk))
    ); */

    const pointsIJK = pointsWorld.map((world) =>
      imageData.worldToIndex(world, ijk)
    );

    /* const roundPointsIJK = pointsIJK.map(ind => Math.round(ind)) */

    this.state.deepgrowPoints.set(segmentId, pointsIJK);

    // when changing label,  delete previous? or just keep track of all provided clicks per labels
    const points = this.state.deepgrowPoints.get(segmentId);

    // Error as ctrlKey is part of the points?

    /* if (!points.length) {
      return;
    }

    const currentPoint = points[points.length - 1]; */

    const config = this.props.onOptionsConfig();

    const labels = info.models[model].labels;

    const params =
      config && config.infer && config.infer[model] ? config.infer[model] : {};

    // block the cursor while waiting for MONAI Label response?

    for (let l in labels){
      if (l === segmentId) {
        console.log('This is the segmentId')
        let p = []
        for (var i = 0; i < pointsIJK.length; i++) {
          p.push(Array.from(pointsIJK[i]));
          console.log(p[i]);
        }
        params[l] = p;
        continue;
      };
      console.log(l);
      params[l] = [];
    }

    const response = await this.props
      .client()
      .infer(model, image, params);


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
        labels,
        'override',
        is3D ? -1 : currentPoint.z
      );
    }

    // Remove the segmentation and create a new one with a differen index
    /* debugger;
    this.props.servicesManager.services.segmentationService.remove('1') */
  };

  getPointData = (evt) => {
    const { x, y, imageId } = evt.detail;
    const z = this.props.viewConstants.imageIdsToIndex.get(imageId);

    console.debug('X: ' + x + '; Y: ' + y + '; Z: ' + z);
    return { x, y, z, data: evt.detail, imageId };
  };

  onSegmentDeleted = (id) => {
    this.clearPoints(id);
    this.setState({ segmentId: null });
  };
  onSegmentSelected = (id) => {
    this.initPoints(id);
    this.setState({ segmentId: id });
  };

  initPoints = (id) => {
    console.log('Initializing points');
  };

  clearPoints = (id) => {
    cornerstoneTools.annotation.state
      .getAnnotationManager()
      .removeAllAnnotations();
    this.props.servicesManager.services.cornerstoneViewportService
      .getRenderingEngine()
      .render();
    console.log('Clearing all points');
  };

  onSelectActionTab = (evt) => {
    this.props.onSelectActionTab(evt.currentTarget.value);
  };

  onEnterActionTab = () => {
    this.props.commandsManager.runCommand('setToolActive', {
      toolName: 'ProbeMONAILabel',
    });
    console.info('Here we activate the probe');

  };

  onLeaveActionTab = () => {
    this.props.commandsManager.runCommand('setToolDisable', {
      toolName: 'ProbeMONAILabel',
    });
    console.info('Here we deactivate the probe');
    /* cornerstoneTools.setToolDisabled('DeepgrowProbe', {});
    this.removeEventListeners(); */
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
        if (
          model.type === 'deepgrow' ||
          model.type === 'deepedit' ||
          model.type === 'vista'
        ) {
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
              <div style={{ fontSize: 'smaller'}}>
                <p>
                  Create a label and annotate <b>any organ</b>.
                </p>
                <a style={{ backgroundColor:'lightgray'}} onClick={() => this.clearPoints()}>
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

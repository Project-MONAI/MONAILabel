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
import './MonaiLabelPanel.css';
import AutoSegmentation from './actions/AutoSegmentation';
import PointPrompts from './actions/PointPrompts';
import ClassPrompts from './actions/ClassPrompts';
import ActiveLearning from './actions/ActiveLearning';
import MonaiLabelClient from '../services/MonaiLabelClient';
import { hideNotification, getLabelColor } from '../utils/GenericUtils';
import { Enums } from '@cornerstonejs/tools';
import { cache, triggerEvent, eventTarget } from '@cornerstonejs/core';
import SegmentationReader from '../utils/SegmentationReader';
import { currentSegmentsInfo } from '../utils/SegUtils';
import SettingsTable from './SettingsTable';
import * as cornerstoneTools from '@cornerstonejs/tools';
import optionsInputDialog from './OptionsInputDialog';

export default class MonaiLabelPanel extends Component {
  static propTypes = {
    commandsManager: PropTypes.any,
    servicesManager: PropTypes.any,
    extensionManager: PropTypes.any,
  };

  notification: any;
  settings;
  actions: {
    activelearning: any;
    segmentation: any;
    pointprompts: any;
    classprompts: any;
  };
  serverURI = 'http://127.0.0.1:8000';
  private _currentSeriesUID: string | null = null;
  private _unsubscribeFromViewportGrid: any = null;

  constructor(props) {
    super(props);

    const { uiNotificationService } = props.servicesManager.services;
    this.notification = uiNotificationService;
    this.settings = React.createRef();
    this.actions = {
      activelearning: React.createRef(),
      segmentation: React.createRef(),
      pointprompts: React.createRef(),
      classprompts: React.createRef(),
    };

    this.state = {
      info: { models: [], datasets: [] },
      action: {},
      options: {},
    };
  }

  client = () => {
    const settings =
      this.settings && this.settings.current && this.settings.current.state
        ? this.settings.current.state
        : null;
    return new MonaiLabelClient(settings ? settings.url : this.serverURI);
  };

  segmentColor(label) {
    const color = getLabelColor(label);
    const rgbColor = [];
    for (const key in color) {
      rgbColor.push(color[key]);
    }
    rgbColor.push(255);
    return rgbColor;
  }

  getActiveViewportInfo = () => {
    const { viewportGridService, displaySetService } =
      this.props.servicesManager.services;
    const { viewports, activeViewportId } = viewportGridService.getState();
    const viewport = viewports.get(activeViewportId);
    const displaySet = displaySetService.getDisplaySetByUID(
      viewport.displaySetInstanceUIDs[0]
    );

    // viewportId = viewport.viewportId
    // SeriesInstanceUID = displaySet.SeriesInstanceUID;
    // StudyInstanceUID = displaySet.StudyInstanceUID;
    // FrameOfReferenceUID = displaySet.instances[0].FrameOfReferenceUID;
    // displaySetInstanceUID = displaySet.displaySetInstanceUID;
    // numImageFrames = displaySet.numImageFrames;
    return { viewport, displaySet };
  };

  onInfo = async (serverURI) => {
    const nid = this.notification.show({
      title: 'MONAI Label',
      message: 'Connecting to MONAI Label',
      type: 'info',
      duration: 2000,
    });

    this.serverURI = serverURI;
    const response = await this.client().info();
    console.log(response.data);

    hideNotification(nid, this.notification);
    if (response.status !== 200) {
      this.notification.show({
        title: 'MONAI Label',
        message: 'Failed to Connect to MONAI Label',
        type: 'error',
        duration: 5000,
      });
      return;
    }

    this.notification.show({
      title: 'MONAI Label',
      message: 'Connected to MONAI Label - Successful',
      type: 'success',
      duration: 2000,
    });

    const all_models = response.data.models;
    const all_model_names = Object.keys(all_models);
    const deepgrow_models = all_model_names.filter(
      (m) => all_models[m].type === 'deepgrow'
    );
    const deepedit_models = all_model_names.filter(
      (m) => all_models[m].type === 'deepedit'
    );
    const vista3d_models = all_model_names.filter(
      (m) => all_models[m].type === 'vista3d'
    );
    const segmentation_models = all_model_names.filter(
      (m) => all_models[m].type === 'segmentation'
    );
    const models = deepgrow_models
      .concat(deepedit_models)
      .concat(vista3d_models)
      .concat(segmentation_models);
    const all_labels = response.data.labels;

    const modelLabelToIdxMap = {};
    const modelIdxToLabelMap = {};
    const modelLabelNames = {};
    const modelLabelIndices = {};
    for (const model of models) {
      const labels = all_models[model]['labels'];
      modelLabelToIdxMap[model] = {};
      modelIdxToLabelMap[model] = {};
      if (Array.isArray(labels)) {
        for (let label_idx = 1; label_idx <= labels.length; label_idx++) {
          const label = labels[label_idx - 1];
          all_labels.push(label);
          modelLabelToIdxMap[model][label] = label_idx;
          modelIdxToLabelMap[model][label_idx] = label;
        }
      } else {
        for (const label of Object.keys(labels)) {
          const label_idx = labels[label];
          all_labels.push(label);
          modelLabelToIdxMap[model][label] = label_idx;
          modelIdxToLabelMap[model][label_idx] = label;
        }
      }
      modelLabelNames[model] = [
        ...Object.keys(modelLabelToIdxMap[model]),
      ].sort();
      modelLabelIndices[model] = [...Object.keys(modelIdxToLabelMap[model])]
        .sort()
        .map(Number);
    }

    const labelsOrdered = [...new Set(all_labels)].sort();

    // Prepare initial segmentation configuration - will be created per-series on inference
    const initialSegs = labelsOrdered.reduce((acc, label, index) => {
            acc[index + 1] = {
              segmentIndex: index + 1,
              label: label,
        active: index === 0,
              locked: false,
              color: this.segmentColor(label),
            };
            return acc;
    }, {});

    const info = {
      models: models,
      labels: labelsOrdered,
      data: response.data,
      modelLabelToIdxMap: modelLabelToIdxMap,
      modelIdxToLabelMap: modelIdxToLabelMap,
      modelLabelNames: modelLabelNames,
      modelLabelIndices: modelLabelIndices,
      initialSegs: initialSegs,
    };

    console.log(info);
    this.setState({ info: info });
    this.setState({ isDataReady: true }); // Mark as ready
    this.setState({ options: {} });
  };

  onSelectActionTab = (name) => {
    for (const action of Object.keys(this.actions)) {
      if (this.state.action === action) {
        if (this.actions[action].current) {
          this.actions[action].current.onLeaveActionTab();
        }
      }
    }

    for (const action of Object.keys(this.actions)) {
      if (name === action) {
        if (this.actions[action].current) {
          this.actions[action].current.onEnterActionTab();
        }
      }
    }
    this.setState({ action: name });

    // Check if we switched series and need to reapply origin correction
    this.checkAndApplyOriginCorrectionOnSeriesSwitch();
  };

  // Check if series has changed and apply origin correction to existing segmentation
  checkAndApplyOriginCorrectionOnSeriesSwitch = () => {
    try {
      const currentViewportInfo = this.getActiveViewportInfo();
      const currentSeriesUID = currentViewportInfo?.displaySet?.SeriesInstanceUID;

      // If series changed
      if (currentSeriesUID && currentSeriesUID !== this._currentSeriesUID) {
        this._currentSeriesUID = currentSeriesUID;
        const segmentationId = `seg-${currentSeriesUID}`;

        // Check if this series already has a segmentation
        const { segmentationService } = this.props.servicesManager.services;
        try {
          const volumeLoadObject = segmentationService.getLabelmapVolume(segmentationId);
          if (volumeLoadObject) {
            // Segmentation exists, apply origin correction
            this.applyOriginCorrection(volumeLoadObject);
          }
      } catch (e) {
          // No segmentation for this series yet, which is fine
        }
      }
    } catch (e) {
      // Ignore errors (e.g., viewport not ready)
    }
  };

  // Apply origin correction - match segmentation origin to image volume origin
  applyOriginCorrection = (volumeLoadObject) => {
    const { displaySet } = this.getActiveViewportInfo();
    const imageVolumeId = displaySet.displaySetInstanceUID;
    let imageVolume = cache.getVolume(imageVolumeId);
    if (!imageVolume) {
      imageVolume = cache.getVolume('cornerstoneStreamingImageVolume:' + imageVolumeId);
    }

    if (imageVolume && displaySet.isMultiFrame) {
      // Simply copy the image volume's origin to the segmentation
      // This way the segmentation matches whatever origin OHIF has set for the image
      volumeLoadObject.origin = [...imageVolume.origin];

      if (volumeLoadObject.imageData) {
        volumeLoadObject.imageData.setOrigin(volumeLoadObject.origin);
      }

      // Trigger render to show the corrected segmentation
      const renderingEngine = this.props.servicesManager.services.cornerstoneViewportService.getRenderingEngine();
      if (renderingEngine) {
        renderingEngine.render();
      }
    }
  };

  updateView = async (
    response,
    model_id,
    labels,
    override = false,
    label_class_unknown = false,
    sidx = -1
  ) => {
    console.log('UpdateView: ', {
      model_id,
      labels,
      override,
      label_class_unknown,
      sidx,
    });
    const ret = SegmentationReader.parseNrrdData(response.data);
    if (!ret) {
      throw new Error('Failed to parse NRRD data');
    }

    const labelNames = {};
    const currentSegs = currentSegmentsInfo(
      this.props.servicesManager.services.segmentationService
    );
    const modelToSegMapping = {};
    modelToSegMapping[0] = 0;

    let tmp_model_seg_idx = 1;
    for (const label of labels) {
      const s = currentSegs.info[label];
      if (!s) {
        for (let i = 1; i <= 255; i++) {
          if (!currentSegs.indices.has(i)) {
            labelNames[label] = i;
            currentSegs.indices.add(i);
            break;
          }
        }
      } else {
        labelNames[label] = s.segmentIndex;
      }

      const seg_idx = labelNames[label];
      let model_seg_idx = this.state.info.modelLabelToIdxMap[model_id][label];
      model_seg_idx = model_seg_idx ? model_seg_idx : tmp_model_seg_idx;
      modelToSegMapping[model_seg_idx] = 0xff & seg_idx;
      tmp_model_seg_idx++;
    }

    console.log('Index Remap', labels, modelToSegMapping);
    const data = new Uint8Array(ret.image);

    // Use series-specific segmentation ID to ensure each series has its own segmentation
    const currentViewportInfo = this.getActiveViewportInfo();
    const currentSeriesUID = currentViewportInfo?.displaySet?.SeriesInstanceUID;
    const segmentationId = `seg-${currentSeriesUID || 'default'}`;

    // Track current series
    this._currentSeriesUID = currentSeriesUID;

    const { segmentationService } = this.props.servicesManager.services;
    let volumeLoadObject = null;

    try {
      volumeLoadObject = segmentationService.getLabelmapVolume(segmentationId);
    } catch (e) {
      // Segmentation doesn't exist yet - create it
      const initialSegs = this.state.info?.initialSegs;
      if (initialSegs) {
        const segmentations = [{
          segmentationId: segmentationId,
          representation: {
            type: Enums.SegmentationRepresentations.Labelmap
          },
          config: {
            label: 'Segmentations',
            segments: initialSegs
          }
        }];

        this.props.commandsManager.runCommand('loadSegmentationsForViewport', {
          segmentations
        });

        // Wait a bit for segmentation to be created, then try again
        setTimeout(() => {
          try {
            const vol = segmentationService.getLabelmapVolume(segmentationId);
            if (vol) {
              this.updateView(response, model_id, labels, override, label_class_unknown, sidx);
            }
          } catch (err) {
            console.error('Failed to create segmentation volume:', err);
          }
        }, 500);
        return;
      }
    }

    if (volumeLoadObject) {
      let convertedData = data;

      // Convert label indices
      for (let i = 0; i < convertedData.length; i++) {
        const midx = convertedData[i];
          const sidx_mapped = modelToSegMapping[midx];
          if (midx && sidx_mapped) {
            convertedData[i] = sidx_mapped;
        } else if (override && label_class_unknown && labels.length === 1) {
          convertedData[i] = midx ? labelNames[labels[0]] : 0;
        } else if (labels.length > 0) {
          convertedData[i] = 0;
        }
      }

      // Handle override mode (partial update)
      if (override === true) {
        const { voxelManager } = volumeLoadObject;
        const scalarData = voxelManager?.getCompleteScalarDataArray();
        const currentSegArray = new Uint8Array(scalarData.length);
        currentSegArray.set(scalarData);

        const updateTargets = new Set(convertedData);
        const numImageFrames = this.getActiveViewportInfo().displaySet.numImageFrames;
        const sliceLength = scalarData.length / numImageFrames;
        const sliceBegin = sliceLength * sidx;
        const sliceEnd = sliceBegin + sliceLength;

        for (let i = 0; i < convertedData.length; i++) {
          if (sidx >= 0 && (i < sliceBegin || i >= sliceEnd)) {
            continue;
          }
          if (convertedData[i] !== 255 && updateTargets.has(currentSegArray[i])) {
            currentSegArray[i] = convertedData[i];
          }
        }
        convertedData = currentSegArray;
      }

      // Apply origin correction for multi-frame volumes
      this.applyOriginCorrection(volumeLoadObject);

      // Apply segment colors
      const { viewport } = this.getActiveViewportInfo();
      for (const label of labels) {
        const segmentIndex = labelNames[label];
        if (segmentIndex) {
          const color = this.segmentColor(label);
          cornerstoneTools.segmentation.config.color.setSegmentIndexColor(
            viewport.viewportId,
        segmentationId,
            segmentIndex,
            color
          );
        }
      }

      // Set the voxel data
      volumeLoadObject.voxelManager.setCompleteScalarDataArray(convertedData);
      triggerEvent(eventTarget, Enums.Events.SEGMENTATION_DATA_MODIFIED, {
        segmentationId: segmentationId
      });
    }
  };

  openConfigurations = (e) => {
    e.preventDefault();

    const { uiDialogService } = this.props.servicesManager.services;
    optionsInputDialog(
      uiDialogService,
      this.state.options,
      this.state.info,
      (options, actionId) => {
        if (actionId === 'save' || actionId == 'reset') {
          this.setState({ options: options });
        }
      }
    );
  };

  async componentDidMount() {
    if (this.state.isDataReady) {
      return;
    }

    console.log('(Component Mounted) Ready to Connect to MONAI Server...');

    // Subscribe to viewport grid state changes to detect series switches
    const { viewportGridService } = this.props.servicesManager.services;

    // Listen to any state change in the viewport grid
    const handleViewportChange = () => {
      // Multiple attempts with delays to catch the viewport at the right time
      setTimeout(() => this.checkAndApplyOriginCorrectionOnSeriesSwitch(), 50);
      setTimeout(() => this.checkAndApplyOriginCorrectionOnSeriesSwitch(), 200);
      setTimeout(() => this.checkAndApplyOriginCorrectionOnSeriesSwitch(), 500);
    };

    this._unsubscribeFromViewportGrid = viewportGridService.subscribe(
      viewportGridService.EVENTS.ACTIVE_VIEWPORT_ID_CHANGED,
      handleViewportChange
    );

    // await this.onInfo();
  }

  componentWillUnmount() {
    if (this._unsubscribeFromViewportGrid) {
      this._unsubscribeFromViewportGrid();
      this._unsubscribeFromViewportGrid = null;
    }
  }

  onOptionsConfig = () => {
    return this.state.options;
  };

  render() {
    const { isDataReady } = this.state;
    return (
      <div className="monaiLabelPanel">
        <br style={{ margin: '3px' }} />

        <SettingsTable ref={this.settings} onInfo={this.onInfo} />
        {isDataReady && (
          <div style={{ color: 'white' }}>
            <p className="subtitle">{this.state.info.data.name}</p>
            <br />
            <hr className="separator" />
            <a href="#" onClick={this.openConfigurations}>
              Options / Configurations
            </a>
            <hr className="separator" />
          </div>
        )}
        {isDataReady && (
          <div className="tabs scrollbar" id="style-3">
            <ActiveLearning
              ref={this.actions['activelearning']}
              tabIndex={1}
              info={this.state.info}
              client={this.client}
              updateView={this.updateView}
              onSelectActionTab={this.onSelectActionTab}
              onOptionsConfig={this.onOptionsConfig}
              getActiveViewportInfo={this.getActiveViewportInfo}
            />
            <AutoSegmentation
              ref={this.actions['segmentation']}
              tabIndex={2}
              info={this.state.info}
              client={this.client}
              updateView={this.updateView}
              onSelectActionTab={this.onSelectActionTab}
              onOptionsConfig={this.onOptionsConfig}
              getActiveViewportInfo={this.getActiveViewportInfo}
            />
            <PointPrompts
              ref={this.actions['pointprompts']}
              tabIndex={3}
              info={this.state.info}
              client={this.client}
              updateView={this.updateView}
              onSelectActionTab={this.onSelectActionTab}
              onOptionsConfig={this.onOptionsConfig}
              getActiveViewportInfo={this.getActiveViewportInfo}
              servicesManager={this.props.servicesManager}
              commandsManager={this.props.commandsManager}
            />
            <ClassPrompts
              ref={this.actions['classprompts']}
              tabIndex={4}
              info={this.state.info}
              client={this.client}
              updateView={this.updateView}
              onSelectActionTab={this.onSelectActionTab}
              onOptionsConfig={this.onOptionsConfig}
              getActiveViewportInfo={this.getActiveViewportInfo}
              servicesManager={this.props.servicesManager}
              commandsManager={this.props.commandsManager}
            />
          </div>
        )}
      </div>
    );
  }
}

import React, { Component } from 'react';
import PropTypes from 'prop-types';
import './MonaiLabelPanel.css';
import ActiveLearning from './actions/ActiveLearning';
import AutoSegmentation from './actions/AutoSegmentation';
import PointPrompts from './actions/PointPrompts';
import ClassPrompts from './actions/ClassPrompts';
import MonaiLabelClient from '../services/MonaiLabelClient';
import { hideNotification, getLabelColor } from '../utils/GenericUtils';
import { Enums } from '@cornerstonejs/tools';
import { cache, triggerEvent, eventTarget } from '@cornerstonejs/core';
import SegmentationReader from '../utils/SegmentationReader';
import { currentSegmentsInfo } from '../utils/SegUtils';
import SettingsTable from './SettingsTable';

export default class MonaiLabelPanel extends Component {
  static propTypes = {
    commandsManager: PropTypes.any,
    servicesManager: PropTypes.any,
    extensionManager: PropTypes.any,
  };

  notification: any;
  settings: any;
  state: { info: {}; action: {} };
  actions: {
    activelearning: any;
    segmentation: any;
    pointprompts: any;
    classprompts: any;
  };
  props: any;
  SeriesInstanceUID: any;
  StudyInstanceUID: any;
  FrameOfReferenceUID: any;
  displaySetInstanceUID: any;

  constructor(props) {
    super(props);

    const { uiNotificationService, displaySetService } =
      props.servicesManager.services;

    this.SeriesInstanceUID =
      displaySetService.activeDisplaySets[0].SeriesInstanceUID;
    this.StudyInstanceUID =
      displaySetService.activeDisplaySets[0].StudyInstanceUID;
    this.notification = uiNotificationService;
    this.actions = {
      activelearning: React.createRef(),
      segmentation: React.createRef(),
      pointprompts: React.createRef(),
      classprompts: React.createRef(),
    };

    this.state = {
      info: { models: [], datasets: [] },
      action: {},
    };

    // Todo: fix this hack
    setTimeout(() => {
      const displaySet = displaySetService.activeDisplaySets[0];
      this.SeriesInstanceUID = displaySet.SeriesInstanceUID;
      this.StudyInstanceUID = displaySet.StudyInstanceUID;
      this.FrameOfReferenceUID = displaySet.instances[0].FrameOfReferenceUID;
      this.displaySetInstanceUID = displaySet.displaySetInstanceUID;
    }, 1000);
  }

  client = () => {
    const settings =
      this.settings && this.settings.current && this.settings.current.state
        ? this.settings.current.state
        : null;
    return new MonaiLabelClient(
      settings ? settings.url : 'http://127.0.0.1:8000'
    );
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

  onInfo = async () => {
    const nid = this.notification.show({
      title: 'MONAI Label',
      message: 'Connecting to MONAI Label',
      type: 'info',
      duration: 2000,
    });

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
      for (const label of Object.keys(labels)) {
        const label_idx = labels[label];
        all_labels.push(label);
        modelLabelToIdxMap[model][label] = label_idx;
        modelIdxToLabelMap[model][label_idx] = label;
      }
      modelLabelNames[model] = [
        ...Object.keys(modelLabelToIdxMap[model]),
      ].sort();
      modelLabelIndices[model] = [...Object.keys(modelIdxToLabelMap[model])]
        .sort()
        .map(Number);
    }

    const labelsOrdered = [...new Set(all_labels)].sort();
    const segmentations = [
      {
        id: '1',
        label: 'Segmentations',
        segments: labelsOrdered.map((label, index) => ({
          segmentIndex: index + 1,
          label: label,
          color: this.segmentColor(label),
        })),
        isActive: true,
        activeSegmentIndex: 1,
      },
    ];
    const initialSegs = segmentations[0].segments;
    const volumeLoadObject = cache.getVolume('1');
    if (!volumeLoadObject) {
      this.props.commandsManager.runCommand('loadSegmentationsForViewport', {
        segmentations,
      });
    }

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
  };

  updateView = async (response, model_id, labels, override = false) => {
    console.log('Update View: ', model_id, labels, override);
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

    // Todo: rename volumeId
    const volumeLoadObject = cache.getVolume('1');
    if (volumeLoadObject) {
      console.log('Volume Object is In Cache....');
      const { scalarData } = volumeLoadObject;
      // console.log('scalarData', scalarData);

      // Model Idx to Segment Idx conversion (merge for multiple models with different label idx for the same name)
      const convertedData = data;
      for (let i = 0; i < convertedData.length; i++) {
        const midx = convertedData[i];
        const sidx = modelToSegMapping[midx];
        if (midx && sidx) {
          convertedData[i] = sidx;
        } else if (labels.length > 0) {
          convertedData[i] = 0; // Ignore unknown label idx
        }
      }

      if (override === true) {
        const scalarDataRecover = new Uint8Array(
          window.ScalarDataBuffer.length
        );
        scalarDataRecover.set(window.ScalarDataBuffer);

        // get unique values to determine which organs to update, keep rest
        const updateTargets = new Set(convertedData);
        for (let i = 0; i < convertedData.length; i++) {
          if (
            convertedData[i] !== 255 &&
            updateTargets.has(scalarDataRecover[i])
          ) {
            scalarDataRecover[i] = convertedData[i];
          }
        }
        scalarData.set(scalarDataRecover);
      } else {
        scalarData.set(convertedData);
      }

      triggerEvent(eventTarget, Enums.Events.SEGMENTATION_DATA_MODIFIED, {
        segmentationId: '1',
      });
      console.debug("updated the segmentation's scalar data");
    } else {
      // TODO:: Remap Index here as well...
      console.log('Volume Object is NOT In Cache....');
      const segmentations = [
        {
          id: '1',
          label: 'Segmentations',
          segments: Object.entries(labelNames).map(([k, v]) => ({
            segmentIndex: v,
            label: k,
            color: this.segmentColor(k),
          })),
          isActive: true,
          activeSegmentIndex: 1,
          scalarData: data,
          FrameOfReferenceUID: this.FrameOfReferenceUID,
        },
      ];

      this.props.commandsManager.runCommand('loadSegmentationsForDisplaySet', {
        displaySetInstanceUID: this.displaySetInstanceUID,
        segmentations,
      });
    }
  };

  async componentDidMount() {
    if (this.state.isDataReady) {
      return;
    }

    console.log('(Component Mounted) Connect to MONAI Server...');
    await this.onInfo();
  }

  render() {
    const { isDataReady, isInteractiveSeg } = this.state;
    return (
      <div className="monaiLabelPanel">
        <br style={{ margin: '3px' }} />

        <SettingsTable ref={this.settings} onInfo={this.onInfo} />
        <hr className="separator" />
        <p className="subtitle">{this.state.info.name}</p>

        {isDataReady && (
          <div className="tabs scrollbar" id="style-3">
            <AutoSegmentation
              ref={this.actions['segmentation']}
              tabIndex={2}
              info={this.state.info}
              viewConstants={{
                SeriesInstanceUID: this.SeriesInstanceUID,
                StudyInstanceUID: this.StudyInstanceUID,
              }}
              client={this.client}
              notification={this.notification}
              updateView={this.updateView}
              onSelectActionTab={this.onSelectActionTab}
            />
            <PointPrompts
              ref={this.actions['pointprompts']}
              tabIndex={3}
              servicesManager={this.props.servicesManager}
              commandsManager={this.props.commandsManager}
              info={this.state.info}
              viewConstants={{
                SeriesInstanceUID: this.SeriesInstanceUID,
                StudyInstanceUID: this.StudyInstanceUID,
              }}
              client={this.client}
              notification={this.notification}
              updateView={this.updateView}
              onSelectActionTab={this.onSelectActionTab}
            />
            <ClassPrompts
              ref={this.actions['classprompts']}
              tabIndex={4}
              servicesManager={this.props.servicesManager}
              commandsManager={this.props.commandsManager}
              info={this.state.info}
              viewConstants={{
                SeriesInstanceUID: this.SeriesInstanceUID,
                StudyInstanceUID: this.StudyInstanceUID,
              }}
              client={this.client}
              notification={this.notification}
              updateView={this.updateView}
              onSelectActionTab={this.onSelectActionTab}
            />
          </div>
        )}
      </div>
    );
  }
}

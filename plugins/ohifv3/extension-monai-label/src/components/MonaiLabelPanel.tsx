import React, { Component } from 'react';
import PropTypes from 'prop-types';
import './MonaiLabelPanel.styl';
import SettingsTable from './SettingsTable';
import AutoSegmentation from './actions/AutoSegmentation';
import OptionTable from './actions/OptionTable';
import MonaiLabelClient from '../services/MonaiLabelClient';
import SegmentationReader from '../../../../ohif/monai-label/src/utils/SegmentationReader';
import MonaiSegmentation from './MonaiSegmentation';

export default class MonaiLabelPanel extends Component {
  static propTypes = {
    commandsManager: PropTypes.any,
    servicesManager: PropTypes.any,
    extensionManager: PropTypes.any,
  };

  notification: any;
  settings: any;
  state: { info: {}; action: {}; segmentations: [] };
  actions: { options: any; activeLearning: any; segmentation: any };
  props: any;
  SeriesInstanceUID: any;
  StudyInstanceUID: any;

  constructor(props) {
    super(props);

    const { uiNotificationService, viewportGridService, displaySetService } =
      props.servicesManager.services;

    // just for debugging
    setTimeout(() => {
      const { viewports, activeViewportIndex } = viewportGridService.getState();
      const viewport = viewports[activeViewportIndex];
      const displaySet = displaySetService.getDisplaySetByUID(
        viewport.displaySetInstanceUIDs[0]
      );

      this.SeriesInstanceUID = displaySet.SeriesInstanceUID;
      this.StudyInstanceUID = displaySet.StudyInstanceUID;
      this.FrameOfReferenceUID = displaySet.instances[0].FrameOfReferenceUID;
      this.displaySetInstanceUID = displaySet.displaySetInstanceUID;

      this.notification = uiNotificationService;
      this.settings = React.createRef();
    }, 2000);

    this.actions = {
      options: React.createRef(),
      activeLearning: React.createRef(),
      segmentation: React.createRef(),
    };

    this.state = {
      info: {},
      action: {},
      segmentations: [],
    };
  }

  async componentDidMount() {
    const { segmentationService } = this.props.servicesManager.services;
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

        this.setState({ segmentations });
      });
      subscriptions.push(unsubscribe);
    });

    this.unsubscribe = () => {
      subscriptions.forEach((unsubscribe) => unsubscribe());
    };
  }

  async componentDidUnmount() {
    this.unsubscribe();
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

  onInfo = async () => {
    this.notification.show({
      title: 'MONAI Label',
      message: 'Connecting to MONAI Label',
      type: 'info',
      duration: 3000,
    });

    const response = await this.client().info();

    if (response.status !== 200) {
      this.notification.show({
        title: 'MONAI Label',
        message: 'Failed to Connect to MONAI Label Server',
        type: 'error',
        duration: 5000,
      });
    } else {
      this.notification.show({
        title: 'MONAI Label',
        message: 'Connected to MONAI Label Server - Successful',
        type: 'success',
        duration: 2000,
      });

      this.setState({ info: response.data });
    }
  };

  onSelectActionTab = (name) => {
    // Leave Event
    for (const action of Object.keys(this.actions)) {
      if (this.state.action === action) {
        if (this.actions[action].current)
          this.actions[action].current.onLeaveActionTab();
      }
    }

    // Enter Event
    for (const action of Object.keys(this.actions)) {
      if (name === action) {
        if (this.actions[action].current)
          this.actions[action].current.onEnterActionTab();
      }
    }
    this.setState({ action: name });
  };

  onOptionsConfig = () => {
    return this.actions['options'].current &&
      this.actions['options'].current.state
      ? this.actions['options'].current.state.config
      : {};
  };

  updateView = async (response, labels) => {
    // Process the obtained binary file from the MONAI Label server
    console.info('These are the predicted labels');
    console.info(labels);

    // for debugging only, we should get the response from the server
  };

  handleSegLoad = async () => {
    const response = await fetch('http://localhost:3000/pred.nrrd');
    if (!response.ok) {
      throw new Error('Network response was not ok');
    }

    const info = {
      spleen: 1,
      kidney_right: 2,
      kidney_left: 3,
      gallbladder: 4,
      liver: 5,
      stomach: 6,
      aorta: 7,
      inferior_vena_cava: 8,
      portal_vein_and_splenic_vein: 9,
      pancreas: 10,
      adrenal_gland_right: 11,
      adrenal_gland_left: 12,
      lung_upper_lobe_left: 13,
      lung_lower_lobe_left: 14,
      lung_upper_lobe_right: 15,
      lung_middle_lobe_right: 16,
      lung_lower_lobe_right: 17,
      esophagus: 42,
      trachea: 43,
      heart_myocardium: 44,
      heart_atrium_left: 45,
      heart_ventricle_left: 46,
      heart_atrium_right: 47,
      heart_ventricle_right: 48,
      pulmonary_artery: 49,
    };

    const nrrd = await response.arrayBuffer();

    const ret = SegmentationReader.parseNrrdData(nrrd);

    if (!ret) {
      throw new Error('Failed to parse NRRD data');
    }

    const { image: buffer, header } = ret;
    const data = new Uint16Array(buffer);

    const segmentations = [
      {
        id: '1',
        label: 'Segmentation 1',
        segments: Object.keys(info).map((key) => ({
          segmentIndex: info[key],
          label: key,
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
  };

  render() {
    return (
      <>
        <div
          className="monaiLabelPanel"
          style={{ backgroundColor: 'black', color: '#90CDF4' }}
        >
          <br style={{ margin: '3px' }} />

          <SettingsTable ref={this.settings} onInfo={this.onInfo} />

          <hr className="separator" />
          <button onClick={this.handleSegLoad}>Load SEG</button>

          <p className="subtitle">{this.state.info.name}</p>

          <div className="tabs scrollbar" id="style-3">
            <OptionTable
              ref={this.actions['options']}
              tabIndex={1}
              info={this.state.info}
              viewConstants={{
                SeriesInstanceUID: this.SeriesInstanceUID,
                StudyInstanceUID: this.StudyInstanceUID,
              }}
              client={this.client}
              notification={this.notification}
              //updateView={this.updateView}
              onSelectActionTab={this.onSelectActionTab}
            />

            <AutoSegmentation
              ref={this.actions['segmentation']}
              tabIndex={3}
              info={this.state.info}
              viewConstants={{
                SeriesInstanceUID: this.SeriesInstanceUID,
                StudyInstanceUID: this.StudyInstanceUID,
              }}
              client={this.client}
              notification={this.notification}
              updateView={this.updateView}
              onSelectActionTab={this.onSelectActionTab}
              onOptionsConfig={this.onOptionsConfig}
            />
          </div>

          <p>&nbsp;</p>
          {this.state.segmentations?.map((segmentation) => (
            <MonaiSegmentation
              segmentation={segmentation}
              servicesManager={this.props.servicesManager}
            />
          ))}
        </div>
      </>
    );
  }
}

import React, { Component } from 'react';
import PropTypes from 'prop-types';
import './MonaiLabelPanel.styl';
import SettingsTable from './SettingsTable';
import AutoSegmentation from './actions/AutoSegmentation';
import SmartEdit from './actions/SmartEdit';
import OptionTable from './actions/OptionTable'; 
import ActiveLearning from './actions/ActiveLearning';
import MonaiLabelClient from '../services/MonaiLabelClient';
import SegmentationReader from '../utils/SegmentationReader';
import MonaiSegmentation from './MonaiSegmentation';

export default class MonaiLabelPanel extends Component {
  static propTypes = {
    commandsManager: PropTypes.any,
    servicesManager: PropTypes.any,
    extensionManager: PropTypes.any,
  };

  notification: any;
  settings: any;
  state: { info: {}; action: {}; };
  actions: { options: any; 
    activelearning: any; 
    segmentation: any;
    smartedit: any; 
  };
  props: any;    
  SeriesInstanceUID: any;
  StudyInstanceUID: any;

  constructor(props) {
    super(props);

    const { uiNotificationService, viewportGridService, displaySetService } =
      props.servicesManager.services;
    
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
    

    this.actions = {
      options: React.createRef(),
      activeLearning: React.createRef(),
      segmentation: React.createRef(),
      smartedit: React.createRef(),
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

  // componentDidUnmount? Doesn't exist this method anymore in V3?
  async componentWillUnmount() {
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

  updateView = async (response, labelNames) => {
    // Process the obtained binary file from the MONAI Label server

    /* const onInfoLabelNames = this.state.info.labels */
    const onInfoLabelNames = labelNames

    console.info('These are the predicted labels');
    console.info(onInfoLabelNames);

    if (onInfoLabelNames.hasOwnProperty("background"))
      delete onInfoLabelNames.background;

    const ret = SegmentationReader.parseNrrdData(response.data);

    if (!ret) {
      throw new Error('Failed to parse NRRD data');
    }

    const { image: buffer, header } = ret;
    const data = new Uint16Array(buffer);

    const segmentations = [
      {
        id: '1',
        label: 'Segmentations',
        segments: Object.keys(onInfoLabelNames).map((key) => ({
          segmentIndex: onInfoLabelNames[key],
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
      <div className="monaiLabelPanel">

        <br style={{ margin: '3px' }} />

        <SettingsTable 
          ref={this.settings} 
          onInfo={this.onInfo}
          />

        <hr className="separator" />

        <p className="subtitle">{this.state.info.name}</p>

        <div className="tabs scrollbar" id="style-3">
          <OptionTable
            ref={this.actions['options']}
            tabIndex={1}
            info={this.state.info}
            viewConstants={{'SeriesInstanceUID': this.SeriesInstanceUID, 
                            'StudyInstanceUID': this.StudyInstanceUID
                          }}
            client={this.client}
            notification={this.notification}
            //updateView={this.updateView}
            onSelectActionTab={this.onSelectActionTab}
          />

          <ActiveLearning
            ref={this.actions['activelearning']}
            tabIndex={2}
            info={this.state.info}
            viewConstants={{'SeriesInstanceUID': this.SeriesInstanceUID, 
                          'StudyInstanceUID': this.StudyInstanceUID
                          }}
            client={this.client}
            notification={this.notification}
            /* updateView={this.updateView} */
            onSelectActionTab={this.onSelectActionTab}
            onOptionsConfig={this.onOptionsConfig}
            // additional function - delete scribbles before submit
            /* onDeleteSegmentByName={this.onDeleteSegmentByName} */
          />

          <AutoSegmentation
            ref={this.actions['segmentation']}
            tabIndex={3}
            info={this.state.info}
            viewConstants={{'SeriesInstanceUID': this.SeriesInstanceUID, 
                            'StudyInstanceUID': this.StudyInstanceUID
                            }}
            client={this.client}
            notification={this.notification}
            updateView={this.updateView}
            onSelectActionTab={this.onSelectActionTab}
            onOptionsConfig={this.onOptionsConfig}
          />
          <SmartEdit
            ref={this.actions['smartedit']}
            tabIndex={4}
            servicesManager = {this.props.servicesManager}
            commandsManager = {this.props.commandsManager}
            info={this.state.info}
            // Here we have to send element - In OHIF V2 - const element = cornerstone.getEnabledElements()[this.props.activeIndex].element;
            viewConstants={{'SeriesInstanceUID': this.SeriesInstanceUID, 
                          'StudyInstanceUID': this.StudyInstanceUID
            }}
            client={this.client}
            notification={this.notification}
            updateView={this.updateView}
            onSelectActionTab={this.onSelectActionTab}
            onOptionsConfig={this.onOptionsConfig}
          />
        </div>
          {this.state.segmentations?.map((segmentation) => (
            <MonaiSegmentation
              segmentation={segmentation}
              servicesManager={this.props.servicesManager}
            />
          ))}
      </div>
    );
  }
}

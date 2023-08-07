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
import { UINotificationService } from '@ohif/core';
import { getImageIdsForDisplaySet } from '../utils/SegmentationUtils';
import PropTypes from 'prop-types'; 
/* import MD5 from 'md5.js'; */
import './MonaiLabelPanel.styl'; 
import SettingsTable from './SettingsTable';
import AutoSegmentation from './actions/AutoSegmentation';
import OptionTable from './actions/OptionTable'; 
import MonaiLabelClient from '../services/MonaiLabelClient';


// Export as a class - React class components

export default class MonaiLabelPanel extends Component {

  static propTypes = {
    studies: PropTypes.any,
    viewports: PropTypes.any,
    activeIndex: PropTypes.any,
  };
  
  notification: any;
  settings: any;
  state: { info: {}; action: {}; };
  actions: { options: any; activelearning: any; segmentation: any; smartedit: any; scribbles: any; };
  props: any;
  
  /* viewConstants: { 
    PatientID: any; 
    StudyInstanceUID: any; 
    SeriesInstanceUID: any; 
    displaySetInstanceUID: any; 
    imageIdsToIndex: Map<any, any>; 
    numberOfFrames: any; 
    cookiePostfix: any; }; */
  

  constructor(props) {
    super(props);

    const { viewports, studies, activeIndex } = props;

    /* this.viewConstants = this.getViewConstants(viewports, studies, activeIndex);
    console.debug(this.viewConstants); */

    this.notification = new UINotificationService();

    this.settings = React.createRef();

    this.actions = {
      options: React.createRef(),
      activelearning: React.createRef(),
      segmentation: React.createRef(),
      smartedit: React.createRef(),
      scribbles: React.createRef(),
    };

    this.state = {
      info: {},
      action: {},
    };

  };

  async componentDidMount() {
    await this.onInfo();
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

  getViewConstants = (viewports, studies, activeIndex) => {
    const viewport = viewports[activeIndex];
    const { PatientID } = studies[activeIndex];

    const {
      StudyInstanceUID,
      SeriesInstanceUID,
      displaySetInstanceUID,
    } = viewport;

    const imageIds = getImageIdsForDisplaySet(
      studies,
      StudyInstanceUID,
      SeriesInstanceUID
    );
    const imageIdsToIndex = new Map();

    for (let i = 0; i < imageIds.length; i++) {
      imageIdsToIndex.set(imageIds[i], i);
    }

    /* const cookiePostfix = new MD5()
      .update(PatientID + StudyInstanceUID + SeriesInstanceUID)
      .digest('hex'); */

    return {
      PatientID: PatientID,
      StudyInstanceUID: StudyInstanceUID,
      SeriesInstanceUID: SeriesInstanceUID,
      displaySetInstanceUID: displaySetInstanceUID,
      imageIdsToIndex: imageIdsToIndex,
      numberOfFrames: imageIds.length,
      /* cookiePostfix: cookiePostfix, */
    };
  };

  onInfo = async () => {
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

      this.state['info'] = response.data;
    }
  };

  onSelectActionTab = name => {
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
    this.state['action'] = name;
  };

  onOptionsConfig = () => {
    return this.actions['options'].current &&
      this.actions['options'].current.state
      ? this.actions['options'].current.state.config
      : {};
  };
  
  render() {
    return (
      <div className="monaiLabelPanel">

        <br style={{ margin: '3px' }} />

        <SettingsTable ref={this.settings} />

        <hr className="separator" />

        <p className="subtitle">{this.state.info.name}</p>

        <div className="tabs scrollbar" id="style-3">
          <OptionTable
            ref={this.actions['options']}
            tabIndex={1}
            info={this.state.info}
            //viewConstants={this.viewConstants}
            client={this.client}
            notification={this.notification}
            //updateView={this.updateView}
            onSelectActionTab={this.onSelectActionTab}
          />

          <AutoSegmentation
            ref={this.actions['segmentation']}
            tabIndex={3}
            info={this.state.info}
            //viewConstants={this.viewConstants}
            client={this.client}
            notification={this.notification}
            //updateView={this.updateView}
            onSelectActionTab={this.onSelectActionTab}
            onOptionsConfig={this.onOptionsConfig}
          />

        </div>

        <p>&nbsp;</p>
      </div>
  );
  }
};  

// Toy examples

/*   render() {
    return (
    <React.Fragment>
    <ButtonGroup color="black" size="inherit">
      <LegacyButton className="px-2 py-2 text-base">
        {'MONAI Label Tab'}
      </LegacyButton>
    </ButtonGroup>
  </React.Fragment>
  );
  }
 */

/*   render() {
    return (
    <React.Fragment>
    <SettingsTable ref={this.settings} />
    <ButtonGroup color="black" size="inherit">
      <LegacyButton className="px-2 py-2 text-base">
        {'MONAI Label Tab'}
      </LegacyButton>
    </ButtonGroup>
  </React.Fragment>
  );
  } */


// Export as a function

/* export default function MonaiLabelPanel(): React.Component {
  
  return (
    <React.Fragment>
    <ButtonGroup color="black" size="inherit">
      <LegacyButton className="px-2 py-2 text-base">
        {'Hello MONAI Label'}
      </LegacyButton>
    </ButtonGroup>
  </React.Fragment>
  );
};  */


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

import { Component } from 'react';
import PropTypes from 'prop-types';

import './BaseTab.styl';
import { UIModalService, UINotificationService } from '@ohif/core';

export default class BaseTab extends Component {
  static propTypes = {
    tabIndex: PropTypes.number,
    info: PropTypes.any,
    segmentId: PropTypes.string,
    viewConstants: PropTypes.any,
    client: PropTypes.func,
    updateView: PropTypes.func,
    onSelectActionTab: PropTypes.func,
    onOptionsConfig: PropTypes.func,
  };

  constructor(props) {
    super(props);
    this.notification = UINotificationService.create({});
    this.uiModelService = UIModalService.create({});
    this.tabId = 'tab-' + this.props.tabIndex;
  }

  onSelectActionTab = evt => {
    this.props.onSelectActionTab(evt.currentTarget.value);
  };

  onEnterActionTab = () => {};
  onLeaveActionTab = () => {};

  onSegmentCreated = id => {};
  onSegmentUpdated = id => {};
  onSegmentDeleted = id => {};
  onSegmentSelected = id => {};
  onSelectModel = model => {};
}

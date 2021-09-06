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
}

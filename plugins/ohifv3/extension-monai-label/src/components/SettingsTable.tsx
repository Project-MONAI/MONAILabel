import React, { Component } from 'react';

import './MonaiLabelPanel.styl';
import { Icon } from '@ohif/ui';
import { CookieUtils } from '../utils/GenericUtils';

export default class SettingsTable extends Component {
  constructor(props) {
    super(props);

    const onInfo = props.onInfo
    this.onInfo = onInfo

    this.state = this.getSettings();
  }

  getSettings = () => {
    const url = CookieUtils.getCookieString(
      'MONAILABEL_SERVER_URL',
      'http://' + window.location.host.split(':')[0] + ':8000/'
    );
    const overlap_segments = CookieUtils.getCookieBool(
      'MONAILABEL_OVERLAP_SEGMENTS',
      true
    );
    const export_format = CookieUtils.getCookieString(
      'MONAILABEL_EXPORT_FORMAT',
      'NRRD'
    );

    return {
      url: url,
      overlap_segments: overlap_segments,
      export_format: export_format,
    };
  };

  onBlurSeverURL = evt => {
    let url = evt.target.value;
    this.setState({ url: url });
    CookieUtils.setCookie('MONAILABEL_SERVER_URL', url);
  };

  render() {
    return (
      <table className="settingsTable">
        <tbody>
          <tr>
            <td>Server:</td>
            <td>
              <input
                className="actionInput"
                name="monailabelServerURL"
                type="text"
                defaultValue={this.state.url}
                onBlur={this.onBlurSeverURL}
              />
            </td>
            <td>&nbsp;</td>
            <td>
              <button className="actionButton" onClick={this.onInfo}>
                <Icon name="tool-reset" width="12px" height="12px" />
              </button>
            </td>
          </tr>
          <tr style={{ fontSize: 'smaller' }}>
            <td>&nbsp;</td>
            <td colSpan="4">
              <a
                href={new URL(this.state.url).toString() + 'info/'}
                target="_blank"
                rel="noopener noreferrer"
              >
                Info
              </a>
              <b>&nbsp;&nbsp;|&nbsp;&nbsp;</b>
              <a
                href={new URL(this.state.url).toString() + 'logs/?lines=100'}
                target="_blank"
                rel="noopener noreferrer"
              >
                Logs
              </a>
            </td>
          </tr>
        </tbody>
      </table>
    );
  }
}

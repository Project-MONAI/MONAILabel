import React, { Component } from 'react';

import './NextSampleForm.styl';
import PropTypes from 'prop-types';

export default class NextSampleForm extends Component {
  static propTypes = {
    info: PropTypes.any,
  };

  constructor(props) {
    super(props);
  }

  onSubmit = () => {
    // TODO:: OHIF Doesn't support loading exact series in URI
    let path = window.location.pathname.split('/');
    path[path.length - 1] = this.props.info.StudyInstanceUID;

    const pathname = path.join('/');
    console.log(pathname);

    let msg =
      'This action will reload current page.  Are you sure to continue?';
    if (!window.confirm(msg)) return;
    window.location.pathname = pathname;
  };

  render() {
    const fields = {
      id: 'Image ID (MONAILabel)',
      Modality: 'Modality',
      StudyDate: 'Study Date',
      StudyTime: 'Study Time',
      PatientID: 'Patient ID',
      StudyInstanceUID: 'Study Instance UID',
      SeriesInstanceUID: 'Series Instance UID',
    };
    return (
      <div>
        <table className="optionsTable">
          <thead>
            <tr>
              <th style={{ width: '30%' }}>Field</th>
              <th>Value</th>
            </tr>
          </thead>
          <tbody>
            {Object.keys(fields).map(field => (
              <tr key={field}>
                <td>{fields[field]}</td>
                {field === 'SeriesInstanceUID' ? (
                  <td>
                    <a
                      rel="noreferrer noopener"
                      target="_blank"
                      href={this.props.info['RetrieveURL']}
                    >
                      {this.props.info[field]}
                    </a>
                  </td>
                ) : (
                  <td>{this.props.info[field]}</td>
                )}
              </tr>
            ))}
          </tbody>
        </table>
        <br />
        <div className="mb-3 text-right">
          <button
            className="actionButton"
            type="submit"
            onClick={this.onSubmit}
          >
            OK
          </button>
        </div>
      </div>
    );
  }
}

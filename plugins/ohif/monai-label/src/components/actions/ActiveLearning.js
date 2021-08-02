import React from 'react';

import './OptionTable.styl';
import BaseTab from './BaseTab';

export default class OptionTable extends BaseTab {
  constructor(props) {
    super(props);
  }

  onClickNextSample = () => {};
  onClickUpdateModel = () => {};
  onClickSubmitLabel = () => {};

  render() {
    const ds = this.props.info.datastore;
    const completed = ds && ds.completed ? ds.completed : 0;
    const total = ds && ds.total ? ds.total : 1;
    const activelearning = 100 * (completed / total) + '%';

    const ts = this.props.info.train_stats;
    const epochs = ts && ts.total_time ? (ts.epoch ? ts.epoch : 1) : 0;
    const total_epochs = ts && ts.total_epochs ? ts.total_epochs : 1;
    const training = 100 * (epochs / total_epochs) + '%';

    const accuracy =
      ts && ts.best_metric ? Math.round(100 * ts.best_metric) + '%' : '0%';

    return (
      <div className="tab">
        <input
          className="tab-switch"
          type="checkbox"
          id={this.tabId}
          name="activelearning"
          value="activelearning"
          defaultChecked
        />
        <label className="tab-label" htmlFor={this.tabId}>
          Active Learning
        </label>
        <div className="tab-content">
          <table style={{ fontSize: 'smaller', width: '100%' }}>
            <tbody>
              <tr>
                <td>
                  <button
                    className="actionInput"
                    onClick={this.onClickNextSample}
                  >
                    Next Sample
                  </button>
                </td>
                <td>&nbsp;</td>
                <td>
                  <button
                    className="actionInput"
                    onClick={this.onClickUpdateModel}
                  >
                    Update Model
                  </button>
                </td>
                <td>&nbsp;&nbsp;&nbsp;&nbsp;</td>
                <td>
                  <button
                    className="actionInput"
                    onClick={this.onClickSubmitLabel}
                  >
                    Submit Label
                  </button>
                </td>
              </tr>
            </tbody>
          </table>
          <br />

          <table className="actionInput">
            <tbody>
              <tr>
                <td>Annotated:</td>
                <td width="80%">
                  <div className="w3-round w3-light-grey w3-tiny">
                    <div
                      className="w3-round w3-container w3-blue w3-center"
                      style={{ width: activelearning }}
                    >
                      {activelearning}
                    </div>
                  </div>
                </td>
              </tr>
              <tr>
                <td>Training:</td>
                <td>
                  <div className="w3-round w3-light-grey w3-tiny">
                    <div
                      className="w3-round w3-container w3-orange w3-center"
                      style={{ width: training }}
                    >
                      {training}
                    </div>
                  </div>
                </td>
              </tr>
              <tr>
                <td>Accuracy:</td>
                <td>
                  <div className="w3-round w3-light-grey w3-tiny">
                    <div
                      className="w3-round w3-container w3-green w3-center"
                      style={{ width: accuracy }}
                    >
                      {accuracy}
                    </div>
                  </div>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    );
  }
}

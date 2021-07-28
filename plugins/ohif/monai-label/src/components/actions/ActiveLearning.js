import React from 'react';

import './OptionTable.styl';
import BaseTab from './BaseTab';

export default class OptionTable extends BaseTab {
  constructor(props) {
    super(props);
  }

  render() {
    const ds = this.props.info.datastore;
    const completed = ds && ds.completed ? ds.completed : 0;
    const total = ds && ds.total ? ds.total : 1;
    const act_perecent = 100 * (completed / total) + '%';

    const ts = this.props.info.train_stats;
    const acc_perecent = ts && ts.best_metric ? ts.best_metric + '%' : '0%';

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
        <label className="tab-label" htmlFor="rd2">
          Active Learning
        </label>
        <div className="tab-content">
          <table style={{ fontSize: 'smaller', width: '100%' }}>
            <tbody>
              <tr>
                <td>
                  <button className="actionInput">Next Sample</button>
                </td>
                <td>&nbsp;</td>
                <td>
                  <button className="actionInput">Update Model</button>
                </td>
                <td>&nbsp;&nbsp;&nbsp;&nbsp;</td>
                <td>
                  <button className="actionInput">Submit Label</button>
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
                      style={{ width: act_perecent }}
                    >
                      {act_perecent}
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
                      style={{ width: acc_perecent }}
                    >
                      {acc_perecent}
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

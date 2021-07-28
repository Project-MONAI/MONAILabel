import React from 'react';

import './OptionTable.styl';
import BaseTab from './BaseTab';

export default class OptionTable extends BaseTab {
  constructor(props) {
    super(props);
  }

  render() {
    const info = this.props.info;
    const config = info && info.config ? info.config['train'] : {};

    return (
      <div className="tab">
        <input
          className="tab-switch"
          type="checkbox"
          id={this.tabId}
          name="options"
          value="options"
        />
        <label className="tab-label" htmlFor={this.tabId}>
          Options
        </label>
        <div className="tab-content">
          <table className="optionsTable">
            <tbody>
              {Object.entries(config).map(([k, v]) => (
                <tr key={k}>
                  <td>{k}</td>
                  <td>
                    {typeof v === 'boolean' ? (
                      <input type="checkbox" defaultChecked={v} />
                    ) : typeof v === 'object' ? (
                      <select className="optionsInput">
                        {Object.keys(v).map(a => (
                          <option key={a} name={a} value={a}>
                            {a}
                          </option>
                        ))}
                      </select>
                    ) : (
                      <input
                        type="text"
                        defaultValue={v}
                        className="optionsInput"
                      />
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    );
  }
}

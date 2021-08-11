import React from 'react';

import './OptionTable.styl';
import BaseTab from './BaseTab';

export default class OptionTable extends BaseTab {
  constructor(props) {
    super(props);
  }

  onChangeConfig = (c, n, evt) => {
    if (typeof this.props.info.config[c][n] === 'boolean')
      this.props.info.config[c][n] = evt.target.checked;
    else this.props.info.config[c][n] = evt.target.value;
    console.log(this.props.info.config);
  };

  render() {
    const info = this.props.info;
    const config = info && info.config ? info.config : {};

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
            <thead>
              <tr>
                <th>Action</th>
                <th>Name</th>
                <th>Value</th>
              </tr>
            </thead>
            <tbody>
              {Object.keys(config).map(c =>
                Object.entries(config[c] ? config[c] : {}).map(
                  ([k, v], index) => (
                    <tr key={c + k}>
                      {!index ? (
                        <td rowSpan={Object.keys(config[c]).length}>{c}</td>
                      ) : null}
                      <td>{k}</td>
                      <td>
                        {typeof v === 'boolean' ? (
                          <input
                            type="checkbox"
                            defaultChecked={v}
                            onChange={e => this.onChangeConfig(c, k, e)}
                          />
                        ) : typeof v === 'object' ? (
                          <select
                            className="optionsInput"
                            onChange={e => this.onChangeConfig(c, k, e)}
                          >
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
                            onChange={e => this.onChangeConfig(c, k, e)}
                          />
                        )}
                      </td>
                    </tr>
                  )
                )
              )}
            </tbody>
          </table>
        </div>
      </div>
    );
  }
}

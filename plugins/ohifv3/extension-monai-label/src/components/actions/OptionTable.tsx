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

import React from 'react';

import './OptionTable.styl';
import BaseTab from './BaseTab';

export default class OptionTable extends BaseTab {
  constructor(props) {
    super(props);

    this.state = {
      section: '',
      name: '',
      config: null,
    };
  }

  onChangeSection = evt => {
    this.state.section = evt.target.value;
    this.setState({ section: evt.target.value });
  };

  onChangeName = evt => {
    this.state.name = evt.target.value;
    this.setState({ name: evt.target.value });
  };

  onChangeConfig = (s, n, k, evt) => {
    console.debug(s + ' => ' + n + ' => ' + k);

    const c = this.state.config;
    if (typeof c[s][n][k] === 'boolean') {
      c[s][n][k] = !!evt.target.checked;
    } else {
      if (typeof c[s][n][k] === 'number')
        c[s][n][k] = Number.isInteger(c[s][n][k])
          ? parseInt(evt.target.value)
          : parseFloat(evt.target.value);
      else c[s][n][k] = evt.target.value;
    }
    this.setState({ config: c });
  };

  render() {
    let config = this.state.config ? this.state.config : {};
    if (!Object.keys(config).length) {
      const info = this.props.info;
      const mapping = {
        infer: 'models',
        train: 'trainers',
        activelearning: 'strategies',
        scoring: 'scoring',
      };
      for (const [m, n] of Object.entries(mapping)) {
        for (const [k, v] of Object.entries(info && info[n] ? info[n] : {})) {
          if (v && v.config && Object.keys(v.config).length) {
            if (!config[m]) config[m] = {};
            config[m][k] = v.config;
          }
        }
      }

      this.state.config = config;
    }

    const section =
      this.state.section.length && config[this.state.section]
        ? this.state.section
        : Object.keys(config).length
        ? Object.keys(config)[0]
        : '';
    this.state.section = section;
    const section_map = config[section] ? config[section] : {};

    const name =
      this.state.name.length && section_map[this.state.name]
        ? this.state.name
        : Object.keys(section_map).length
        ? Object.keys(section_map)[0]
        : '';
    this.state.name = name;
    const name_map = section_map[name] ? section_map[name] : {};

    //console.log('Section: ' + section + '; Name: ' + name);
    //console.log(name_map);

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
          <table>
            <tbody>
              <tr>
                <td>Section:</td>
                <td>
                  <select
                    className="selectBox"
                    name="selectSection"
                    onChange={this.onChangeSection}
                    value={this.state.section}
                  >
                    {Object.keys(config).map(k => (
                      <option key={k} value={k}>
                        {`${k} `}
                      </option>
                    ))}
                  </select>
                </td>
              </tr>
              <tr>
                <td>Name:</td>
                <td>
                  <select
                    className="selectBox"
                    name="selectName"
                    onChange={this.onChangeName}
                    value={this.state.name}
                  >
                    {Object.keys(section_map).map(k => (
                      <option key={k} value={k}>
                        {`${k} `}
                      </option>
                    ))}
                  </select>
                </td>
              </tr>
            </tbody>
          </table>

          <table className="optionsTable">
            <thead>
              <tr>
                <th>Key</th>
                <th>Value</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(name_map).map(([k, v]) => (
                <tr key={this.state.section + this.state.name + k}>
                  <td>{k}</td>
                  <td>
                    {v !== null && typeof v === 'boolean' ? (
                      <input
                        type="checkbox"
                        defaultChecked={v}
                        onChange={e =>
                          this.onChangeConfig(
                            this.state.section,
                            this.state.name,
                            k,
                            e
                          )
                        }
                      />
                    ) : v !== null && typeof v === 'object' ? (
                      <select
                        className="optionsInput"
                        onChange={e =>
                          this.onChangeConfig(
                            this.state.section,
                            this.state.name,
                            k,
                            e
                          )
                        }
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
                        defaultValue={v ? v : ''}
                        className="optionsInput"
                        onChange={e =>
                          this.onChangeConfig(
                            this.state.section,
                            this.state.name,
                            k,
                            e
                          )
                        }
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

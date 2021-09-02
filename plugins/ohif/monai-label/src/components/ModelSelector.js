import React, { Component } from 'react';
import PropTypes from 'prop-types';

import './ModelSelector.styl';

export default class ModelSelector extends Component {
  static propTypes = {
    name: PropTypes.string,
    title: PropTypes.string,
    models: PropTypes.array,
    usage: PropTypes.any,
    onClick: PropTypes.func,
  };

  constructor(props) {
    super(props);

    this.state = {
      models: props.models,
      currentModel: props.models.length > 0 ? props.models[0] : '',
      buttonDisabled: false,
    };
  }

  static getDerivedStateFromProps(props, current_state) {
    if (current_state.models !== props.models) {
      return {
        models: props.models,
        currentModel: props.models.length > 0 ? props.models[0] : '',
      };
    }
    return null;
  }

  onChangeModel = evt => {
    console.info('Selected Model: ' + evt.target.value);
    this.setState({ currentModel: evt.target.value });
  };

  onClickBtn = async () => {
    if (this.state.buttonDisabled) {
      return;
    }

    let model = this.state.currentModel;
    if (!model) {
      console.error('This should never happen!');
      model = this.props.models.length > 0 ? this.props.models[0] : '';
    }

    this.setState({ buttonDisabled: true });
    await this.props.onClick(model);
    this.setState({ buttonDisabled: false });
  };

  render() {
    return (
      <div className="modelSelector">
        <table>
          <tbody>
            <tr>
              <td colSpan="3">Models:</td>
            </tr>
            <tr>
              <td width="80%">
                <select
                  className="selectBox"
                  name={this.props.name + 'Select'}
                  onChange={this.onChangeModel}
                  value={this.state.currentModel}
                >
                  {this.props.models.map(model => (
                    <option key={model} value={model}>
                      {`${model} `}
                    </option>
                  ))}
                </select>
              </td>
              <td width="2%">&nbsp;</td>
              <td width="18%">
                <button
                  className="actionButton"
                  name={this.props.name + 'Button'}
                  onClick={this.onClickBtn}
                  title={'Run ' + this.props.title}
                  disabled={
                    this.state.isButtonDisabled || !this.props.models.length
                  }
                  style={{ display: this.props.onClick ? 'block' : 'none' }}
                >
                  Run
                </button>
              </td>
            </tr>
          </tbody>
        </table>
        {this.props.usage}
      </div>
    );
  }
}

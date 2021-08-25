import React from 'react';

import './AutoSegmentation.styl';
import ModelSelector from '../ModelSelector';
import BaseTab from './BaseTab';

export default class AutoSegmentation extends BaseTab {
  constructor(props) {
    super(props);

    this.modelSelector = React.createRef();
  }

  onSegmentation = async () => {
    const nid = this.notification.show({
      title: 'MONAI Label',
      message: 'Running Auto-Segmentation...',
      type: 'info',
      duration: 60000,
    });

    // TODO:: Fix Image ID...
    const { info, viewConstants } = this.props;
    const image = viewConstants.SeriesInstanceUID;
    const model = this.modelSelector.current.state.currentModel;

    const labels = info.models[model].labels;
    const response = await this.props.client().segmentation(model, image);

    // Bug:: Notification Service on show doesn't return id
    if (!nid) {
      window.snackbar.hideAll();
    } else {
      this.notification.hide(nid);
    }

    if (response.status !== 200) {
      this.notification.show({
        title: 'MONAI Label',
        message: 'Failed to Run Segmentation',
        type: 'error',
        duration: 5000,
      });
    } else {
      this.notification.show({
        title: 'MONAI Label',
        message: 'Run Segmentation - Successful',
        type: 'success',
        duration: 2000,
      });

      await this.props.updateView(response, labels);
    }
  };

  render() {
    let models = [];
    if (this.props.info && this.props.info.models) {
      for (let [name, model] of Object.entries(this.props.info.models)) {
        if (model.type === 'segmentation') {
          models.push(name);
        }
      }
    }

    return (
      <div className="tab">
        <input
          type="radio"
          name="rd"
          id={this.tabId}
          className="tab-switch"
          value="segmentation"
          onClick={this.onSelectActionTab}
          defaultChecked
        />
        <label htmlFor={this.tabId} className="tab-label">
          Segmentation
        </label>
        <div className="tab-content">
          <ModelSelector
            ref={this.modelSelector}
            name="segmentation"
            title="Segmentation"
            models={models}
            onClick={this.onSegmentation}
            usage={
              <p style={{ fontSize: 'smaller' }}>
                Fully automated segmentation <b>without any user input</b>. Just
                select a model and click to run
              </p>
            }
          />
        </div>
      </div>
    );
  }
}

import React from 'react';
import ModelSelector from '../ModelSelector';
import BaseTab from './BaseTab';
import { segmentColors } from './colormap';

export default class AutoSegmentation extends BaseTab {
  constructor(props) {
    super(props);

    this.modelSelector = React.createRef();
    this.state = {
      currentModel: null,
      selectedOrgans: {},
    };
    this.state.selectedOrgans = this.getModelOrgans();

  }

  onSelectModel = model => {
    this.setState({ currentModel: model });
  };

  getModelOrgans() {
    const selectedOrgans = {};


    if (this.props.viewConstants.SupportedClasses) {
      const labels = this.props.viewConstants.SupportedClasses
      let labelIndex = 1;

      for (const key in labels) {
        const organName = labels[key];
        console.log(organName)
        if (organName.toLowerCase() !== 'background') {
          const hexColor = segmentColors[labelIndex] || '#000000';

          selectedOrgans[organName] = { checked: false, color: hexColor };
          labelIndex++;
        }
      }
    }
    return selectedOrgans;
  }

  onSegmentation = async () => {
    const nid = this.notification.show({
      title: 'MONAI Label',
      message: 'Running Auto-Segmentation...',
      type: 'info',
      duration: 6000,
    });

    // TODO:: Fix Image ID...
    const { info, viewConstants } = this.props;
    const image = viewConstants.SeriesInstanceUID;
    const model = this.modelSelector.current.currentModel();
    const config = this.props.onOptionsConfig();
    const params =
      config && config.infer && config.infer[model] ? config.infer[model] : {};

    const selectedClasses = [];
    for (const organ in this.state.selectedOrgans) {
      if (this.state.selectedOrgans[organ].checked) {
        selectedClasses.push(organ);
      }
    }
    const data = {
      label_prompt : selectedClasses,
    }
    const updatedParams = {
      ...params,   // Spread the existing params
      ...data      // Add or override with the new data (label_prompt)
    };


    const labels = info.models[model].labels;
    const response = await this.props
      .client()
      .segmentation(model, image, updatedParams);

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


  onChangeOrgans = (organ, evt) => {
    this.setState((prevState) => {
      const selectedOrgans = { ...prevState.selectedOrgans };

      selectedOrgans[organ] = {
        ...selectedOrgans[organ],
        checked: evt.target.checked,
      };

      return { selectedOrgans };
    });
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
          Automatic Segmentation
        </label>
        <div className="tab-content">
          <ModelSelector
            ref={this.modelSelector}
            name="segmentation"
            title="Segmentation"
            models={models}
            currentModel={this.state.currentModel}
            onClick={this.onSegmentation}
            onSelectModel={this.onSelectModel}
            usage={
              <p style={{ fontSize: 'smaller' }}>
                Fully automated segmentation <b>with class prompt</b>. Just
                select classes and click <b>Run</b>
              </p>
            }
          />
        </div>
        <div className="tab-content">
            <div
              style={{
                height: '300px',
                overflowY: 'auto',
                border: '1px solid #000000',
                borderRadius: '4px',
                boxShadow: '0px 2px 4px rgba(0, 0, 0, 0.1)',
              }}
            >
              <div style={{ height: '100%' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                  <tbody>
                    {Object.entries(this.state.selectedOrgans).map(([organ, { checked, color }], index) => (
                      <tr key={index} style={{ height: '24px' }}>
                        <td style={{ padding: '2px 4px', verticalAlign: 'middle' }}>
                          <input
                            type="checkbox"
                            checked={checked}
                            onChange={(e) => this.onChangeOrgans(organ, e)}
                            style={{ width: '14px', height: '14px' }}
                          />
                        </td>
                        <td style={{ padding: '2px 4px' }}>
                          <span
                            className="segColor"
                            style={{
                              display: 'inline-block',
                              width: '12px',
                              height: '12px',
                              borderRadius: '50%',
                              backgroundColor: color, // Set color from segmentColors
                            }}
                          />
                        </td>
                        <td style={{ padding: '2px 4px', verticalAlign: 'middle' }}>{organ}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

        </div>
      </div>
    );
  }
}

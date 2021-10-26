import React from 'react';

import './Scribbles.styl';
import ModelSelector from '../ModelSelector';
import BaseTab from './BaseTab';
import cornerstoneTools from 'cornerstone-tools';
import {
  flattenLabelmaps,
  getLabelMaps,
} from '../../utils/SegmentationUtils';

export default class Scribbles extends BaseTab {
  constructor(props) {
    super(props);

    this.modelSelector = React.createRef();
    this.state = {
      currentModel: null,
    };
  }

  onSelectModel = model => {
    this.setState({ currentModel: model });
  };

  onSegmentation = async () => {
    const nid = this.notification.show({
      title: 'MONAI Label',
      message: 'Running Scribbles method - ' + this.modelSelector.current.currentModel(),
      type: 'info',
      duration: 60000,
    });

    const { getters } = cornerstoneTools.getModule('segmentation');
    const { labelmaps3D } = getters.labelmaps3D(
      this.props.viewConstants.element
    );
    if (!labelmaps3D) {
      this.notification.show({
        title: 'MONAI Label',
        message: 'Please create/select a label first',
        type: 'warning',
      });
      console.info('LabelMap3D is empty.. so zero segments');
      return;
    }

    // TODO:: Fix Image ID...
    const { info, viewConstants } = this.props;
    const image = viewConstants.SeriesInstanceUID;
    const model = this.modelSelector.current.currentModel();
    const config = this.props.onOptionsConfig();
    let params =
      config && config.infer && config.infer[model] ? config.infer[model] : {};

    // const labels = info.models[model].labels;
    let labels = info.models[model].labels;

    let label = null;
    
    // get label/scribbles
    const labelmap3D = labelmaps3D[0];
    console.debug(labelmap3D)
    if (!labelmap3D) {
      console.warn('Missing Label; so ignore');
      // continue;
    }

    const metadata = labelmap3D.metadata.data
      ? labelmap3D.metadata.data
      : labelmap3D.metadata;
    // if (!metadata || !metadata.length) {
    //   console.warn('Missing Meta; so ignore');
    //   continue;
    // }

    console.debug(metadata);

    const segments = flattenLabelmaps(
      getLabelMaps(this.props.viewConstants.element)
    );
    console.debug(segments);

    if (metadata.length !== segments.length + 1) {
      console.warn('Segments and Metadata NOT matching; So Ignore');
    }

    label = {name: "label", fileName: "label.bin", data: new Blob([labelmap3D.buffer], {
      type: 'application/octet-stream',
    })};
    params["label_info"] = segments;

    const response = await this.props
      .client()
      .segmentation(model, image, params, label);

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

      await this.props.updateView(response, labels, "overlap", undefined, undefined, 1);
    }
  };

  render() {
    let models = [];
    if (this.props.info && this.props.info.models) {
      for (let [name, model] of Object.entries(this.props.info.models)) {
        if (model.type === 'scribbles') {
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
        />
        <label htmlFor={this.tabId} className="tab-label">
          Scribbles
        </label>
        <div className="tab-content">
          <ModelSelector
            ref={this.modelSelector}
            name="segmentation"
            title="Scribbles"
            models={models}
            currentModel={this.state.currentModel}
            onClick={this.onSegmentation}
            onSelectModel={this.onSelectModel}
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

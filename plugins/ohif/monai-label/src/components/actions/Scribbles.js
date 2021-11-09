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
    this.main_label = null;
  }

  onSelectModel = model => {
    this.setState({ currentModel: model });
  };

  onSegmentation = async () => {
    const nid = this.notification.show({
      title: 'MONAI Label',
      message: 'Running Scribbles method: ' + this.modelSelector.current.currentModel(),
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

    const scribblesLabelMapIndex = this.props.getIndexByName("main_scribbles").labelmapIndex;
    const labels = info.models[model].labels;

    // get label/scribbles
    const labelmap3D = labelmaps3D[scribblesLabelMapIndex];
    console.debug(labelmap3D)
    if (!labelmap3D) {
      this.notification.show({
        title: 'MONAI Label',
        message: 'Missing Label; so ignore',
        type: 'warning',
      });
      console.warn('Missing Label; so ignore');
      return;
    }

    const metadata = labelmap3D.metadata.data
      ? labelmap3D.metadata.data
      : labelmap3D.metadata;   
    if (!metadata || !metadata.length) {
      this.notification.show({
        title: 'MONAI Label',
        message: 'Missing Meta; so ignore',
        type: 'warning',
      });
      console.warn('Missing Meta; so ignore');    
      return;
    }

    console.debug(metadata);

    // only select segments with labelmapIndex==scribblesLabelMapIndex
    const segments = flattenLabelmaps(
      getLabelMaps(this.props.viewConstants.element)
    ).filter(
      function(seg){
        return seg.labelmapIndex == scribblesLabelMapIndex;
      }
    );
    console.debug(segments);

    if (metadata.length !== segments.length + 1) {
      this.notification.show({
        title: 'MONAI Label',
        message: 'Segments and Metadata NOT matching; So Ignore',
        type: 'warning',
      });
      console.warn('Segments and Metadata NOT matching; So Ignore');
      return;
    }

    const label = {
      name: "label", fileName: "label.bin", data: new Blob([labelmap3D.buffer], {
        type: 'application/octet-stream',
      })
    };
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
        message: 'Failed to Run Scribbles Segmentation',
        type: 'error',
        duration: 5000,
      });
    } else {
      this.notification.show({
        title: 'MONAI Label',
        message: 'Run Scribbles method: ' + + this.modelSelector.current.currentModel() + ' successful',
        type: 'success',
        duration: 2000,
      });

      await this.props.updateView(
        response, 
        labels, 
        "override", 
        undefined, 
        undefined, 
        this.props.getIndexByName(this.main_label)
        );
    }
  };

  onSelectActionTab = evt => {
    console.info("Scribbles: SelectActionTab");
    this.props.onSelectActionTab(evt.currentTarget.value);
  };

  onScribblesExist = () => {
    // fetch both background and foreground scribbles
    let main_scribbles = this.props.getIndexByName("main_scribbles");
    let background_scribbles = this.props.getIndexByName("background_scribbles");
    let foreground_scribbles = this.props.getIndexByName("foreground_scribbles");

    // return true if scribbles volume exist
    return main_scribbles != null && background_scribbles != null && foreground_scribbles != null;
  }

  onEnterActionTab = () => {
    console.info("Scribbles: EnterActionTab");

    // select our brush tool
    cornerstoneTools.setToolActive('SphericalBrush', { mouseButtonMask: 1 });

    // fetch the segmentation volume, and add additional segments for scribbles
    const { getters } = cornerstoneTools.getModule('segmentation');
    const { labelmaps3D } = getters.labelmaps3D(
      this.props.viewConstants.element
    );

    // if empty, then add a main segmentation volume
    if (!labelmaps3D) {
      console.info('LabelMap3D is empty.. adding an empty segment');
      this.main_label = "generic";
      this.props.onAddSegment("generic", "generic tissue seg", "#D683E6", false, true);
    }
    else {
      this.main_label = this.props.getNameByIndex(this.props.getSelectedActiveIndex());
    }

    // if no scribbles segmentation volume exists then add them now
    if (!this.onScribblesExist()) {
      console.debug(this.onScribblesExist());
      console.debug("no scribbles segments found, adding....")

      this.props.onAddSegment("main_scribbles", "main segmentation volume for scribbles", "#E2EF83", false, true);
      this.props.onAddSegment("background_scribbles", "background scribbles", "#FF0000", false);
      this.props.onAddSegment("foreground_scribbles", "foreground scribbles", "#00FF00", false);

      // all done setting up scribbles volumes, now make one active
      this.setActiveScribbles("foreground_scribbles");
    }
    else {
      console.debug("scribbles segments already exist, skipping....")
    }
  };

  onLeaveActionTab = () => {
    console.info("Scribbles: LeaveActionTab");
    cornerstoneTools.setToolDisabled('SphericalBrush', {});

    // commenting the following to make scribbles persist even in other tabs
    // keeping this in here in case if the persist options needs to be disabled
    this.props.onDeleteSegmentByName("main_scribbles");
    this.props.onDeleteSegmentByName("background_scribbles");
    this.props.onDeleteSegmentByName("foreground_scribbles");
  };

  clearScribbles = () => {
    console.info("Scribbles: Clear Scribbles");
    this.props.onClearSegmentByName("main_scribbles");
    this.props.onClearSegmentByName("background_scribbles");
    this.props.onClearSegmentByName("foreground_scribbles");
  };

  setActiveScribbles = name => {
    const { element } = this.props.viewConstants;
    const activeIndex = this.props.getIndexByName(name);

    if(activeIndex){
      const { setters } = cornerstoneTools.getModule('segmentation');
      const { labelmapIndex, segmentIndex } = activeIndex;
  
      setters.activeLabelmapIndex(element, labelmapIndex);
      setters.activeSegmentIndex(element, segmentIndex);
  
      // Refresh
      cornerstone.updateImage(element);
  
    }
    else{
      console.info("Scribbles: setActiveScribbles - unable to find segment " + name);
    }    
  };

  onChangeScribbles = evt => {
    const value = evt.target.value;
    console.info(value);
    this.setActiveScribbles(value);
  }

  render() {
    let models = [];
    if (this.props.info && this.props.info.models) {
      for (let [name, model] of Object.entries(this.props.info.models)) {
        if (model.type === 'scribbles') {
          models.push(name);
        }
      }
    }

    let scribbles = [];
    scribbles.push("Foreground");
    scribbles.push("Background");

    return (
      <div className="tab">
        <input
          type="radio"
          name="rd"
          id={this.tabId}
          className="tab-switch"
          value="scribbles"
          onClick={this.onSelectActionTab}
        />
        <label htmlFor={this.tabId} className="tab-label">
          Scribbles
        </label>
        <div className="tab-content">
          <ModelSelector
            ref={this.modelSelector}
            name="scribbles"
            title="Scribbles"
            models={models}
            currentModel={this.state.currentModel}
            onClick={this.onSegmentation}
            onSelectModel={this.onSelectModel}
            scribblesSelector={
              <div>
                <tr>
                  <td width="18%">Label:</td>
                  <td width="2%">&nbsp;</td>
                  <td width="80%">
                    <select
                      name="scribblesSelectorBox"
                      className="selectBox"
                      onChange={this.onChangeScribbles}
                    >
                      {scribbles.map(scribbles => (
                        <option key={scribbles} name={scribbles} 
                          value={scribbles.toLowerCase() + "_scribbles"}>
                          {`${scribbles} `}
                        </option>
                      ))}
                    </select>
                  </td>
                </tr>
              </div>
            }
            usage={
              <div style={{ fontSize: 'smaller' }}>
                <p>
                  Select a scribbles layer, click to add and ctrl+click to remove scribbles.
                </p>
                <a href="#" onClick={() => this.clearScribbles()}>
                  Clear Scribbles
                </a>
              </div>

            }
          />
        </div>
      </div>
    );
  }
}

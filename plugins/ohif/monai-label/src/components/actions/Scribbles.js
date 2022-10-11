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

import './Scribbles.styl';
import ModelSelector from '../ModelSelector';
import BaseTab from './BaseTab';
import cornerstoneTools from 'cornerstone-tools';
import { flattenLabelmaps, getLabelMaps } from '../../utils/SegmentationUtils';

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
      message:
        'Running Scribbles method: ' +
        this.modelSelector.current.currentModel(),
      type: 'info',
      duration: 60000,
    });

    // get current labelmap nodes
    const { getters } = cornerstoneTools.getModule('segmentation');
    const { labelmaps3D } = getters.labelmaps3D(
      this.props.viewConstants.element
    );

    // if no labelmaps exists then exit
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

    // get the labelmap index for scribbles e.g. in 1+2 1 is labelmap index
    const scribblesLabelMapIndex = this.props.getIndexByName('main_scribbles')
      .labelmapIndex;
    const labels = info.models[model].labels;

    // get segmentation labels associated with scribbles
    const labelmap3D = labelmaps3D[scribblesLabelMapIndex];
    console.debug(labelmap3D);

    // if no scribbles labelmap nodes found then exit
    if (!labelmap3D) {
      this.notification.show({
        title: 'MONAI Label',
        message: 'Missing Label; so ignore',
        type: 'warning',
      });
      console.warn('Missing Label; so ignore');
      return;
    }

    // fetch relevant metadata and perform sanity checks
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

    // fetch scribbles segments info with labelmapIndex==scribblesLabelMapIndex
    const segments = flattenLabelmaps(
      getLabelMaps(this.props.viewConstants.element)
    ).filter(function(seg) {
      return seg.labelmapIndex == scribblesLabelMapIndex;
    });

    // convert  indices to expected label - needed for scribbles server to identify scribbles label
    for (let i = 0; i < segments.length; i++)
    {
      segments[i].id = parseInt(segments[i].id.split("+")[1]);
    }
    console.debug(segments);

    // perform sanity check and exit if not passed
    if (metadata.length !== segments.length + 1) {
      this.notification.show({
        title: 'MONAI Label',
        message: 'Segments and Metadata NOT matching; So Ignore',
        type: 'warning',
      });
      console.warn('Segments and Metadata NOT matching; So Ignore');
      return;
    }

    // prepare binary blob to send scribbles labelmap
    const label = {
      name: 'label',
      fileName: 'label.bin',
      data: new Blob([labelmap3D.buffer], {
        type: 'application/octet-stream',
      }),
    };
    params['label_info'] = segments;

    // send scribbles labelmap to server and capture response
    const response = await this.props
      .client()
      .segmentation(model, image, params, label);

    // Bug:: Notification Service on show doesn't return id
    if (!nid) {
      window.snackbar.hideAll();
    } else {
      this.notification.hide(nid);
    }

    // notify about the return status
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
        message:
          'Run Scribbles method: ' +
          +this.modelSelector.current.currentModel() +
          ' successful',
        type: 'success',
        duration: 2000,
      });

      // update labelmap with returned reponse from server
      await this.props.updateView(
        response,
        labels,
        'override',
        undefined,
        undefined,
        this.props.getIndexByName(this.main_label)
      );
    }
  };

  onSelectActionTab = evt => {
    console.info('Scribbles: SelectActionTab');
    this.props.onSelectActionTab(evt.currentTarget.value);
  };

  onScribblesExist = () => {
    // fetch main, background and foreground scribbles indices
    let main_scribbles = this.props.getIndexByName('main_scribbles');
    let background_scribbles = this.props.getIndexByName(
      'background_scribbles'
    );
    let foreground_scribbles = this.props.getIndexByName(
      'foreground_scribbles'
    );

    // return true if scribbles volumes exist
    return (
      main_scribbles != null &&
      background_scribbles != null &&
      foreground_scribbles != null
    );
  };

  onEnterActionTab = () => {
    console.info('Scribbles: EnterActionTab');

    // select our brush tool
    cornerstoneTools.setToolActive('SphericalBrush', { mouseButtonMask: 1 });

    // fetch the segmentation volume, and add additional segments for scribbles if needed
    const { getters } = cornerstoneTools.getModule('segmentation');
    const { labelmaps3D } = getters.labelmaps3D(
      this.props.viewConstants.element
    );

    // if empty, then add a main segmentation volume
    if (!labelmaps3D) {
      console.info('LabelMap3D is empty.. adding an empty segment');
      this.main_label = 'generic';
      this.props.onAddSegment(
        'generic',
        'generic tissue seg',
        '#D683E6',
        false,
        true
      );
    } else {
      this.main_label = this.props.getNameByIndex(
        this.props.getSelectedActiveIndex()
      );
    }

    // if no scribbles segmentation volume exists then add them now
    if (!this.onScribblesExist()) {
      console.debug(this.onScribblesExist());
      console.debug('no scribbles segments found, adding....');

      // add scribbles segmentation volumes as new set of labels
      this.props.onAddSegment(
        'main_scribbles',
        'main segmentation volume for scribbles',
        '#E2EF83',
        false,
        true
      );
      this.props.onAddSegment(
        'background_scribbles',
        'background scribbles',
        '#FF0000',
        false
      );
      this.props.onAddSegment(
        'foreground_scribbles',
        'foreground scribbles',
        '#00FF00',
        false
      );

      // all done setting up scribbles volumes,
      // last added volume will automatically be active at this point (i.e. foreground_scribbles)
    } else {
      console.debug('scribbles segments already exist, skipping....');
    }
  };

  onLeaveActionTab = () => {
    console.info('Scribbles: LeaveActionTab');

    // disable our brush tool
    cornerstoneTools.setToolDisabled('SphericalBrush', {});

    // delete scribbles segment
    // comment the following to make scribbles persist in other tabs
    this.props.onDeleteSegmentByName('main_scribbles');
    this.props.onDeleteSegmentByName('background_scribbles');
    this.props.onDeleteSegmentByName('foreground_scribbles');
  };

  clearScribbles = () => {
    console.info('Scribbles: Clear Scribbles');

    // clear scribbles segment
    this.props.onClearSegmentByName('main_scribbles');
    this.props.onClearSegmentByName('background_scribbles');
    this.props.onClearSegmentByName('foreground_scribbles');
  };

  setActiveScribbles = name => {
    // get element and selected index
    const { element } = this.props.viewConstants;
    const selectedIndex = this.props.getIndexByName(name);

    // update if scribbles volume with name exists
    if (selectedIndex) {
      const { setters } = cornerstoneTools.getModule('segmentation');
      const { labelmapIndex, segmentIndex } = selectedIndex;

      setters.activeLabelmapIndex(element, labelmapIndex);
      setters.activeSegmentIndex(element, segmentIndex);

      // Refresh
      cornerstone.updateImage(element);
    } else {
      console.info(
        'Scribbles: setActiveScribbles - unable to find segment ' + name
      );
      this.notification.show({
        title: 'MONAI Label',
        message: 'Unable to setActive scribbles volume ' + name,
        type: 'error',
        duration: 5000,
      });
    }
  };

  onChangeScribbles = evt => {
    const value = evt.target.value;
    console.info(value);
    this.setActiveScribbles(value);
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

    let scribbles = [];
    scribbles.push('Foreground');
    scribbles.push('Background');

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
                        <option
                          key={scribbles}
                          name={scribbles}
                          value={scribbles.toLowerCase() + '_scribbles'}
                        >
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
                  Select a scribbles layer, click to add and ctrl+click to
                  remove scribbles.
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

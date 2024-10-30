import React from 'react';
import './PointPrompts.css';
import ModelSelector from '../ModelSelector';
import BaseTab from './BaseTab';
import * as cornerstoneTools from '@cornerstonejs/tools';
import { hideNotification } from '../../utils/GenericUtils';
import { cache } from '@cornerstonejs/core';
import { segmentColors } from './colormap';
import { vec3 } from 'gl-matrix';

export default class PointPrompts extends BaseTab {
  modelSelector: any;

  constructor(props) {
    super(props);

    this.modelSelector = React.createRef();
    this.state = {
      currentModel: null,
      currentLabel: null,
      clickPoints: new Map(),
      availableOrgans: {},
    };
    this.state.selectedOrgans = this.getModelOrgans();

  }
  onEnterActionTab = () => {
    console.log('here in action tab')
    this.props.commandsManager.runCommand('setToolActive', {
      toolName: 'ProbeMONAILabel',
    });
    this.onSelectModel(this.state.currentModel);
    window.current_point_class = null;
    console.info('Here we activate the probe');
  };

  onLeaveActionTab = () => {
    this.onChangeLabel(null);
    this.props.commandsManager.runCommand('setToolDisable', {
      toolName: 'ProbeMONAILabel',
    });

    console.info('Here we deactivate the probe');
  };

  getModelOrgans() {
    const selectedOrgans = {};
    console.log(this.props.viewConstants.SupportedClasses)


    if (this.props.viewConstants.SupportedClasses) {
      const labels = this.props.viewConstants.SupportedClasses
      let labelIndex = 1; 

      for (const key in labels) {
        const organName = labels[key];
        if (organName.toLowerCase() !== 'background') {
          const hexColor = segmentColors[labelIndex] || '#000000'; 
          selectedOrgans[organName] = { checked: false, color: hexColor };
          labelIndex++;
        }
      }
    }
    return selectedOrgans;
  }

  onRunInference = async () => {
    const nid = this.notification.show({
      title: 'VISTA3D',
      message: 'Running Point Based Inference...',
      type: 'info',
      duration: 4000,
    });

    const manager = cornerstoneTools.annotation.state.getAnnotationManager();
    const annotations = manager.saveAnnotations(null, 'ProbeMONAILabel');

    const { currentLabel, clickPoints } = this.state;

    // Check if clickPoints is null or empty
    if (!annotations || Object.keys(annotations).length === 0) {
      hideNotification(nid, this.notification);
      this.notification.show({
        title: 'Notification',
        message: 'Error: clickPoints is empty or null',
        type: 'error',
        duration: 6000,
      });
      return;
    }

    clickPoints[currentLabel] = annotations;

    if (currentLabel === null || currentLabel === 'background') {
      this.notification.show({
        title: 'VISTA3D',
        message: 'Please click a foreground anatomy for point click editing',
        type: 'error',
        duration: 10000,
      });
      return;
    }

    const { info, viewConstants } = this.props;
    const image = viewConstants.SeriesInstanceUID;
    const model = this.modelSelector.current.currentModel();
    const config = this.props.onOptionsConfig();
    const params =
      config && config.infer && config.infer[model] ? config.infer[model] : {};
      

    // let seriesInstanceUID = viewConstants.SeriesInstanceUID;

    // const { displaySetService } = this.props.servicesManager.services;
    // const activeDisplaySets = displaySetService.activeDisplaySets
    // for (const item of activeDisplaySets){
    //   if (item.Modality === "CT"){
    //     seriesInstanceUID = item.SeriesInstanceUID;
    //   }
    // }



    // const models = info.models;
    // let selectedModel = 0;
    // for (const model of models) {
    //   if (!this.state.currentModel || model.id == this.state.currentModel)
    //     break;
    //   selectedModel++;
    // }

    // const model = models.length > 0 ? models[selectedModel] : null;
    // if (!model) {
    //   console.log('Something went error..');
    //   return;
    // }

    // console.log('Using model', model);
    // const model_id = model.id;
    // let label_names = [];

    const { cornerstoneViewportService, viewportGridService} = this.props.servicesManager.services;
    const viewPort = cornerstoneViewportService.viewportsById.get('mpr-axial');
    const { worldToIndex } = viewPort.viewportData.data[0].volume.imageData;

    // console.log(seriesInstanceUID);
    // console.log(viewPort);

    // const pts = cornerstoneTools.annotation.state.getAnnotations(
    //   'ProbeMONAILabel',
    //   viewPort.element
    // );

    // const pointsWorld = pts.map((pt) => pt.data.handles.points[0]);
    // const { activeViewportId } = viewportGridService.getState();

    // const viewPortIJK = cornerstoneViewportService.getCornerstoneViewport(activeViewportId);

    // const { imageData } = viewPortIJK.getImageData();
    // const ijk = vec3.fromValues(0, 0, 0);

    // // Rounding is not working
    // /* const pointsIJK = pointsWorld.map((world) =>
    //   Math.round(imageData.worldToIndex(world, ijk))
    // ); */

    // const pointsIJK = pointsWorld.map((world) =>
    //   imageData.worldToIndex(world, ijk)
    // );

    // console.log('pointsIJK', pointsIJK)


    const points = {};
    for (const label in clickPoints) {
      // console.log(clickPoints[label]);
      for (const uid in clickPoints[label]) {
        const annotations = clickPoints[label][uid]['ProbeMONAILabel'];
        points[label] = [];
        for (const annotation of annotations) {
          const pt = annotation.data.handles.points[0];
          points[label].push(worldToIndex(pt).map(Math.round));
        }
      }
      // label_names.push(label);
    }
    // const points = {};
    // for (const label in clickPoints) {
    //   // console.log(clickPoints[label]);
    //   for (const uid in clickPoints[label]) {
    //     const annotations = clickPoints[label][uid]['ProbeMONAILabel'];
    //     points[label] = [];
    //     for (const annotation of annotations) {
    //       const pt = annotation.data.handles.points[0];
    //       const indexPt = worldToIndex(pt).map(Math.round);


    //       // Swap the x (indexPt[0]) and y (indexPt[1]) coordinates
    //       const swappedPt = [indexPt[1], indexPt[0], indexPt[2]]; // Assuming pt has 3 components (x, y, z)

    //       points[label].push(swappedPt);
    //     }
    //   }
    //   // label_names.push(label);
    // }

    let supportedClassPoint
    console.log(' here in points!!!',points)
    const pointPrompts = {};
    const index2name = this.props.viewConstants.SupportedClasses

    for (const [label, coordinates] of Object.entries(points)) {
      // Check if the label's numeric value is >= 133


      if (label !== 'background') {
        pointPrompts[label] = coordinates;
        supportedClassPoint = Object.keys(index2name).find(
          (key) => index2name[key] === label
        );
      } else {
        pointPrompts[label] = coordinates;
      }
    }
    // if (points['background'] && points['background'].length > 0) {
    //   for (let i = 0; i < points['background'].length; i++) {
    //     pointPrompts.negative_label = points['background'];

    //     // pointPrompts['point_labels'].push(0);
    //   }
    // }

    // const supportedClassPointLabel = supportedClassPoint
    // ? supportedClassPoint.label + 1
    // : 119;

    const data = {
      label_prompt: [parseInt(supportedClassPoint)+1], // Converts to an integer
      result_dtype : 'uint8',
    };
    data['points'] = points[currentLabel];

    data['point_labels'] = new Array(data['points'].length).fill(1);

    if (points['background'] && points['background'].length > 0) {
      for (let i = 0; i < points['background'].length; i++) {
        data['point_labels'].push(0);
      }
      data['points'] = data['points'].concat(points['background']);
    }


    // params['label_prompt'] = [info.modelLabelToIdxMap[model_id][currentLabel]];

    // label_names = [currentLabel];
    // const data = {
    //   image: '/nim_workspace/CT', // '/nim_workspace/CT'
    //   prompts: {
    //     points: pointPrompts,
    //     classes: ['liver'],
    //   },
    // }
    const updatedParams = {
      ...params,   // Spread the existing params
      ...data      // Add or override with the new data (label_prompt)
    };
    console.log(' final point params data', updatedParams)
    const labels = info.models[model].labels;

    const response = await this.props
      .client()
      .segmentation(model, image, updatedParams);

    // const response = await this.props.client().inference(data);
    console.log(response)

    // hideNotification(nid, this.notification);
    if (response.status >= 400) {
      this.notification.show({
        title: 'VISTA3D',
        message: 'Failed to Run Point Prompts',
        type: 'error',
        duration: 6000,
      });
      console.log(response.data)
      return;
    }

    this.notification.show({
      title: 'VISTA3D',
      message: 'Run Point Prompts - Successful',
      type: 'success',
      duration: 4000,
    });

    this.props.updateView(response, labels, supportedClassPoint);
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

  segColorToRgb(s) {
    const c = s ? s.color : [0, 0, 0];
    return `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
  }
  initPoints = () => {
    const label = this.state.currentLabel;
    if (!label) {
      console.log('Current Label is Null (No need to init)');
      return;
    }
    console.log('In init points:', label)
    const { toolGroupService, viewportGridService } = this.props.servicesManager.services;
    const { viewports, activeViewportId } = viewportGridService.getState();
    const viewport = viewports.get(activeViewportId);
    console.log(viewport)

    const { viewportOptions } = viewport;
    const toolGroupId = viewportOptions.toolGroupId;

    // const colorMap = this.segmentInfo();
    // const customColor = this.segColorToRgb(colorMap[label]);

    const customColor = label.toLowerCase() === 'background' ? '#ff0000' : '#0521f5'; // Red for background, blue otherwise

    toolGroupService.setToolConfiguration(toolGroupId, 'ProbeMONAILabel', {
      customColor: customColor,
    });

    const annotations = this.state.clickPoints[label];
    if (annotations) {
      const manager = cornerstoneTools.annotation.state.getAnnotationManager();

      // annotations.forEach((annotation) => {
      //   // Preserve original color if it exists, otherwise fallback to annotation's default
      //   const originalColor = annotation.metadata.color || '#0521f5'; // Default to black if no color is set
      //   annotation.metadata.customColor = originalColor;
      // });

      manager.restoreAnnotations(annotations, null, 'ProbeMONAILabel');
    }
  };

  clearPoints = () => {
    cornerstoneTools.annotation.state.getAnnotationManager().removeAllAnnotations();
    this.props.servicesManager.services.cornerstoneViewportService.getRenderingEngine().render();
  };

  clearAllPoints = () => {
    const clickPoints = new Map();
    this.setState({ clickPoints: clickPoints });
    this.clearPoints();
  };


  onChangeLabel = name => {
    console.log(name, this.state.currentLabel);
    if (name === this.state.currentLabel) {
      console.log('Both new and prev are same');
      return;
    }
    const prev = this.state.currentLabel;

    if (prev !== 'background' && name !== 'background'){
      this.clearPoints();
    }

    const clickPoints = this.state.clickPoints;
    if (prev) {
      const manager = cornerstoneTools.annotation.state.getAnnotationManager();
      const annotations = manager.saveAnnotations(null, 'ProbeMONAILabel');
      console.log('Saving Prev annotations...', annotations);
      this.state.clickPoints[prev] = annotations;
    }

    this.state.currentLabel = name;
    window.current_point_class = name;
    this.setState({ currentLabel: name, clickPoints: clickPoints });
    this.initPoints();
  };

  render() {
    let models = [];
    if (this.props.info && this.props.info.models) {
      for (let [name, model] of Object.entries(this.props.info.models)) {
        if (
          name === 'vista3d'
        ) {
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
          value="pointprompts"
          onClick={this.onSelectActionTab}
        />
        <label htmlFor={this.tabId} className="tab-label">
          Interactive Segmentation
        </label>

        <div className='tab-content'>
          <ModelSelector
            ref={this.modelSelector}
            name="smartedit"
            title="Interactive Segmentation"
            models={models}
            currentModel={this.state.currentModel}
            onClick={this.onRunInference}
            onSelectModel={this.onSelectModel}
            usage={
              <div style={{ fontSize: 'smaller'}}>
                <p>
                  Create a label and annotate <b>any organ</b>.
                </p>
                <a style={{ backgroundColor:'lightgray'}} onClick={() => this.clearPoints()}>
                  Clear Points
                </a>
              </div>
            }
          />
          <div className='optionsTableContainer'
            style={{
              height: '300px',
              overflowY: 'auto',
              border: '1px solid #000000',
              borderRadius: '4px',
              boxShadow: '0px 2px 4px rgba(0, 0, 0, 0.1)',
            }}
          >
            <div className='bodyTableContainer'>
              <table className='optionsTable'>
                <tbody>
                <tr
                  key="background"
                  className='clickable-row'
                  style={{
                    backgroundColor: this.state.currentLabel === 'background' ? 'darkred' : 'transparent',
                    cursor: 'pointer',
                  }}
                  onClick={() => this.onChangeLabel('background')}>
                  <td>
                    <span
                      className='segColor'
                      style={{ 'backgroundColor': '#000000' }}
                      />
                  </td>
                  <td>Background</td>
                </tr>

                {Object.entries(this.state.selectedOrgans).map(([organ, { color, checked }]) => (
                  
                <tr
                  key={organ}
                  className='clickable-row'
                  style={{
                    backgroundColor: this.state.currentLabel === organ ? 'darkblue' : 'transparent',
                    cursor: 'pointer',
                  }}
                  onClick={() => this.onChangeLabel(organ)}
                >
                  <td>
                    <span
                      className='segColor'
                      style={{
                        backgroundColor: color,
                        display: 'inline-block',
                        width: '16px',
                        height: '16px',
                        borderRadius: '50%',
                      }}
                    />
                  </td>
                  <td>{organ}</td>
                </tr>

                ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>


        <div className="tab-content">
            <button
                className="actionInput"
                style={{backgroundColor:'lightgray'}}
                onClick={this.onRunInference}
              >
                Run
            </button>
        </div>

      </div>
    );
  }
}

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

import React, { Component } from 'react';
import PropTypes from 'prop-types';
import './MonaiLabelPanel.css';
import AutoSegmentation from './actions/AutoSegmentation';
import PointPrompts from './actions/PointPrompts';
import ClassPrompts from './actions/ClassPrompts';
import ActiveLearning from './actions/ActiveLearning';
import MonaiLabelClient from '../services/MonaiLabelClient';
import { hideNotification, getLabelColor } from '../utils/GenericUtils';
import { Enums } from '@cornerstonejs/tools';
import { cache, triggerEvent, eventTarget } from '@cornerstonejs/core';
import SegmentationReader from '../utils/SegmentationReader';
import { currentSegmentsInfo } from '../utils/SegUtils';
import SettingsTable from './SettingsTable';
import * as cornerstoneTools from '@cornerstonejs/tools';
import optionsInputDialog from './OptionsInputDialog';

export default class MonaiLabelPanel extends Component {
  static propTypes = {
    commandsManager: PropTypes.any,
    servicesManager: PropTypes.any,
    extensionManager: PropTypes.any,
  };

  notification: any;
  settings;
  actions: {
    activelearning: any;
    segmentation: any;
    pointprompts: any;
    classprompts: any;
  };
  serverURI = 'http://127.0.0.1:8000';

  // Private properties for segmentation management
  private _pendingSegmentationData: any = null;
  private _pendingRetryTimer: any = null;
  private _currentSegmentationSeriesUID: string | null = null;
  private _originCorrectedSeries: Set<string> = new Set();
  private _lastCheckedSeriesUID: string | null = null;
  private _seriesCheckInterval: any = null;

  constructor(props) {
    super(props);

    const { uiNotificationService } = props.servicesManager.services;
    this.notification = uiNotificationService;
    this.settings = React.createRef();
    this.actions = {
      activelearning: React.createRef(),
      segmentation: React.createRef(),
      pointprompts: React.createRef(),
      classprompts: React.createRef(),
    };

    this.state = {
      info: { models: [], datasets: [] },
      action: {},
      options: {},
    };
  }

  client = () => {
    const settings =
      this.settings && this.settings.current && this.settings.current.state
        ? this.settings.current.state
        : null;
    return new MonaiLabelClient(settings ? settings.url : this.serverURI);
  };

  segmentColor(label) {
    const color = getLabelColor(label);
    const rgbColor = [];
    for (const key in color) {
      rgbColor.push(color[key]);
    }
    rgbColor.push(255);
    return rgbColor;
  }

  getActiveViewportInfo = () => {
    const { viewportGridService, displaySetService } =
      this.props.servicesManager.services;
    const { viewports, activeViewportId } = viewportGridService.getState();
    const viewport = viewports.get(activeViewportId);
    const displaySet = displaySetService.getDisplaySetByUID(
      viewport.displaySetInstanceUIDs[0]
    );

    // viewportId = viewport.viewportId
    // SeriesInstanceUID = displaySet.SeriesInstanceUID;
    // StudyInstanceUID = displaySet.StudyInstanceUID;
    // FrameOfReferenceUID = displaySet.instances[0].FrameOfReferenceUID;
    // displaySetInstanceUID = displaySet.displaySetInstanceUID;
    // numImageFrames = displaySet.numImageFrames;
    return { viewport, displaySet };
  };

  onInfo = async (serverURI) => {
    const nid = this.notification.show({
      title: 'MONAI Label',
      message: 'Connecting to MONAI Label',
      type: 'info',
      duration: 2000,
    });

    this.serverURI = serverURI;
    const response = await this.client().info();
    console.log(response.data);

    hideNotification(nid, this.notification);
    if (response.status !== 200) {
      this.notification.show({
        title: 'MONAI Label',
        message: 'Failed to Connect to MONAI Label',
        type: 'error',
        duration: 5000,
      });
      return;
    }

    this.notification.show({
      title: 'MONAI Label',
      message: 'Connected to MONAI Label - Successful',
      type: 'success',
      duration: 2000,
    });

    const all_models = response.data.models;
    const all_model_names = Object.keys(all_models);
    const deepgrow_models = all_model_names.filter(
      (m) => all_models[m].type === 'deepgrow'
    );
    const deepedit_models = all_model_names.filter(
      (m) => all_models[m].type === 'deepedit'
    );
    const vista3d_models = all_model_names.filter(
      (m) => all_models[m].type === 'vista3d'
    );
    const segmentation_models = all_model_names.filter(
      (m) => all_models[m].type === 'segmentation'
    );
    const models = deepgrow_models
      .concat(deepedit_models)
      .concat(vista3d_models)
      .concat(segmentation_models);
    const all_labels = response.data.labels;

    const modelLabelToIdxMap = {};
    const modelIdxToLabelMap = {};
    const modelLabelNames = {};
    const modelLabelIndices = {};
    for (const model of models) {
      const labels = all_models[model]['labels'];
      modelLabelToIdxMap[model] = {};
      modelIdxToLabelMap[model] = {};
      if (Array.isArray(labels)) {
        for (let label_idx = 1; label_idx <= labels.length; label_idx++) {
          const label = labels[label_idx - 1];
          all_labels.push(label);
          modelLabelToIdxMap[model][label] = label_idx;
          modelIdxToLabelMap[model][label_idx] = label;
        }
      } else {
        for (const label of Object.keys(labels)) {
          const label_idx = labels[label];
          all_labels.push(label);
          modelLabelToIdxMap[model][label] = label_idx;
          modelIdxToLabelMap[model][label_idx] = label;
        }
      }
      modelLabelNames[model] = [
        ...Object.keys(modelLabelToIdxMap[model]),
      ].sort();
      modelLabelIndices[model] = [...Object.keys(modelIdxToLabelMap[model])]
        .sort()
        .map(Number);
    }

    const labelsOrdered = [...new Set(all_labels)].sort();
    
    // Prepare the initial segmentation configuration but DON'T create it yet
    // Segmentations will be created per-series when inference is actually run
    // This prevents creating a default segmentation with ID '1' that would interfere
    const initialSegs = labelsOrdered.reduce((acc, label, index) => {
            acc[index + 1] = {
              segmentIndex: index + 1,
              label: label,
              active: index === 0, // First segment is active
              locked: false,
              color: this.segmentColor(label),
            };
            return acc;
    }, {});
    
    console.log('[Initialization] Segmentation config prepared - will be created per-series on inference');
    console.log('[Initialization] Labels:', labelsOrdered);

    const info = {
      models: models,
      labels: labelsOrdered,
      data: response.data,
      modelLabelToIdxMap: modelLabelToIdxMap,
      modelIdxToLabelMap: modelIdxToLabelMap,
      modelLabelNames: modelLabelNames,
      modelLabelIndices: modelLabelIndices,
      initialSegs: initialSegs,
    };

    console.log(info);
    this.setState({ info: info });
    this.setState({ isDataReady: true }); // Mark as ready
    this.setState({ options: {} });
  };

  onSelectActionTab = (name) => {
    for (const action of Object.keys(this.actions)) {
      if (this.state.action === action) {
        if (this.actions[action].current) {
          this.actions[action].current.onLeaveActionTab();
        }
      }
    }

    for (const action of Object.keys(this.actions)) {
      if (name === action) {
        if (this.actions[action].current) {
          this.actions[action].current.onEnterActionTab();
        }
      }
    }
    this.setState({ action: name });
  };

  // Helper: Apply origin correction for multi-frame volumes
  applyOriginCorrection = (volumeLoadObject, logPrefix = '') => {
    try {
      const { displaySet } = this.getActiveViewportInfo();
      const imageVolumeId = displaySet.displaySetInstanceUID;
      let imageVolume = cache.getVolume(imageVolumeId);
      if (!imageVolume) {
        imageVolume = cache.getVolume('cornerstoneStreamingImageVolume:' + imageVolumeId);
      }
      
      console.log(`${logPrefix}[Origin] Checking correction`);
      console.log(`${logPrefix}[Origin]   Image origin:`, imageVolume?.origin);
      console.log(`${logPrefix}[Origin]   Seg origin:`, volumeLoadObject?.origin);
      
      if (imageVolume && displaySet.isMultiFrame) {
        const instance = displaySet.instances?.[0];
        if (instance?.PerFrameFunctionalGroupsSequence?.length > 0) {
          const firstFrame = instance.PerFrameFunctionalGroupsSequence[0];
          const lastFrame = instance.PerFrameFunctionalGroupsSequence[instance.PerFrameFunctionalGroupsSequence.length - 1];
          const firstIPP = firstFrame.PlanePositionSequence?.[0]?.ImagePositionPatient;
          const lastIPP = lastFrame.PlanePositionSequence?.[0]?.ImagePositionPatient;
          
          if (firstIPP && lastIPP && firstIPP.length === 3 && lastIPP.length === 3) {
            // Check if correction is needed (all 3 coordinates must match within tolerance)
            const tolerance = 0.01;
            const originMatchesFirst = 
              Math.abs(imageVolume.origin[0] - firstIPP[0]) < tolerance &&
              Math.abs(imageVolume.origin[1] - firstIPP[1]) < tolerance &&
              Math.abs(imageVolume.origin[2] - firstIPP[2]) < tolerance;
            
            // Track if this series has already been corrected to prevent double-correction
            const seriesUID = displaySet.SeriesInstanceUID;
            if (!this._originCorrectedSeries) {
              this._originCorrectedSeries = new Set();
            }
            const alreadyCorrected = this._originCorrectedSeries.has(seriesUID);
            
            console.log(`${logPrefix}[Origin] Origin check:`);
            console.log(`${logPrefix}[Origin]   Matches first frame: ${originMatchesFirst}`);
            console.log(`${logPrefix}[Origin]   Already corrected: ${alreadyCorrected}`);
            
            // Skip if already corrected in this session (prevents redundant corrections)
            if (alreadyCorrected) {
              // Don't log on every check - only log if this is not from the series monitor
              if (!logPrefix.includes('Origin Check')) {
                console.log(`${logPrefix}[Origin] ✓ Already corrected in this session, skipping`);
              }
              return false;
            }
            
            // Calculate the offset needed (will be [0,0,0] if origins already match)
            const originOffset = [
              firstIPP[0] - imageVolume.origin[0],
              firstIPP[1] - imageVolume.origin[1],
              firstIPP[2] - imageVolume.origin[2]
            ];
            
            console.log(`${logPrefix}[Origin] Applying correction`);
            console.log(`${logPrefix}[Origin]   First IPP:`, firstIPP);
            console.log(`${logPrefix}[Origin]   Offset:`, originOffset);
            
            // Update volume origins (even if they already match, this ensures consistency)
            imageVolume.origin = [firstIPP[0], firstIPP[1], firstIPP[2]];
            volumeLoadObject.origin = [firstIPP[0], firstIPP[1], firstIPP[2]];
            
            if (imageVolume.imageData) {
              imageVolume.imageData.setOrigin(imageVolume.origin);
            }
            if (volumeLoadObject.imageData) {
              volumeLoadObject.imageData.setOrigin(volumeLoadObject.origin);
            }
            
            // Adjust camera positions ONLY if there's a non-zero offset
            // If offset is zero, origins are already correct and cameras don't need adjustment
            const hasNonZeroOffset = originOffset[0] !== 0 || originOffset[1] !== 0 || originOffset[2] !== 0;
            
            if (hasNonZeroOffset) {
              console.log(`${logPrefix}[Origin]   Non-zero offset detected, adjusting viewport cameras`);
              const renderingEngine = this.props.servicesManager.services.cornerstoneViewportService.getRenderingEngine();
              if (renderingEngine) {
                const viewportIds = renderingEngine.getViewports().map(vp => vp.id);
                console.log(`${logPrefix}[Origin]   Adjusting ${viewportIds.length} viewport cameras`);
                
                viewportIds.forEach(viewportId => {
                  const viewport = renderingEngine.getViewport(viewportId);
                  if (viewport && viewport.getCamera) {
                    const camera = viewport.getCamera();
                    
                    const oldPosition = [...camera.position];
                    const oldFocalPoint = [...camera.focalPoint];
                    
                    camera.position = [
                      camera.position[0] + originOffset[0],
                      camera.position[1] + originOffset[1],
                      camera.position[2] + originOffset[2]
                    ];
                    camera.focalPoint = [
                      camera.focalPoint[0] + originOffset[0],
                      camera.focalPoint[1] + originOffset[1],
                      camera.focalPoint[2] + originOffset[2]
                    ];
                    viewport.setCamera(camera);
                    
                    console.log(`${logPrefix}[Origin]     Viewport ${viewportId}: Adjusted`);
                    console.log(`${logPrefix}[Origin]       Position: ${oldPosition} → ${camera.position}`);
                    console.log(`${logPrefix}[Origin]       Focal: ${oldFocalPoint} → ${camera.focalPoint}`);
                  }
                });
                
                renderingEngine.render();
              }
            } else {
              console.log(`${logPrefix}[Origin]   Offset is zero - origins already correct`);
              console.log(`${logPrefix}[Origin]   Attempting to reset viewport cameras to fix misalignment`);
              
              // When offset is zero but we're being called (e.g., after series switch),
              // the issue is that OHIF hasn't properly reset the viewport cameras
              // Try to reset each viewport to its default view
              const renderingEngine = this.props.servicesManager.services.cornerstoneViewportService.getRenderingEngine();
              if (renderingEngine) {
                const viewportIds = renderingEngine.getViewports().map(vp => vp.id);
                console.log(`${logPrefix}[Origin]   Resetting ${viewportIds.length} viewport cameras`);
                
                viewportIds.forEach(viewportId => {
                  const viewport = renderingEngine.getViewport(viewportId);
                  if (viewport && viewport.resetCamera) {
                    console.log(`${logPrefix}[Origin]     Viewport ${viewportId}: Calling resetCamera()`);
                    viewport.resetCamera();
                  } else if (viewport) {
                    console.log(`${logPrefix}[Origin]     Viewport ${viewportId}: No resetCamera() method available`);
                  }
                });
                
                renderingEngine.render();
              }
            }
            
            // Mark this series as corrected
            this._originCorrectedSeries.add(seriesUID);
            
            console.log(`${logPrefix}[Origin] ✓ Correction applied and series marked`);
            return true;
          }
        }
      }
      return false;
    } catch (e) {
      console.warn(`${logPrefix}[Origin] ✗ Error:`, e);
      return false;
    }
  };
  
  // Helper: Apply segment colors
  applySegmentColors = (segmentationId, labels, labelNames, logPrefix = '') => {
    try {
      const { viewport } = this.getActiveViewportInfo();
      if (viewport && labels && labelNames) {
        console.log(`${logPrefix}[Colors] Applying segment colors`);
        for (const label of labels) {
          const segmentIndex = labelNames[label];
          if (segmentIndex) {
            const color = this.segmentColor(label);
            cornerstoneTools.segmentation.config.color.setSegmentIndexColor(
              viewport.viewportId,
              segmentationId,
              segmentIndex,
              color
            );
            console.log(`${logPrefix}[Colors]   ${label} (${segmentIndex}):`, color);
          }
        }
        console.log(`${logPrefix}[Colors] ✓ Colors applied`);
        return true;
      }
      return false;
    } catch (e) {
      console.warn(`${logPrefix}[Colors] ✗ Error:`, e.message);
      return false;
    }
  };
  
  // Helper: Check and apply origin correction for current viewport
  // This is called when switching series to ensure existing segmentations are properly aligned
  ensureOriginCorrectionForCurrentSeries = () => {
    try {
      const currentViewportInfo = this.getActiveViewportInfo();
      const currentSeriesUID = currentViewportInfo?.displaySet?.SeriesInstanceUID;
      const segmentationId = `seg-${currentSeriesUID || 'default'}`;
      
      // Check if this series has a segmentation
      const segmentationService = this.props.servicesManager.services.segmentationService;
      
      let volumeLoadObject = null;
      try {
        volumeLoadObject = segmentationService.getLabelmapVolume(segmentationId);
      } catch (e) {
        // Segmentation doesn't exist yet - this is normal during early checks
        return;
      }
      
      if (volumeLoadObject) {
        console.log('[Origin Check] ========================================');
        console.log('[Origin Check] Found segmentation for', currentSeriesUID);
        const correctionApplied = this.applyOriginCorrection(volumeLoadObject, '[Origin Check] ');
        if (correctionApplied) {
          console.log('[Origin Check] ✓ Correction successfully applied');
        } else {
          console.log('[Origin Check] ✓ No correction needed (already applied)');
        }
        console.log('[Origin Check] ========================================');
      }
    } catch (e) {
      console.error('[Origin Check] Error:', e);
      console.error('[Origin Check] Stack:', e.stack);
    }
  };
  
  // Helper: Apply segmentation data to volume
  applySegmentationDataToVolume = (volumeLoadObject, segmentationId, data, modelToSegMapping, override, label_class_unknown, labels, labelNames, logPrefix = '') => {
    try {
      console.log(`${logPrefix}[Data] Converting and applying voxel data`);
      
      // Convert the data with proper label mapping
      let convertedData = data;
      for (let i = 0; i < convertedData.length; i++) {
        const midx = convertedData[i];
        const sidx = modelToSegMapping[midx];
        if (midx && sidx) {
          convertedData[i] = sidx;
        } else if (override && label_class_unknown && labels.length === 1) {
          convertedData[i] = midx ? labelNames[labels[0]] : 0;
        } else if (labels.length > 0) {
          convertedData[i] = 0;
        }
      }
      
      // Apply origin correction
      this.applyOriginCorrection(volumeLoadObject, logPrefix);
      
      // Apply segment colors
      this.applySegmentColors(segmentationId, labels, labelNames, logPrefix);
      
      // Set the voxel data
      volumeLoadObject.voxelManager.setCompleteScalarDataArray(convertedData);
      triggerEvent(eventTarget, Enums.Events.SEGMENTATION_DATA_MODIFIED, {
        segmentationId: segmentationId
      });
      
      console.log(`${logPrefix}[Data] ✓✓✓ Segmentation applied for ${segmentationId}`);
      return true;
    } catch (e) {
      console.error(`${logPrefix}[Data] ✗ Error:`, e);
      return false;
    }
  };

  updateView = async (
    response,
    model_id,
    labels,
    override = false,
    label_class_unknown = false,
    sidx = -1
  ) => {
    console.log('UpdateView: ', {
      model_id,
      labels,
      override,
      label_class_unknown,
      sidx,
    });
    const ret = SegmentationReader.parseNrrdData(response.data);
    if (!ret) {
      throw new Error('Failed to parse NRRD data');
    }
    
    // Log NRRD metadata received from server
    console.log('[NRRD Client] Received NRRD from server:');
    console.log('[NRRD Client]   Dimensions:', ret.header.sizes);
    console.log('[NRRD Client]   Space Origin:', ret.header.spaceOrigin);
    console.log('[NRRD Client]   Space Directions:', ret.header.spaceDirections);
    console.log('[NRRD Client]   Space:', ret.header.space);

    const labelNames = {};
    const currentSegs = currentSegmentsInfo(
      this.props.servicesManager.services.segmentationService
    );
    const modelToSegMapping = {};
    modelToSegMapping[0] = 0;

    let tmp_model_seg_idx = 1;
    for (const label of labels) {
      const s = currentSegs.info[label];
      if (!s) {
        for (let i = 1; i <= 255; i++) {
          if (!currentSegs.indices.has(i)) {
            labelNames[label] = i;
            currentSegs.indices.add(i);
            break;
          }
        }
      } else {
        labelNames[label] = s.segmentIndex;
      }

      const seg_idx = labelNames[label];
      let model_seg_idx = this.state.info.modelLabelToIdxMap[model_id][label];
      model_seg_idx = model_seg_idx ? model_seg_idx : tmp_model_seg_idx;
      modelToSegMapping[model_seg_idx] = 0xff & seg_idx;
      tmp_model_seg_idx++;
    }

    console.log('Index Remap', labels, modelToSegMapping);
    const data = new Uint8Array(ret.image);

    // Get series-specific segmentation ID to ensure each series has its own segmentation
    const currentViewportInfo = this.getActiveViewportInfo();
    const currentSeriesUID = currentViewportInfo?.displaySet?.SeriesInstanceUID;
    const segmentationId = `seg-${currentSeriesUID || 'default'}`;

    console.log('[Segmentation ID] Using series-specific ID:', segmentationId);
    console.log('[Segmentation ID] Series UID:', currentSeriesUID);

    // Track the current series for logging purposes
    console.log('[Series Tracking] Current series:', currentSeriesUID);
    console.log('[Series Tracking] Previous series:', this._currentSegmentationSeriesUID);

    if (this._currentSegmentationSeriesUID && this._currentSegmentationSeriesUID !== currentSeriesUID) {
      console.log('[Series Switch] Switched from', this._currentSegmentationSeriesUID, 'to', currentSeriesUID);
      console.log('[Series Switch] Each series has its own segmentation ID - no cleanup needed');
      
      // Clear the origin correction flag for the current series
      // This ensures origin correction will be reapplied if needed when switching back
      // (OHIF may have reset camera positions during series switch)
      if (this._originCorrectedSeries && this._originCorrectedSeries.has(currentSeriesUID)) {
        console.log('[Series Switch] Clearing origin correction flag for', currentSeriesUID);
        console.log('[Series Switch] This allows re-checking/re-applying correction after series switch');
        this._originCorrectedSeries.delete(currentSeriesUID);
      }
    }

    // Store the current series UID for future checks
    this._currentSegmentationSeriesUID = currentSeriesUID;

    const { segmentationService } = this.props.servicesManager.services;
    let volumeLoadObject = null;
    try {
      volumeLoadObject = segmentationService.getLabelmapVolume(segmentationId);
    } catch (e) {
      console.log('[Segmentation] Could not get labelmap volume:', e.message);
    }

    if (volumeLoadObject) {
      console.log('[Segmentation] Volume exists, applying data directly');
      
      // Handle override mode (partial update of specific slice)
      let dataToApply = data;
      if (override === true) {
        console.log('[Segmentation] Override mode: merging with existing data');
        const { voxelManager } = volumeLoadObject;
        const scalarData = voxelManager?.getCompleteScalarDataArray();
        const currentSegArray = new Uint8Array(scalarData.length);
        currentSegArray.set(scalarData);
        
        // Convert new data first
        let convertedData = new Uint8Array(data);
      for (let i = 0; i < convertedData.length; i++) {
        const midx = convertedData[i];
          const sidx_mapped = modelToSegMapping[midx];
          if (midx && sidx_mapped) {
            convertedData[i] = sidx_mapped;
        } else if (override && label_class_unknown && labels.length === 1) {
          convertedData[i] = midx ? labelNames[labels[0]] : 0;
        } else if (labels.length > 0) {
          convertedData[i] = 0;
        }
      }

        // Merge with existing data
        const updateTargets = new Set(convertedData);
        const numImageFrames = this.getActiveViewportInfo().displaySet.numImageFrames;
        const sliceLength = scalarData.length / numImageFrames;
        const sliceBegin = sliceLength * sidx;
        const sliceEnd = sliceBegin + sliceLength;
        for (let i = 0; i < convertedData.length; i++) {
          if (sidx >= 0 && (i < sliceBegin || i >= sliceEnd)) {
            continue;
          }
          if (convertedData[i] !== 255 && updateTargets.has(currentSegArray[i])) {
            currentSegArray[i] = convertedData[i];
          }
        }
        dataToApply = currentSegArray;
      }
      
      // Use shared helper method to apply data, origin correction, and colors
      this.applySegmentationDataToVolume(
        volumeLoadObject,
        segmentationId,
        dataToApply,
        modelToSegMapping,
        override,
        label_class_unknown,
        labels,
        labelNames,
        '[Main] '
      );
    } else {
      console.log('[Segmentation] No cached volume - this is first inference or after series switch');
      console.log('[Segmentation] Storing data for later - will be picked up by OHIF on next render');
      
      // Cancel any pending retries from a previous series
      if (this._pendingRetryTimer) {
        console.log('[Segmentation] Cancelling previous pending retries');
        clearTimeout(this._pendingRetryTimer);
        this._pendingRetryTimer = null;
      }
      
      // Store the segmentation data so it can be applied when OHIF creates the volume
      // This happens automatically when the viewport renders
      // Tag it with the current series UID to ensure we don't apply it to wrong series
      this._pendingSegmentationData = {
        data: data,
        modelToSegMapping: modelToSegMapping,
        override: override,
        label_class_unknown: label_class_unknown,
        labels: labels,
        labelNames: labelNames,
        seriesUID: currentSeriesUID,
        segmentationId: segmentationId
      };
      
      console.log('[Segmentation] Data stored for series:', currentSeriesUID);
      console.log('[Segmentation] Will retry applying data');
      
      // Start retry mechanism
      const tryApplyPendingData = (attempt = 1, maxAttempts = 50) => {
        const delay = attempt * 200; // 200ms, 400ms, 600ms, etc.
        
        this._pendingRetryTimer = setTimeout(() => {
          console.log(`[Segmentation] Retry ${attempt}/${maxAttempts}: Checking for volume`);
          try {
            // First, verify we're still on the same series
            const currentViewportInfo = this.getActiveViewportInfo();
            const currentActiveSeriesUID = currentViewportInfo?.displaySet?.SeriesInstanceUID;
            const pendingDataSeriesUID = this._pendingSegmentationData?.seriesUID;
            
            if (currentActiveSeriesUID !== pendingDataSeriesUID) {
              console.log(`[Segmentation] Retry ${attempt}: Series changed!`);
              console.log(`[Segmentation]   Pending data for series: ${pendingDataSeriesUID}`);
              console.log(`[Segmentation]   Current active series: ${currentActiveSeriesUID}`);
              console.log(`[Segmentation]   Aborting retry - data is for different series`);
              this._pendingSegmentationData = null;
              this._pendingRetryTimer = null;
              return;
            }
            
            console.log(`[Segmentation] Retry ${attempt}: Confirmed still on series ${currentActiveSeriesUID}`);
            
            // Check if segmentations exist in the service first
            const segmentationService = this.props.servicesManager.services.segmentationService;
            const allSegmentations = segmentationService.getSegmentations();
            const pendingSegmentationId = this._pendingSegmentationData?.segmentationId;
            
            console.log(`[Segmentation] Retry ${attempt}: Available segmentations:`, Object.keys(allSegmentations || {}));
            
            // Check cache for volume
            const cachedVolume = cache.getVolume(pendingSegmentationId);
            console.log(`[Segmentation] Retry ${attempt}: Cache volume '${pendingSegmentationId}' exists:`, !!cachedVolume);
            
            let retryVolumeLoadObject = null;
            try {
              retryVolumeLoadObject = segmentationService.getLabelmapVolume(pendingSegmentationId);
              console.log(`[Segmentation] Retry ${attempt}: Got labelmap volume from service`);
            } catch (e) {
              console.log(`[Segmentation] Retry ${attempt}: Cannot get labelmap volume:`, e.message);
            }
            
            // Check if the segmentation for THIS series exists (not just any segmentation)
            const segmentationExistsForThisSeries = allSegmentations && allSegmentations[pendingSegmentationId];
            
            if (!segmentationExistsForThisSeries) {
              console.log(`[Segmentation] Retry ${attempt}: Segmentation for this series doesn't exist yet`);
              
              // After a series switch, we need to create the segmentation for the new series
              // Try this on attempt 3 to give OHIF time to initialize
              if (attempt === 3) {
                console.log(`[Segmentation] Retry ${attempt}: Creating segmentation for new series`);
                try {
                  // Get the segment configuration from state
                  const initialSegs = this.state.info?.initialSegs;
                  const labelsOrdered = this.state.info?.labels;
                  
                  if (initialSegs && labelsOrdered) {
                    const segmentations = [{
                      segmentationId: pendingSegmentationId,
                      representation: {
                        type: Enums.SegmentationRepresentations.Labelmap
                      },
                      config: {
                        label: 'Segmentations',
                        segments: initialSegs
                      }
                    }];
                    
                    this.props.commandsManager.runCommand('loadSegmentationsForViewport', {
                      segmentations
                    });
                    console.log(`[Segmentation] Retry ${attempt}: Triggered segmentation creation for ${pendingSegmentationId}`);
                  } else {
                    console.log(`[Segmentation] Retry ${attempt}: Cannot create - segment config not available in state`);
                  }
                } catch (e) {
                  console.log(`[Segmentation] Retry ${attempt}: Could not create segmentation:`, e.message);
                }
              }
            } else if (!retryVolumeLoadObject && attempt % 5 === 0) {
              // If we have a segmentation in the service but no volume, try to trigger viewport render
              console.log(`[Segmentation] Retry ${attempt}: Triggering viewport render to force volume creation`);
              try {
                const renderingEngine = this.props.servicesManager.services.cornerstoneViewportService.getRenderingEngine();
                if (renderingEngine) {
                  renderingEngine.render();
                }
              } catch (e) {
                console.log(`[Segmentation] Retry ${attempt}: Could not trigger render:`, e.message);
              }
            }
            
            if (retryVolumeLoadObject && retryVolumeLoadObject.voxelManager && this._pendingSegmentationData) {
              console.log(`[Segmentation] Retry ${attempt}: ✓ Volume now exists, applying pending data`);
              
              const { data, modelToSegMapping, override, label_class_unknown, labels, labelNames } = this._pendingSegmentationData;
              
              // Use shared helper method to apply data, origin correction, and colors
              const success = this.applySegmentationDataToVolume(
                retryVolumeLoadObject,
                pendingSegmentationId,
                data,
                modelToSegMapping,
                override,
                label_class_unknown,
                labels,
                labelNames,
                `[Retry ${attempt}] `
              );
              
              if (success) {
                this._pendingSegmentationData = null;
                this._pendingRetryTimer = null;
              } else {
                console.error(`[Segmentation] Retry ${attempt}: Failed to apply data`);
              }
            } else if (attempt < maxAttempts) {
              console.log(`[Segmentation] Retry ${attempt}: Volume not ready, will try again`);
              tryApplyPendingData(attempt + 1, maxAttempts);
            } else {
              console.error('[Segmentation] ❌ Failed to apply segmentation after', maxAttempts, 'attempts');
              console.error('[Segmentation] Final diagnostics:');
              console.error('[Segmentation]   - Segmentations in service:', allSegmentations ? Object.keys(allSegmentations) : 'none');
              console.error('[Segmentation]   - Volume in cache:', !!cachedVolume);
              console.error('[Segmentation]   - Labelmap volume available:', !!retryVolumeLoadObject);
              
              this._pendingSegmentationData = null;
              this._pendingRetryTimer = null;
              
              // Show a user notification
              if (this.notification) {
                this.notification.show({
                  title: 'Segmentation Error',
                  message: 'Failed to apply segmentation data. Please ensure the viewport is active and try again.',
                  type: 'error',
                  duration: 5000
                });
              }
            }
          } catch (e) {
            console.error(`[Segmentation] Retry ${attempt}: Error:`, e);
            if (attempt < maxAttempts) {
              tryApplyPendingData(attempt + 1, maxAttempts);
            } else {
              // Max attempts reached after error
              this._pendingSegmentationData = null;
              this._pendingRetryTimer = null;
            }
          }
        }, delay);
      };
      
      // Start the retry process
      tryApplyPendingData();
    }
  };

  openConfigurations = (e) => {
    e.preventDefault();

    const { uiDialogService } = this.props.servicesManager.services;
    optionsInputDialog(
      uiDialogService,
      this.state.options,
      this.state.info,
      (options, actionId) => {
        if (actionId === 'save' || actionId == 'reset') {
          this.setState({ options: options });
        }
      }
    );
  };

  async componentDidMount() {
    if (this.state.isDataReady) {
      return;
    }

    console.log('(Component Mounted) Ready to Connect to MONAI Server...');
    
    // Set up periodic check for series changes to apply origin correction
    // This handles the case where user switches series by clicking in the left panel
    // without running new inference or entering/leaving tabs
    console.log('[Series Monitor] Starting periodic series change detection');
    this._lastCheckedSeriesUID = null;
    this._seriesCheckInterval = setInterval(() => {
      try {
        const currentViewportInfo = this.getActiveViewportInfo();
        const currentSeriesUID = currentViewportInfo?.displaySet?.SeriesInstanceUID;
        
        // If series changed since last check
        if (currentSeriesUID && currentSeriesUID !== this._lastCheckedSeriesUID) {
          console.log('[Series Monitor] Series change detected:', this._lastCheckedSeriesUID, '→', currentSeriesUID);
          this._lastCheckedSeriesUID = currentSeriesUID;
          
          // Clear the origin correction flag for the current series
          // This ensures origin correction will be reapplied if needed when switching back
          // (OHIF resets camera positions during series switch)
          if (this._originCorrectedSeries && this._originCorrectedSeries.has(currentSeriesUID)) {
            console.log('[Series Monitor] Clearing origin correction flag for', currentSeriesUID);
            console.log('[Series Monitor] This allows re-checking/re-applying correction after series switch');
            this._originCorrectedSeries.delete(currentSeriesUID);
          }
          
          // Apply origin correction with multiple attempts at different intervals
          // to catch the segmentation as soon as it's loaded and minimize visual glitch
          // Try immediately (might be too early but worth a shot)
          setTimeout(() => {
            console.log('[Series Monitor] Attempt 1: Applying origin correction for', currentSeriesUID);
            this.ensureOriginCorrectionForCurrentSeries();
          }, 50);
          
          // Try again soon
          setTimeout(() => {
            console.log('[Series Monitor] Attempt 2: Re-checking origin correction for', currentSeriesUID);
            this.ensureOriginCorrectionForCurrentSeries();
          }, 150);
          
          // Final attempt
          setTimeout(() => {
            console.log('[Series Monitor] Attempt 3: Final check for origin correction for', currentSeriesUID);
            this.ensureOriginCorrectionForCurrentSeries();
          }, 300);
        }
      } catch (e) {
        // Silently ignore errors during periodic check
        // (e.g., if viewport is not yet initialized)
      }
    }, 1000); // Check every second
    
    // await this.onInfo();
  }
  
  componentWillUnmount() {
    // Clean up the series monitoring interval
    if (this._seriesCheckInterval) {
      console.log('[Series Monitor] Stopping periodic series change detection');
      clearInterval(this._seriesCheckInterval);
      this._seriesCheckInterval = null;
    }
  }

  onOptionsConfig = () => {
    return this.state.options;
  };

  render() {
    const { isDataReady } = this.state;
    return (
      <div className="monaiLabelPanel">
        <br style={{ margin: '3px' }} />

        <SettingsTable ref={this.settings} onInfo={this.onInfo} />
        {isDataReady && (
          <div style={{ color: 'white' }}>
            <p className="subtitle">{this.state.info.data.name}</p>
            <br />
            <hr className="separator" />
            <a href="#" onClick={this.openConfigurations}>
              Options / Configurations
            </a>
            <hr className="separator" />
          </div>
        )}
        {isDataReady && (
          <div className="tabs scrollbar" id="style-3">
            <ActiveLearning
              ref={this.actions['activelearning']}
              tabIndex={1}
              info={this.state.info}
              client={this.client}
              updateView={this.updateView}
              onSelectActionTab={this.onSelectActionTab}
              onOptionsConfig={this.onOptionsConfig}
              getActiveViewportInfo={this.getActiveViewportInfo}
            />
            <AutoSegmentation
              ref={this.actions['segmentation']}
              tabIndex={2}
              info={this.state.info}
              client={this.client}
              updateView={this.updateView}
              onSelectActionTab={this.onSelectActionTab}
              onOptionsConfig={this.onOptionsConfig}
              getActiveViewportInfo={this.getActiveViewportInfo}
            />
            <PointPrompts
              ref={this.actions['pointprompts']}
              tabIndex={3}
              info={this.state.info}
              client={this.client}
              updateView={this.updateView}
              onSelectActionTab={this.onSelectActionTab}
              onOptionsConfig={this.onOptionsConfig}
              getActiveViewportInfo={this.getActiveViewportInfo}
              servicesManager={this.props.servicesManager}
              commandsManager={this.props.commandsManager}
              ensureOriginCorrectionForCurrentSeries={this.ensureOriginCorrectionForCurrentSeries}
            />
            <ClassPrompts
              ref={this.actions['classprompts']}
              tabIndex={4}
              info={this.state.info}
              client={this.client}
              updateView={this.updateView}
              onSelectActionTab={this.onSelectActionTab}
              onOptionsConfig={this.onOptionsConfig}
              getActiveViewportInfo={this.getActiveViewportInfo}
              servicesManager={this.props.servicesManager}
              commandsManager={this.props.commandsManager}
            />
          </div>
        )}
      </div>
    );
  }
}

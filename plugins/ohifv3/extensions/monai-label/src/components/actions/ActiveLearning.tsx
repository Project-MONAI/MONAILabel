import React from 'react';
import BaseTab from './BaseTab';
import { cache, triggerEvent, eventTarget } from '@cornerstonejs/core';
import { Enums } from '@cornerstonejs/tools';
import NextSampleForm from './NextSampleForm';

export default class ActiveLearning extends BaseTab {
  constructor(props) {
    super(props);
  }

  onClickNextSample = async () => {
    const nid = this.notification.show({
      title: 'MONAI Label',
      message: 'Fetching Next Recommended Image for Annotation',
      type: 'info',
      duration: 4000,
    });

    const response = await this.props.client().next_image();
    console.log(response.data)

    if (!nid) {
      window.snackbar.hideAll();
    } else {
      this.notification.hide(nid);
    }

    if (response.status !== 201) {
      this.notification.show({
        title: 'Active Learning',
        message: 'Failed to Fetch Next Sample',
        type: 'error',
        duration: 6000,
      });
    } else {
      this.uiModelService.show({
        content: NextSampleForm,
        contentProps: {
          info: response.data.meta,
        },
        shouldCloseOnEsc: true,
        title: 'Active Learning - Next Sample',
        customClassName: 'nextSampleForm',
      });
    }
  };

  onClickRecoverSeg = async () => {
    const nid = this.notification.show({
      title: 'MONAI Label',
      message: 'Recover Annotation from Most Recent Inference',
      type: 'info',
      duration: 2000,
    });

    const scalarDataRecover = window.ScalarDataBuffer;
    const volumeLoadObject = cache.getVolume('1');

    console.log(volumeLoadObject)

    if (volumeLoadObject) {
      const { scalarData } = volumeLoadObject;
      scalarData.set(scalarDataRecover);
      triggerEvent(eventTarget, Enums.Events.SEGMENTATION_DATA_MODIFIED, {
        segmentationId: '1',
      });
      console.debug('updated the segmentation\'s scalar data');
    } else {
      this.notification.show({
        title: 'MONAI Label',
        message: 'No latest inference to recover',
        type: 'error',
        duration: 6000,
      });
      return
    }
  };


  onClickNotifyServer = async () => { 
    const { displaySetService } = this.props.servicesManager.services;
    const activeDisplaySets = displaySetService.activeDisplaySets



    // iterate all display sets and check if SEG exists, get series IDs
    let segCount = 0;
    let latestLabelSeriesTimestamp = 0;
    let latestLabelSeriesInstanceUID = "";
    let latestImageSeriesInstanceUID = "";

    for (const item of activeDisplaySets){
      if (item.Modality === "SEG"){
        segCount++
        const curLabelTimestamp = `${item.SeriesDate}${item.instance.SeriesTime}`;
        if (curLabelTimestamp > latestLabelSeriesTimestamp) {
          latestLabelSeriesTimestamp = curLabelTimestamp;
          latestLabelSeriesInstanceUID = item.SeriesInstanceUID;
        }
      } else if (item.Modality === "CT"){ //TODO: if image modality is other than CT, add others
        latestImageSeriesInstanceUID = item.SeriesInstanceUID;
      }
    } 


    if (segCount === 0) {
      this.notification.show({
        title: 'MONAI Label',
        message: 'No segmentation added.',
        type: 'error',
        duration: 5000,
      });
      return;
    }
    const data = {
      added: [],
      updated: [],
      removed: []
    };

    data[segCount > 1 ? "updated" : "added"].push({
      "image": latestImageSeriesInstanceUID,
      "label": latestLabelSeriesInstanceUID
    });
    

    const response = await this.props.client().notify(data);

    console.log(response.data)
    if (response.status !== 201) {
      this.notification.show({
        title: 'MONAI Label',
        message: 'Failed. Response Error',
        type: 'error',
        duration: 5000,
      });
      return;
    }

    this.notification.show({
      title: 'MONAI Label',
      message: 'Notified Server  - Successful',
      type: 'success',
      duration: 2000,
    });

  };

  render() {
    return (
      <div className='tab'>
        <input
          className='tab-switch'
          type='checkbox'
          id={this.tabId}
          name='activelearning'
          value='activelearning'
          defaultChecked
        />
        <label className='tab-label' htmlFor={this.tabId}>
          Active Learning
        </label>
        <div className='tab-content'>
          <table style={{ fontSize: 'smaller', width: '100%' }}>
            <tbody>
            <tr>
              <td>
                <button
                  className='actionInput'
                  style={{ backgroundColor: 'lightgray', padding: '1px 1px', borderRadius: '2px', border: 'none' }}
                  onClick={this.onClickNextSample}
                >
                  Next Sample
                </button>
              </td>
            </tr>
            <tr>
              <td>
                <button
                  className='actionInput'
                  style={{ backgroundColor: 'lightgray', padding: '1px 1px', borderRadius: '2px', border: 'none'  }}
                  onClick={this.onClickNotifyServer}
                >
                  Notify Server
                </button>
              </td>
            </tr>
            </tbody>
          </table>
          <br />
          <u>
            <a href='#' style={{ color: 'lightyellow', fontSize: 'smaller' }} onClick={() => this.onClickRecoverSeg()}>Recover Seg</a>
          </u>
        </div>
      </div>
    );
  }
}

import cornerstoneTools from 'cornerstone-tools';

const { ProbeTool, getToolState } = cornerstoneTools;
const triggerEvent = cornerstoneTools.importInternal('util/triggerEvent');
const draw = cornerstoneTools.importInternal('drawing/draw');
const drawHandles = cornerstoneTools.importInternal('drawing/drawHandles');
const getNewContext = cornerstoneTools.importInternal('drawing/getNewContext');

export class DeepgrowProbeTool extends ProbeTool {
  constructor(props = {}) {
    const defaultProps = {
      name: 'DeepgrowProbe',
      supportedInteractionTypes: ['Mouse'],
      configuration: {
        drawHandles: true,
        handleRadius: 2,
        eventName: 'monailabel_deepgrow_probe_event',
        color: ['red', 'blue'],
      },
    };

    const initialProps = Object.assign(defaultProps, props);
    super(initialProps);
  }

  uuidv4() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
      let r = (Math.random() * 16) | 0,
        v = c === 'x' ? r : (r & 0x3) | 0x8;

      return v.toString(16);
    });
  }

  createNewMeasurement(eventData) {
    console.debug(eventData);
    let res = super.createNewMeasurement(eventData);
    if (res) {
      res.uuid = res.uuid || this.uuidv4();
      res.color = this.configuration.color[eventData.event.ctrlKey ? 1 : 0];
      res.ctrlKey = eventData.event.ctrlKey;
      res.imageId = eventData.image.imageId;
      res.x = eventData.currentPoints.image.x;
      res.y = eventData.currentPoints.image.y;

      console.info(
        'TRIGGERING DEEPGROW PROB EVENT: ' + this.configuration.eventName
      );
      console.info(res);
      triggerEvent(eventData.element, this.configuration.eventName, res);
    }
    return res;
  }

  renderToolData(evt) {
    const eventData = evt.detail;
    const { handleRadius } = this.configuration;

    const toolData = getToolState(evt.currentTarget, this.name);
    if (!toolData) {
      return;
    }

    const context = getNewContext(eventData.canvasContext.canvas);
    for (let i = 0; i < toolData.data.length; i++) {
      const data = toolData.data[i];
      if (data.imageId !== eventData.image.imageId) {
        continue;
      }
      if (data.visible === false) {
        continue;
      }

      draw(context, context => {
        const color = data.color;
        drawHandles(context, eventData, data.handles, {
          handleRadius,
          color,
        });
      });
    }
  }
}

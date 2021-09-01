import { DeepgrowProbeTool } from './index';
import csTools from 'cornerstone-tools';

/**
 *
 * @param {Object} servicesManager
 * @param {Object} configuration
 */
export default function init({ servicesManager, configuration }) {
  const {
    BrushTool,
    SphericalBrushTool,
    CorrectionScissorsTool,
    CircleScissorsTool,
    FreehandScissorsTool,
    RectangleScissorsTool,
  } = csTools;

  const tools = [
    DeepgrowProbeTool,
    BrushTool,
    SphericalBrushTool,
    CorrectionScissorsTool,
    CircleScissorsTool,
    FreehandScissorsTool,
    RectangleScissorsTool,
  ];

  tools.forEach(tool => csTools.addTool(tool));

  csTools.addTool(BrushTool, {
    name: 'BrushEraser',
    configuration: {
      alwaysEraseOnClick: true,
    },
  });
  csTools.addTool(CircleScissorsTool, {
    name: 'CircleScissorsEraser',
    defaultStrategy: 'ERASE_INSIDE',
  });
  csTools.addTool(FreehandScissorsTool, {
    name: 'FreehandScissorsEraser',
    defaultStrategy: 'ERASE_INSIDE',
  });
  csTools.addTool(RectangleScissorsTool, {
    name: 'RectangleScissorsEraser',
    defaultStrategy: 'ERASE_INSIDE',
  });
}

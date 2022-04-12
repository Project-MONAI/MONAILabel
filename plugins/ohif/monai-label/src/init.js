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

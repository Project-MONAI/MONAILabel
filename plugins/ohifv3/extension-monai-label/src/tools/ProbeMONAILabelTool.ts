import { Types, 
  metaData, 
  utilities as csUtils 
} from '@cornerstonejs/core';
import {
  ProbeTool,
} from '@cornerstonejs/tools';

export default class ProbeMONAILabelTool extends ProbeTool {
  static toolName = 'ProbeMONAILabel';
  
  constructor(
    toolProps = {},
    defaultToolProps = {
      configuration: {},
    }
  ) {
    super(toolProps, defaultToolProps);
  }

}


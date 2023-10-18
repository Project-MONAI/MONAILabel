const brushInstanceNames = {
  CircularBrush: 'CircularBrush',
  CircularEraser: 'CircularEraser',
  SphereBrush: 'SphereBrush',
  SphereEraser: 'SphereEraser',
  ThresholdCircularBrush: 'ThresholdCircularBrush',
  ThresholdSphereBrush: 'ThresholdSphereBrush',
};

const brushStrategies = {
  [brushInstanceNames.CircularBrush]: 'FILL_INSIDE_CIRCLE',
  [brushInstanceNames.CircularEraser]: 'ERASE_INSIDE_CIRCLE',
  [brushInstanceNames.SphereBrush]: 'FILL_INSIDE_SPHERE',
  [brushInstanceNames.SphereEraser]: 'ERASE_INSIDE_SPHERE',
  [brushInstanceNames.ThresholdCircularBrush]: 'THRESHOLD_INSIDE_CIRCLE',
  [brushInstanceNames.ThresholdSphereBrush]: 'THRESHOLD_INSIDE_SPHERE',
};

function initDefaultToolGroup(
  extensionManager,
  toolGroupService,
  commandsManager,
  toolGroupId
) {
  const utilityModule = extensionManager.getModuleEntry(
    '@ohif/extension-cornerstone.utilityModule.tools'
  );

  const { toolNames, Enums } = utilityModule.exports;

  const tools = {
    active: [
      {
        toolName: toolNames.WindowLevel,
        bindings: [{ mouseButton: Enums.MouseBindings.Primary }],
      },
      {
        toolName: toolNames.Pan,
        bindings: [{ mouseButton: Enums.MouseBindings.Auxiliary }],
      },
      {
        toolName: toolNames.Zoom,
        bindings: [{ mouseButton: Enums.MouseBindings.Secondary }],
      },
      { toolName: toolNames.StackScrollMouseWheel, bindings: [] },
    ],
    passive: [
      { toolName: toolNames.CircleScissors },
      { toolName: toolNames.RectangleScissors },
      { toolName: toolNames.SphereScissors },
      {
        toolName: brushInstanceNames.CircularBrush,
        parentTool: 'Brush',
        configuration: {
          activeStrategy: brushStrategies.CircularBrush,
        },
      },
      {
        toolName: brushInstanceNames.CircularEraser,
        parentTool: 'Brush',
        configuration: {
          activeStrategy: brushStrategies.CircularEraser,
        },
      },
      {
        toolName: brushInstanceNames.SphereEraser,
        parentTool: 'Brush',
        configuration: {
          activeStrategy: brushStrategies.SphereEraser,
        },
      },
      {
        toolName: brushInstanceNames.SphereBrush,
        parentTool: 'Brush',
        configuration: {
          activeStrategy: brushStrategies.SphereBrush,
        },
      },
      {
        toolName: brushInstanceNames.ThresholdCircularBrush,
        parentTool: 'Brush',
        configuration: {
          activeStrategy: brushStrategies.ThresholdCircularBrush,
        },
      },
      {
        toolName: brushInstanceNames.ThresholdSphereBrush,
        parentTool: 'Brush',
        configuration: {
          activeStrategy: brushStrategies.ThresholdSphereBrush,
        },
      },
      { toolName: toolNames.StackScroll },
      { toolName: toolNames.Magnify },
      { toolName: toolNames.SegmentationDisplay },
      { toolName: 'ProbeMONAILabel' },
    ],
    // enabled
    // disabled
    disabled: [{ toolName: toolNames.ReferenceLines }],
  };

  toolGroupService.createToolGroupAndAddTools(toolGroupId, tools);
}

function initMPRToolGroup(extensionManager, toolGroupService, commandsManager) {
  const utilityModule = extensionManager.getModuleEntry(
    '@ohif/extension-cornerstone.utilityModule.tools'
  );

  const { toolNames, Enums } = utilityModule.exports;

  const tools = {
    active: [
      {
        toolName: toolNames.WindowLevel,
        bindings: [{ mouseButton: Enums.MouseBindings.Primary }],
      },
      {
        toolName: toolNames.Pan,
        bindings: [{ mouseButton: Enums.MouseBindings.Auxiliary }],
      },
      {
        toolName: toolNames.Zoom,
        bindings: [{ mouseButton: Enums.MouseBindings.Secondary }],
      },
      { toolName: toolNames.StackScrollMouseWheel, bindings: [] },
    ],
    passive: [
      { toolName: toolNames.CircleScissors },
      { toolName: toolNames.RectangleScissors },
      { toolName: toolNames.SphereScissors },
      {
        toolName: brushInstanceNames.CircularBrush,
        parentTool: 'Brush',
        configuration: {
          activeStrategy: brushStrategies.CircularBrush,
        },
      },
      {
        toolName: brushInstanceNames.CircularEraser,
        parentTool: 'Brush',
        configuration: {
          activeStrategy: brushStrategies.CircularEraser,
        },
      },
      {
        toolName: brushInstanceNames.SphereEraser,
        parentTool: 'Brush',
        configuration: {
          activeStrategy: brushStrategies.SphereEraser,
        },
      },
      {
        toolName: brushInstanceNames.SphereBrush,
        parentTool: 'Brush',
        configuration: {
          activeStrategy: brushStrategies.SphereBrush,
        },
      },
      {
        toolName: brushInstanceNames.ThresholdCircularBrush,
        parentTool: 'Brush',
        configuration: {
          activeStrategy: brushStrategies.ThresholdCircularBrush,
        },
      },
      {
        toolName: brushInstanceNames.ThresholdSphereBrush,
        parentTool: 'Brush',
        configuration: {
          activeStrategy: brushStrategies.ThresholdSphereBrush,
        },
      },
      { toolName: toolNames.SegmentationDisplay },
      { toolName: 'ProbeMONAILabel' },
      { toolName: 'ProbeMONAILabel' },
    ],
    disabled: [
      {
        toolName: toolNames.Crosshairs,
        configuration: {
          viewportIndicators: false,
          autoPan: {
            enabled: false,
            panSize: 10,
          },
        },
      },
      { toolName: toolNames.ReferenceLines },
    ],
    // enabled
    // disabled
  };

  toolGroupService.createToolGroupAndAddTools('mpr', tools);
}

function initToolGroups(extensionManager, toolGroupService, commandsManager) {
  initDefaultToolGroup(
    extensionManager,
    toolGroupService,
    commandsManager,
    'default'
  );
  initMPRToolGroup(extensionManager, toolGroupService, commandsManager);
}

export default initToolGroups;

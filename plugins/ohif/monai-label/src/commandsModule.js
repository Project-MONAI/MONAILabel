const commandsModule = ({ servicesManager }) => {
  const { UINotificationService } = servicesManager.services;

  const MONAILabelMessage = (message, type = 'success', debug = true) => {
    if (debug) {
      console.debug('MONAI Label - ' + message);
    }
    if (UINotificationService) {
      UINotificationService.show({
        title: 'MONAI Label',
        message: message,
        type: type,
      });
    }
  };

  const actions = {
    segmentation: ({ model_name }) => {
      MONAILabelMessage('Running segmentation API with ' + model_name);
    },
    deepgrow: ({ model_name }) => {
      MONAILabelMessage('Running deepgrow API with ' + model_name);
    },
  };

  const definitions = {
    segmentation: {
      commandFn: actions.segmentation,
      storeContexts: ['viewports'],
      options: {},
    },
    deepgrow: {
      commandFn: actions.deepgrow,
      storeContexts: ['viewports'],
      options: {},
    },
  };

  return {
    definitions,
    defaultContext: 'ACTIVE_VIEWPORT::CORNERSTONE',
  };
};

export default commandsModule;

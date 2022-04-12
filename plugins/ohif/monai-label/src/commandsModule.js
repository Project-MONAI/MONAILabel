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

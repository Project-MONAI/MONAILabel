import { ServicesManager, CommandsManager, ExtensionManager } from '@ohif/core';
import {
  Enums,
} from '@cornerstonejs/tools';

export default function getCommandsModule({
  servicesManager,
  commandsManager,
  extensionManager,
}: {
  servicesManager: ServicesManager;
  commandsManager: CommandsManager;
  extensionManager: ExtensionManager;
}) {
  const {
    viewportGridService,
    toolGroupService,
    cineService,
    toolbarService,
    uiNotificationService,
  } = servicesManager.services;

  const actions = {
    setToolActive: ({ toolName }) => {

        uiNotificationService.show({
          title: 'MONAI Label probe',
          message:
            'MONAI Label Probe Activated.',
          type: 'info',
          duration: 3000,
        });
    },
  };

  const definitions = {
    /* setToolActive: {
      commandFn: actions.setToolActive,
    }, */
  };

  return {
    actions,
    definitions,
    defaultContext: 'MONAILabel',
  };
}

import { id } from './id';
import getPanelModule from './getPanelModule';

export default {
  id,

  preRegistration: ({
    servicesManager,
    commandsManager,
    configuration = {},
  }) => {}, 

  getPanelModule,

  getViewportModule: ({
    servicesManager,
    commandsManager,
    extensionManager,
  }) => {},

  getToolbarModule: ({
    servicesManager,
    commandsManager,
    extensionManager,
  }) => {},

  getLayoutTemplateModule: ({
    servicesManager,
    commandsManager,
    extensionManager,
  }) => {},

  getSopClassHandlerModule: ({
    servicesManager,
    commandsManager,
    extensionManager,
  }) => {},

  getHangingProtocolModule: ({
    servicesManager,
    commandsManager,
    extensionManager,
  }) => {},

  getCommandsModule: ({
    servicesManager,
    commandsManager,
    extensionManager,
  }) => {},

  getContextModule: ({
    servicesManager,
    commandsManager,
    extensionManager,
  }) => {},

  getDataSourcesModule: ({
    servicesManager,
    commandsManager,
    extensionManager,
  }) => {},
};

import { id } from './id';
import getPanelModule from './getPanelModule';
import getCommandsModule from './getCommandsModule';
import preRegistration from './init';

export default {
  id,

  preRegistration,

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

  getCommandsModule,

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

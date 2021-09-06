import toolbarModule from './toolbarModule';
import panelModule from './panelModule.js';
import init from './init';
import { DeepgrowProbeTool } from './tools/DeepgrowProbeTool';

export { DeepgrowProbeTool };

export default {
  id: 'com.ohif.monai-label',

  preRegistration({ servicesManager, configuration = {} }) {
    init({ servicesManager, configuration });
  },
  getToolbarModule({ servicesManager }) {
    return toolbarModule;
  },
  getPanelModule({ servicesManager, commandsManager }) {
    return panelModule({ servicesManager, commandsManager });
  },
};

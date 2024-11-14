import React from 'react';
import MonaiLabelPanel from './components/MonaiLabelPanel';
import { ServicesManager, CommandsManager, ExtensionManager } from '@ohif/core';

function getPanelModule({
  servicesManager,
  commandsManager,
  extensionManager,
}: {
  servicesManager: ServicesManager;
  commandsManager: CommandsManager;
  extensionManager: ExtensionManager;
}) {
  const { uiNotificationService, viewportGridService, displaySetService } =
    servicesManager.services;

  const WrappedMonaiLabelPanel = () => {
    return (
      <MonaiLabelPanel
        commandsManager={commandsManager}
        servicesManager={servicesManager}
        extensionManager={extensionManager}
      />
    );
  };

  return [
    {
      name: 'monailabel',
      iconName: 'tab-patient-info',
      iconLabel: 'MONAI',
      label: 'MONAI Label',
      secondaryLabel: 'MONAI Label',
      component: WrappedMonaiLabelPanel,
    },
  ];
}

export default getPanelModule;

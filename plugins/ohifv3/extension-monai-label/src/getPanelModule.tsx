import React from 'react';
import MonaiLabelPanel from './components/MonaiLabelPanel';


function getPanelModule({
  commandsManager,
  extensionManager,
  servicesManager,
}) {

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
      iconName: 'tab-segmentation',
      iconLabel: 'MONAI',
      label: 'MONAI Label',
      secondaryLabel: 'MONAI Label',
      component: WrappedMonaiLabelPanel,
    },
  ];
}

export default getPanelModule;



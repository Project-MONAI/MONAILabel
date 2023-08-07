import React from 'react';
import MonaiLabelPanel from './components/MonaiLabelPanel';


/* function getPanelModule({
  commandsManager,
  extensionManager,
  servicesManager,
}) {

  const WrappedMonaiLabelPanel = () => {
    return (
      <MonaiLabelPanel/>
    );
  };

  return [
    {
      name: 'monailabel',
      iconName: 'tab-linear',
      iconLabel: 'MONAI',
      label: 'MONAI Label',
      secondaryLabel: 'MONAI Label',
      component: WrappedMonaiLabelPanel,
    },
  ];
}

export default getPanelModule; */

const WrappedMonaiLabelPanel = () => {
  return [
    {
      name: 'monailabel',
      // Select icon from this list: 
      // https://github.com/OHIF/Viewers/blob/58d38495f097afc6333937b6fbaf60ae473957c0/platform/ui/src/components/Icon/getIcon.js#L279
      iconName: 'tab-segmentation', 
      iconLabel: 'MONAI',
      label: 'MONAI Label',
      secondaryLabel: 'MONAI Label',
      component: MonaiLabelPanel,
    },
  ];
};

export default WrappedMonaiLabelPanel



import React from 'react';
import { LegacyButton, ButtonGroup } from '@ohif/ui';

// TODO:
// - No loading UI exists yet
// - cancel promises when component is destroyed
// - show errors in UI for thumbnails if promise fails

function getPanelModule({
  commandsManager,
  extensionManager,
  servicesManager,
}) {
  const MonaiLabelPanel = () => {
    return (
      <React.Fragment>
      <ButtonGroup color="black" size="inherit">
        {/* TODO Revisit design of ButtonGroup later - for now use LegacyButton for its children.*/}
        <LegacyButton className="px-2 py-2 text-base">
          {'Hello MONAI Label'}
        </LegacyButton>
      </ButtonGroup>
    </React.Fragment>
    );
  };

  return [
    {
      name: 'monailabel',
      iconName: 'tab-linear',
      iconLabel: 'MONAI',
      label: 'MONAI Label',
      secondaryLabel: 'MONAI Label',
      component: MonaiLabelPanel,
    },
  ];
}

export default getPanelModule;

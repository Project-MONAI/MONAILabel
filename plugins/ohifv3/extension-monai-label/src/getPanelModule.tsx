import React from 'react';
import { WrappedPanelStudyBrowser, PanelMeasurementTable } from './Panels';
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
  const wrappedMeasurementPanel = () => {
    return (
      <PanelMeasurementTable
        commandsManager={commandsManager}
        servicesManager={servicesManager}
        extensionManager={extensionManager}
      />
    );
  };

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
      name: 'seriesList',
      iconName: 'group-layers',
      iconLabel: 'Studies',
      label: 'Studies',
      component: WrappedPanelStudyBrowser.bind(null, {
        commandsManager,
        extensionManager,
        servicesManager,
      }),
    },
    {
      name: 'measure',
      iconName: 'tab-linear',
      iconLabel: 'Measure',
      label: 'Measurements',
      secondaryLabel: 'Measurements',
      component: wrappedMeasurementPanel,
    },
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

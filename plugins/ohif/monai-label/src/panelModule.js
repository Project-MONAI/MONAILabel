import MonaiLabelPanel from './components/MonaiLabelPanel.js';

const panelModule = ({ commandsManager }) => {
  return {
    menuOptions: [
      {
        icon: 'list',
        label: 'MONAI Label',
        from: 'right',
        target: 'monai-label-panel',
      },
    ],
    components: [
      {
        id: 'monai-label-panel',
        component: MonaiLabelPanel,
      },
    ],
    defaultContext: ['VIEWER'],
  };
};

export default panelModule;

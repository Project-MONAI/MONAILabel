import React from 'react';
import { Dialog, ButtonEnums } from '@ohif/ui';
import OptionsForm from './actions/OptionsForm';

function optionsInputDialog(uiDialogService, config, info, callback) {
  const dialogId = 'monai-label-options';
  const optionsRef = React.createRef<OptionsForm>();

  const onSubmitHandler = ({ action }) => {
    switch (action.id) {
      case 'save':
        callback(optionsRef.current.state.config, action.id);
        uiDialogService.dismiss({ id: dialogId });
        break;
      case 'cancel':
        callback({}, action.id);
        uiDialogService.dismiss({ id: dialogId });
        break;
      case 'reset':
        optionsRef.current.onReset();
        break;
    }
  };

  uiDialogService.create({
    id: dialogId,
    centralize: true,
    isDraggable: false,
    showOverlay: true,
    content: Dialog,
    contentProps: {
      title: 'Options / Configurations',
      noCloseButton: true,
      onClose: () => uiDialogService.dismiss({ id: dialogId }),
      actions: [
        { id: 'reset', text: 'Reset', type: ButtonEnums.type.secondary },
        { id: 'cancel', text: 'Cancel', type: ButtonEnums.type.secondary },
        { id: 'save', text: 'Confirm', type: ButtonEnums.type.primary },
      ],
      onSubmit: onSubmitHandler,
      body: () => {
        return <OptionsForm ref={optionsRef} config={config} info={info} />;
      },
    },
  });
}

export default optionsInputDialog;

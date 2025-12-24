/*
Copyright (c) MONAI Consortium
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

import React from 'react';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter, Button } from '@ohif/ui-next';
import OptionsForm from './actions/OptionsForm';

function OptionsInputDialogComponent({ config, info, callback, hide }) {
  const optionsRef = React.useRef<OptionsForm>(null);

  const handleSave = () => {
    if (optionsRef.current) {
      callback(optionsRef.current.state.config, 'save');
    }
    hide();
  };

  const handleCancel = () => {
    callback({}, 'cancel');
    hide();
  };

  const handleReset = () => {
    if (optionsRef.current) {
      optionsRef.current.onReset();
    }
  };

  return (
    <Dialog open={true} onOpenChange={(open) => !open && handleCancel()}>
      <DialogContent className="max-w-4xl">
        <DialogHeader>
          <DialogTitle>Options / Configurations</DialogTitle>
        </DialogHeader>

        <div className="py-4">
          <OptionsForm ref={optionsRef} config={config} info={info} />
        </div>

        <DialogFooter className="flex gap-2">
          <Button variant="outline" onClick={handleReset}>
            Reset
          </Button>
          <Button variant="outline" onClick={handleCancel}>
            Cancel
          </Button>
          <Button onClick={handleSave}>
            Confirm
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

function optionsInputDialog(uiDialogService, config, info, callback) {
  const dialogId = 'monai-label-options';

  uiDialogService.show({
    id: dialogId,
    title: 'Options / Configurations',
    content: OptionsInputDialogComponent,
    shouldCloseOnEsc: true,
    contentProps: {
      config,
      info,
      callback,
    },
  });
}

export default optionsInputDialog;

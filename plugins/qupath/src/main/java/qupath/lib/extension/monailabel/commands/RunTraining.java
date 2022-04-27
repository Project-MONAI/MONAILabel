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

package qupath.lib.extension.monailabel.commands;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import qupath.lib.extension.monailabel.MonaiLabelClient;
import qupath.lib.extension.monailabel.MonaiLabelClient.ResponseInfo;
import qupath.lib.gui.dialogs.Dialogs;
import qupath.lib.plugins.parameters.ParameterList;

public class RunTraining implements Runnable {
	private final static Logger logger = LoggerFactory.getLogger(RunTraining.class);

	@Override
	public void run() {
		try {
			ResponseInfo info = MonaiLabelClient.info();
			List<String> names = new ArrayList<String>();
			Map<String, String[]> labels = new HashMap<String, String[]>();
			for (String n : info.models.keySet()) {
				names.add(n);
				labels.put(n, info.models.get(n).labels.labels());
			}

			ParameterList list = new ParameterList();
			list.addChoiceParameter("Model", "Model Name", names.isEmpty() ? "" : names.get(0), names);
			list.addIntParameter("max_epochs", "max_epochs", 10);
			list.addIntParameter("train_batch_size", "train_batch_size", 1);
			list.addIntParameter("val_batch_size", "val_batch_size", 1);

			if (Dialogs.showParameterDialog("MONAILabel", list)) {
				String model = (String) list.getChoiceParameterValue("Model");
				int max_epochs = list.getIntParameterValue("max_epochs").intValue();
				int train_batch_size = list.getIntParameterValue("train_batch_size").intValue();
				int val_batch_size = list.getIntParameterValue("val_batch_size").intValue();

				String params = "{" + "\"max_epochs\":" + max_epochs + "," + "\"train_batch_size\":" + train_batch_size
						+ "," + "\"val_batch_size\":" + val_batch_size + "}";
				logger.info("TRAINING:: model = " + model);
				logger.info("TRAINING:: max_epochs = " + max_epochs);

				String res = MonaiLabelClient.train(model, params);
				logger.info("TRAINING:: resp = " + res);
				Dialogs.showInfoNotification("MONALabel", "Training job started...");
			}
		} catch (Exception ex) {
			ex.printStackTrace();
			Dialogs.showErrorMessage("MONAILabel", ex);
		}
	}
}
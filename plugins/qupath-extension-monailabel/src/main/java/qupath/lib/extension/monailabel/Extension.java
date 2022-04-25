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

package qupath.lib.extension.monailabel;

import org.controlsfx.control.action.ActionUtils;

import qupath.lib.common.Version;
import qupath.lib.extension.monailabel.commands.RunInference;
import qupath.lib.extension.monailabel.commands.RunTraining;
import qupath.lib.extension.monailabel.commands.SubmitLabel;
import qupath.lib.gui.ActionTools;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.gui.extensions.QuPathExtension;
import qupath.lib.gui.tools.MenuTools;

public class Extension implements QuPathExtension {

	@Override
	public void installExtension(QuPathGUI qupath) {

		var runInfer = ActionTools.createAction(new RunInference(qupath), "Run...");
		runInfer.disabledProperty().bind(qupath.imageDataProperty().isNull());
		MenuTools.addMenuItems(qupath.getMenu("MONAI Label", true), runInfer);

		var submit = ActionTools.createAction(new SubmitLabel(qupath), "Submit Label");
		submit.disabledProperty().bind(qupath.imageDataProperty().isNull());
		MenuTools.addMenuItems(qupath.getMenu("MONAI Label", true), submit);

		MenuTools.addMenuItems(qupath.getMenu("MONAI Label", true), ActionUtils.ACTION_SEPARATOR);

		var training = ActionTools.createAction(new RunTraining(qupath), "Training...");
		MenuTools.addMenuItems(qupath.getMenu("MONAI Label", true), training);

		new Settings().addProperties(qupath);
	}

	@Override
	public String getName() {
		return "MONAILabel extension";
	}

	@Override
	public String getDescription() {
		return "MONAILabel - Active Learning Solution";
	}

	@Override
	public Version getQuPathVersion() {
		return getVersion();
	}

}
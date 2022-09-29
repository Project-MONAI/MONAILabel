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

import java.net.URL;

import org.controlsfx.control.action.ActionUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javafx.geometry.Orientation;
import javafx.scene.control.Button;
import javafx.scene.control.ContextMenu;
import javafx.scene.control.Separator;
import javafx.scene.control.Tooltip;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.input.KeyCombination;
import qupath.lib.common.Version;
import qupath.lib.extension.monailabel.commands.NextSample;
import qupath.lib.extension.monailabel.commands.RunInference;
import qupath.lib.extension.monailabel.commands.RunTraining;
import qupath.lib.extension.monailabel.commands.SubmitLabel;
import qupath.lib.gui.ActionTools;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.gui.extensions.QuPathExtension;
import qupath.lib.gui.tools.MenuTools;

public class Extension implements QuPathExtension {
	final private static Logger logger = LoggerFactory.getLogger(Extension.class);

	@Override
	public void installExtension(QuPathGUI qupath) {

		var activeLearning = ActionTools.createAction(new NextSample(qupath), "Next Sample/Patch...");
		activeLearning.setAccelerator(KeyCombination.keyCombination("ctrl+n"));
		activeLearning.disabledProperty().bind(qupath.imageDataProperty().isNull());
		MenuTools.addMenuItems(qupath.getMenu("MONAI Label", true), activeLearning);

		MenuTools.addMenuItems(qupath.getMenu("MONAI Label", true), ActionUtils.ACTION_SEPARATOR);

		var runInfer = ActionTools.createAction(new RunInference(qupath), "Annotations...");
		runInfer.setAccelerator(KeyCombination.keyCombination("ctrl+m"));
		runInfer.disabledProperty().bind(qupath.imageDataProperty().isNull());
		MenuTools.addMenuItems(qupath.getMenu("MONAI Label", true), runInfer);

		var submit = ActionTools.createAction(new SubmitLabel(qupath), "Submit Label");
		submit.setAccelerator(KeyCombination.keyCombination("alt+m"));
		submit.disabledProperty().bind(qupath.imageDataProperty().isNull());
		MenuTools.addMenuItems(qupath.getMenu("MONAI Label", true), submit);

		MenuTools.addMenuItems(qupath.getMenu("MONAI Label", true), ActionUtils.ACTION_SEPARATOR);

		var training = ActionTools.createAction(new RunTraining(), "Training...");
		training.setAccelerator(KeyCombination.keyCombination("ctrl+t"));
		MenuTools.addMenuItems(qupath.getMenu("MONAI Label", true), training);

		new Settings().addProperties(qupath);

		// Add buttons to toolbar
		var toolbar = qupath.getToolBar();
		toolbar.getItems().add(new Separator(Orientation.VERTICAL));
		try {
			ImageView imageView = new ImageView(
					getMonaiLabelIcon(QuPathGUI.TOOLBAR_ICON_SIZE, QuPathGUI.TOOLBAR_ICON_SIZE));

			Button btnAnnotation = new Button();
			btnAnnotation.setGraphic(imageView);
			btnAnnotation.setTooltip(new Tooltip("MONAILabel Annotation"));

			ContextMenu popup = new ContextMenu();
			popup.getItems().addAll(ActionTools.createMenuItem(runInfer), ActionTools.createMenuItem(submit));
			btnAnnotation.setOnMouseClicked(e -> {
				popup.show(btnAnnotation, e.getScreenX(), e.getScreenY());
			});

			toolbar.getItems().add(btnAnnotation);
		} catch (Exception e) {
			logger.error("Error adding toolbar buttons", e);
		}
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

	public static Image getMonaiLabelIcon(final int width, final int height) {
		try {
			URL url = Extension.class.getClassLoader().getResource("MONAI-Label.png");
			return new Image(url.toString(), width, height, true, true);
		} catch (Exception e) {
			logger.error("Unable to load ImageJ icon!", e);
		}
		return null;
	}
}
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

import javafx.application.Platform;
import javafx.beans.property.BooleanProperty;
import javafx.beans.property.StringProperty;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.gui.prefs.PathPrefs;

public class Settings {
	private static StringProperty serverURL = PathPrefs.createPersistentPreference("serverURL",
			"http://127.0.0.1:8000");

	public static StringProperty serverURLProperty() {
		return serverURL;
	}

	private static BooleanProperty wsi = PathPrefs.createPersistentPreference("wsi", Boolean.FALSE);

	public static BooleanProperty wsiProperty() {
		return wsi;
	}

	void addProperties(QuPathGUI qupath) {
		if (!Platform.isFxApplicationThread()) {
			Platform.runLater(() -> addProperties(qupath));
			return;
		}

		qupath.getPreferencePane().addPropertyPreference(Settings.serverURLProperty(), String.class, "Server URL",
				"MONAI Label", "Set MONAI Label Server URL (default: http://127.0.0.1:8000)");
		qupath.getPreferencePane().addPropertyPreference(Settings.wsiProperty(), Boolean.class, "Enable WSI",
				"MONAI Label", "Allow WSI Inference when ROI is not selected");

	}
}

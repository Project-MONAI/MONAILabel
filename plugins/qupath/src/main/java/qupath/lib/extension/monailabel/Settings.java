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

import java.io.File;

import javafx.application.Platform;
import javafx.beans.property.IntegerProperty;
import javafx.beans.property.StringProperty;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.gui.prefs.PathPrefs;

public class Settings {
	private static StringProperty serverURL = PathPrefs.createPersistentPreference("serverURL",
			"http://127.0.0.1:8000");
	private static StringProperty localStoragePath = PathPrefs.createPersistentPreference("localStoragePath",
			System.getProperty("user.home") + File.separator + "QuPath" + File.separator + "monailabel");
	private static IntegerProperty maxWorkers = PathPrefs.createPersistentPreference("max_workers", 1);

	public static StringProperty serverURLProperty() {
		return serverURL;
	}

	public static StringProperty localStoragePathProperty() {
		return localStoragePath;
	}

	public static IntegerProperty maxWorkersProperty() {
		return maxWorkers;
	}

	void addProperties(QuPathGUI qupath) {
		if (!Platform.isFxApplicationThread()) {
			Platform.runLater(() -> addProperties(qupath));
			return;
		}

		qupath.getPreferencePane().addPropertyPreference(Settings.serverURLProperty(), String.class, "Server URL",
				"MONAI Label", "Set MONAI Label Server URL (default: http://127.0.0.1:8000)");
		qupath.getPreferencePane().addPropertyPreference(Settings.localStoragePathProperty(), String.class,
				"Local Storage Path", "MONAI Label", "Local Storage Path for downloaded images/samples");
		qupath.getPreferencePane().addPropertyPreference(Settings.maxWorkersProperty(), Integer.class, "Max Workers",
				"MONAI Label", "Max Workers (WSI Inference)");

	}
}

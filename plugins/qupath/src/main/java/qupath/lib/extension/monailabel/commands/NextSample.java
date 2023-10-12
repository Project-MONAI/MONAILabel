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

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import qupath.lib.common.GeneralTools;
import qupath.lib.extension.monailabel.MonaiLabelClient;
import qupath.lib.extension.monailabel.MonaiLabelClient.NextSampleInfo;
import qupath.lib.extension.monailabel.MonaiLabelClient.ResponseInfo;
import qupath.lib.extension.monailabel.Settings;
import qupath.lib.extension.monailabel.Utils;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.gui.dialogs.Dialogs;
import qupath.lib.objects.PathObjects;
import qupath.lib.plugins.parameters.ParameterList;
import qupath.lib.regions.ImagePlane;
import qupath.lib.roi.ROIs;
import qupath.lib.roi.interfaces.ROI;

public class NextSample implements Runnable {
	private final static Logger logger = LoggerFactory.getLogger(NextSample.class);

	private QuPathGUI qupath;
	private static String selectedStrategy;
	private static int[] selectedPatchSize = { 1024, 1024 };

	public NextSample(QuPathGUI qupath) {
		this.qupath = qupath;
	}

	@Override
	public void run() {
		try {
			var viewer = qupath.getViewer();
			var imageData = viewer.getImageData();
			String image = "";
			if (imageData != null) {
				var uris = imageData.getServer().getURIs();
				File imageFile = GeneralTools.toPath(uris.iterator().next()).toFile();
				image = GeneralTools.getNameWithoutExtension(imageFile);
			}

			ResponseInfo info = MonaiLabelClient.info();
			List<String> names = new ArrayList<String>();
			for (String n : info.strategies.keySet()) {
				names.add(n);
			}

			if (selectedStrategy == null || selectedStrategy.isEmpty()) {
				selectedStrategy = names.isEmpty() ? "" : names.get(0);
			}

			boolean imageLoaded = qupath.imageDataProperty() == null || qupath.imageDataProperty().isNull().get()
					? false
					: true;
			boolean nextPatch = imageLoaded;
			int[] patchSize = { 1024, 1024 };
			int[] imageSize = { 0, 0 };

			ParameterList list = new ParameterList();
			list.addChoiceParameter("Strategy", "Active Learning Strategy", selectedStrategy, names);
			if (nextPatch) {
				list.addBooleanParameter("NextPatch", "Next Patch (from current Image)", nextPatch);
				list.addStringParameter("PatchSize", "PatchSize", Arrays.toString(selectedPatchSize));
			}

			if (Dialogs.showParameterDialog("MONAILabel", list)) {
				String strategy = (String) list.getChoiceParameterValue("Strategy");
				if (nextPatch) {
					nextPatch = list.getBooleanParameterValue("NextPatch").booleanValue();
					patchSize = Utils.parseStringArray(list.getStringParameterValue("PatchSize"));
					imageSize = new int[] { imageData.getServer().getWidth(), imageData.getServer().getHeight() };

					logger.info(String.join(",", imageData.getProperties().keySet()));
				}

				selectedStrategy = strategy;
				selectedPatchSize = patchSize;

				if (!nextPatch) {
					image = "";
				}

				StringBuilder sb = new StringBuilder();
				sb.append("{");
				sb.append("\"image\": \"" + image + "\", ");
				sb.append("\"patch_size\": " + Arrays.toString(patchSize) + ",");
				sb.append("\"image_size\": " + Arrays.toString(imageSize));
				sb.append("}");

				NextSampleInfo sample = MonaiLabelClient.nextSample(strategy, sb.toString());
				logger.info("MONAILabel:: Active Learning => " + sample.id);
				if (nextPatch) {
					logger.info("MONAILabel:: New Patch => " + Arrays.toString(sample.bbox));
					ImagePlane plane = ImagePlane.getPlane(0, 0);
					ROI roi = ROIs.createRectangleROI(sample.bbox[0], sample.bbox[1], sample.bbox[2], sample.bbox[3],
							plane);

					var obj = PathObjects.createAnnotationObject(roi);
					imageData.getHierarchy().addPathObject(obj);
					imageData.getHierarchy().getSelectionModel().setSelectedObject(obj);
				} else {
					String message = "This will close the current image without saving.\nAre you sure to continue?";
					if (!imageLoaded || Dialogs.showConfirmDialog("MONAILabel", message)) {
						File f = new File(sample.path);
						if (f.isFile() && f.exists()) {
							qupath.openImage(f.getAbsolutePath(), false, false);
						} else {
							File path = new File(Settings.localStoragePathProperty().get());
							if (!path.exists()) {
								path.mkdirs();
							}
							File o = new File(path.getPath() + File.separator + f.getName());
							if (!o.exists()) {
								logger.info("Image (" + sample.id + ") exists;  Will try to download it from Server");
								// Dialogs.showInfoNotification("MONALabel", "Downloading Sample. Please
								// wait...");
								MonaiLabelClient.downloadImage(sample.id, o);
							}
							qupath.openImage(o.getAbsolutePath(), false, false);
						}
					}
				}
			}

			Dialogs.showInfoNotification("MONALabel", "Active Learning");
		} catch (Exception ex) {
			ex.printStackTrace();
			Dialogs.showErrorMessage("MONAILabel", ex);
		}
	}
}

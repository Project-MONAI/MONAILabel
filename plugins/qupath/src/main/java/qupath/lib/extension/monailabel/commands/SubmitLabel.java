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

import java.awt.image.BufferedImage;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;
import javax.xml.transform.OutputKeys;
import javax.xml.transform.Transformer;
import javax.xml.transform.TransformerException;
import javax.xml.transform.TransformerFactory;
import javax.xml.transform.dom.DOMSource;
import javax.xml.transform.stream.StreamResult;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.w3c.dom.Document;
import org.w3c.dom.Element;

import qupath.lib.extension.monailabel.MonaiLabelClient;
import qupath.lib.extension.monailabel.MonaiLabelClient.ImageInfo;
import qupath.lib.extension.monailabel.Utils;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.gui.dialogs.Dialogs;
import qupath.lib.images.ImageData;
import qupath.lib.images.writers.ImageWriterTools;
import qupath.lib.objects.PathObject;
import qupath.lib.plugins.parameters.ParameterList;
import qupath.lib.regions.RegionRequest;
import qupath.lib.roi.ROIs;
import qupath.lib.roi.interfaces.ROI;

public class SubmitLabel implements Runnable {
	private final static Logger logger = LoggerFactory.getLogger(SubmitLabel.class);

	private QuPathGUI qupath;

	public SubmitLabel(QuPathGUI qupath) {
		this.qupath = qupath;
	}

	@Override
	public void run() {
		Path annotationXML = null;
		Path imagePatch = null;
		try {
			var viewer = qupath.getViewer();
			var imageData = viewer.getImageData();
			String image = Utils.getNameWithoutExtension(imageData.getServerPath());

			if (!MonaiLabelClient.imageExists(image)) {
				logger.info("Image does not exist on Server.");
				var selected = imageData.getHierarchy().getSelectionModel().getSelectedObject();
				var roi = selected != null ? selected.getROI() : null;
				int[] bbox = Utils.getBBOX(roi);
				if (bbox[2] <= 0 && bbox[3] <= 0) {
					Dialogs.showWarningNotification("MONAILabel",
							"Please select the Annotation ROI/Rectangle for submission");
					return;
				}

				String patchName = image + String.format("-patch-%d_%d_%d_%d", bbox[0], bbox[1], bbox[2], bbox[3]);
				ParameterList list = new ParameterList();
				list.addStringParameter("Location", "Patch (x,y,w,h)", Arrays.toString(bbox));
				list.addStringParameter("Patch", "Patch Name", patchName);

				if (Dialogs.showParameterDialog("MONAILabel - Save Patch + Label", list)) {
					bbox = Utils.parseStringArray(list.getStringParameterValue("Location"));
					patchName = list.getStringParameterValue("Patch");
					if (Dialogs.showYesNoDialog("MONAILabel",
							"This will upload BOTH image patch + annotation to MONAI Label Server.\n\n"
									+ "Do you want to continue?")) {

						annotationXML = getAnnotationsXml(image, imageData, bbox);
						logger.info("MONAILabel:: Annotations XML: " + annotationXML);

						imagePatch = java.nio.file.Files.createTempFile("patch", ".png");
						var requestROI = RegionRequest.createInstance(imageData.getServer().getPath(), 1, roi);
						ImageWriterTools.writeImageRegion(imageData.getServer(), requestROI, imagePatch.toString());

						ImageInfo imageInfo = MonaiLabelClient.saveImage(patchName, imagePatch.toFile(), "{}");
						logger.info("MONAILabel:: New Image ID => " + imageInfo.image);
						Dialogs.showInfoNotification("MONALabel", "Image Patch uploaded to MONAILabel Server");

						MonaiLabelClient.saveLabel(imageInfo.image, annotationXML.toFile(), null, "{}");
						Dialogs.showInfoNotification("MONALabel", "Label/Annotations saved in Server");
					}
				}
			} else {
				if (Dialogs.showYesNoDialog("MONAILabel",
						"Do you like to submit this annotation to MONAI Label Server?\n\n"
								+ "This might override any existing annotations in MONAI Label server for: '" + image
								+ "'")) {

					annotationXML = getAnnotationsXml(image, imageData, new int[4]);
					logger.info("Annotations XML: " + annotationXML);

					MonaiLabelClient.saveLabel(image, annotationXML.toFile(), null, "{}");
					Dialogs.showInfoNotification("MONALabel", "Label/Annotations saved in Server");
				}
			}
		} catch (Exception ex) {
			ex.printStackTrace();
			Dialogs.showErrorMessage("MONAILabel - Pathology", ex);
		} finally {
			Utils.deleteFile(annotationXML);
			Utils.deleteFile(imagePatch);
		}
	}

	private Path getAnnotationsXml(String image, ImageData<BufferedImage> imageData, int[] bbox)
			throws IOException, ParserConfigurationException, TransformerException {
		DocumentBuilderFactory docFactory = DocumentBuilderFactory.newInstance();
		DocumentBuilder docBuilder = docFactory.newDocumentBuilder();

		// root elements
		Document doc = docBuilder.newDocument();
		Element rootElement = doc.createElement("ASAP_Annotations");
		doc.appendChild(rootElement);

		Element annotations = doc.createElement("Annotations");
		annotations.setAttribute("Name", "");
		annotations.setAttribute("Description", "");
		annotations.setAttribute("X", String.valueOf(bbox[0]));
		annotations.setAttribute("Y", String.valueOf(bbox[1]));
		annotations.setAttribute("W", String.valueOf(bbox[2]));
		annotations.setAttribute("H", String.valueOf(bbox[3]));
		rootElement.appendChild(annotations);

		ROI patchROI = (bbox[2] > 0 && bbox[3] > 0) ? ROIs.createRectangleROI(bbox[0], bbox[1], bbox[2], bbox[3], null)
				: null;

		int count = 0;
		var groups = new HashMap<String, String>();
		List<PathObject> objs = imageData.getHierarchy().getFlattenedObjectList(null);
		for (int i = 0; i < objs.size(); i++) {
			var a = objs.get(i);

			// Ignore which doesn't have class
			String name = a.getPathClass() != null ? a.getPathClass().getName() : null;
			if (name == null || name.isEmpty()) {
				continue;
			}

			// Ignore Points
			var roi = a.getROI();
			if (roi == null || roi.isPoint()) {
				continue;
			}

			// Ignore other objects not part of BBOX
			if (patchROI != null && !patchROI.contains(roi.getCentroidX(), roi.getCentroidY())) {
				continue;
			}

			var points = roi.getAllPoints();
			var color = String.format("#%06x", 0xFFFFFF & a.getPathClass().getColor());
			groups.put(name, color);

			Element annotation = doc.createElement("Annotation");
			annotation.setAttribute("Name", name);
			annotation.setAttribute("Type", roi.getRoiName());
			annotation.setAttribute("PartOfGroup", name);
			annotation.setAttribute("Color", color);
			annotations.appendChild(annotation);

			Element coordinates = doc.createElement("Coordinates");
			annotation.appendChild(coordinates);

			for (int j = 0; j < points.size(); j++) {
				var p = points.get(j);
				Element coordinate = doc.createElement("Coordinate");
				coordinate.setAttribute("Order", String.valueOf(j));
				coordinate.setAttribute("X", String.valueOf((int) p.getX() - bbox[0]));
				coordinate.setAttribute("Y", String.valueOf((int) p.getY() - bbox[1]));
				coordinates.appendChild(coordinate);
			}
			count++;
		}

		Element annotationGroups = doc.createElement("AnnotationGroups");
		rootElement.appendChild(annotationGroups);

		for (String group : groups.keySet()) {
			Element annotationGroup = doc.createElement("Group");
			annotationGroup.setAttribute("Name", group);
			annotationGroup.setAttribute("PartOfGroup", "None");
			annotationGroup.setAttribute("Color", groups.get(group));
			annotationGroups.appendChild(annotationGroup);
		}

		logger.info("Total Objects saved: " + count);
		if (count == 0) {
			throw new IOException("ZERO annotations found (nothing to save/submit)");
		}
		return writeXml(image, doc);
	}

	// write doc to output stream
	private Path writeXml(String image, Document doc) throws TransformerException, IOException {
		FileOutputStream output = null;
		try {
			var path = java.nio.file.Files.createTempFile(image, ".xml");
			output = new FileOutputStream(path.toFile());

			TransformerFactory transformerFactory = TransformerFactory.newInstance();
			Transformer transformer = transformerFactory.newTransformer();

			// pretty print
			transformer.setOutputProperty(OutputKeys.INDENT, "yes");

			DOMSource source = new DOMSource(doc);
			StreamResult result = new StreamResult(output);
			transformer.transform(source, result);
			output.close();
			return path;
		} finally {
			if (output != null) {
				output.close();
			}
		}
	}
}
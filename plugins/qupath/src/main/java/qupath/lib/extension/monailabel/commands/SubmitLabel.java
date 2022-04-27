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
import java.nio.file.Files;
import java.nio.file.Path;
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
import qupath.lib.extension.monailabel.Utils;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.gui.dialogs.Dialogs;
import qupath.lib.images.ImageData;
import qupath.lib.objects.PathObject;

public class SubmitLabel implements Runnable {
	private final static Logger logger = LoggerFactory.getLogger(SubmitLabel.class);

	private QuPathGUI qupath;

	public SubmitLabel(QuPathGUI qupath) {
		this.qupath = qupath;
	}

	@Override
	public void run() {
		Path path = null;
		try {
			var viewer = qupath.getViewer();
			var imageData = viewer.getImageData();
			String image = Utils.getNameWithoutExtension(imageData.getServerPath());
			path = getAnnotationsXml(image, imageData);
			logger.info("Annotations XML: " + path);

			String params = "{}";

			if (Dialogs.showYesNoDialog("MONAILabel", "Do you like to submit this annotation to MONAI Label Server?\n\n"
					+ "This might override any existing annotations in MONAI Label server for: '" + image + "'")) {
				String res = MonaiLabelClient.saveLabel(image, path.toFile(), null, params);
				logger.info("SUBMIT:: resp = " + res);
				Dialogs.showInfoNotification("MONALabel", "Label/Annotations saved in Server");
			}
		} catch (Exception ex) {
			ex.printStackTrace();
			Dialogs.showErrorMessage("MONAI Label - Pathology", ex);
		} finally {
			if (path != null) {
				try {
					Files.deleteIfExists(path);
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}
	}

	private Path getAnnotationsXml(String image, ImageData<BufferedImage> imageData)
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
		annotations.setAttribute("X", "0");
		annotations.setAttribute("Y", "0");
		annotations.setAttribute("W", "0");
		annotations.setAttribute("H", "0");
		rootElement.appendChild(annotations);

		int count = 0;
		var groups = new HashMap<String, String>();
		List<PathObject> objs = imageData.getHierarchy().getFlattenedObjectList(null);
		for (int i = 0; i < objs.size(); i++) {
			var a = objs.get(i);
			String name = a.getPathClass() != null ? a.getPathClass().getName() : null;
			if (name == null || name.isEmpty()) {
				continue;
			}

			var roi = a.getROI();
			if (roi == null || roi.isPoint()) {
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
				coordinate.setAttribute("X", String.valueOf((int) p.getX()));
				coordinate.setAttribute("Y", String.valueOf((int) p.getY()));
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
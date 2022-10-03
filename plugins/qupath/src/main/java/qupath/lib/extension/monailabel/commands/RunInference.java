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

import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import javax.xml.parsers.ParserConfigurationException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.w3c.dom.Document;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;
import org.xml.sax.SAXException;

import qupath.lib.extension.monailabel.MonaiLabelClient;
import qupath.lib.extension.monailabel.MonaiLabelClient.RequestInfer;
import qupath.lib.extension.monailabel.MonaiLabelClient.ResponseInfo;
import qupath.lib.extension.monailabel.Utils;
import qupath.lib.geom.Point2;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.gui.dialogs.Dialogs;
import qupath.lib.images.ImageData;
import qupath.lib.images.writers.ImageWriterTools;
import qupath.lib.objects.PathObject;
import qupath.lib.objects.PathObjects;
import qupath.lib.objects.classes.PathClass;
import qupath.lib.objects.classes.PathClassFactory;
import qupath.lib.plugins.parameters.ParameterList;
import qupath.lib.regions.ImagePlane;
import qupath.lib.regions.RegionRequest;
import qupath.lib.roi.ROIs;
import qupath.lib.roi.interfaces.ROI;
import qupath.lib.scripting.QP;

public class RunInference implements Runnable {
	private final static Logger logger = LoggerFactory.getLogger(RunInference.class);

	private QuPathGUI qupath;
	private static String selectedModel;
	private static int[] selectedBBox;
	private static int selectedTileSize = 1024;

	public RunInference(QuPathGUI qupath) {
		this.qupath = qupath;
	}

	@Override
	public void run() {
		try {
			var viewer = qupath.getViewer();
			var imageData = viewer.getImageData();
			var selected = imageData.getHierarchy().getSelectionModel().getSelectedObject();
			var roi = selected != null ? selected.getROI() : null;
			int[] bbox = Utils.getBBOX(roi);
			int tileSize = selectedTileSize;
			if (bbox[2] == 0 && bbox[3] == 0 && selectedBBox != null) {
				bbox = selectedBBox;
			}

			ResponseInfo info = MonaiLabelClient.info();
			List<String> names = new ArrayList<String>();
			Map<String, String[]> labels = new HashMap<String, String[]>();
			for (String n : info.models.keySet()) {
				names.add(n);
				labels.put(n, info.models.get(n).labels.labels());
			}

			ParameterList list = new ParameterList();
			if (selectedModel == null || selectedModel.isEmpty()) {
				selectedModel = names.isEmpty() ? "" : names.get(0);
			}

			list.addChoiceParameter("Model", "Model Name", selectedModel, names);
			list.addStringParameter("Location", "Location (x,y,w,h)", Arrays.toString(bbox));
			list.addIntParameter("TileSize", "TileSize", tileSize);

			boolean override = !info.models.get(selectedModel).nuclick;
			list.addBooleanParameter("Override", "Override", override);

			if (Dialogs.showParameterDialog("MONAILabel", list)) {
				String model = (String) list.getChoiceParameterValue("Model");
				bbox = Utils.parseStringArray(list.getStringParameterValue("Location"));
				override = list.getBooleanParameterValue("Override").booleanValue();
				tileSize = list.getIntParameterValue("TileSize").intValue();

				selectedModel = model;
				selectedBBox = bbox;
				selectedTileSize = tileSize;

				boolean validateClicks = info.models.get(selectedModel).nuclick;
				runInference(model, new HashSet<String>(Arrays.asList(labels.get(model))), bbox, tileSize, imageData,
						override, validateClicks);
			}
		} catch (Exception ex) {
			ex.printStackTrace();
			Dialogs.showErrorMessage("MONAILabel", ex);
		}
	}

	ArrayList<Point2> getClicks(String name, ImageData<BufferedImage> imageData, ROI monaiLabelROI, int offsetX,
			int offsetY) {
		List<PathObject> objs = imageData.getHierarchy().getFlattenedObjectList(null);
		ArrayList<Point2> clicks = new ArrayList<Point2>();
		for (int i = 0; i < objs.size(); i++) {
			String pname = objs.get(i).getPathClass() == null ? "" : objs.get(i).getPathClass().getName();
			if (pname.equalsIgnoreCase(name)) {
				ROI r = objs.get(i).getROI();
				List<Point2> points = r.getAllPoints();
				for (Point2 p : points) {
					if (monaiLabelROI.contains(p.getX(), p.getY())) {
						clicks.add(new Point2(p.getX() - offsetX, p.getY() - offsetY));
					}
				}
			}
		}

		logger.info("MONAILabel:: Total " + name + " clicks/points: " + clicks.size());
		return clicks;
	}

	private void runInference(String model, Set<String> labels, int[] bbox, int tileSize,
			ImageData<BufferedImage> imageData, boolean override, boolean validateClicks)
			throws SAXException, IOException, ParserConfigurationException, InterruptedException {
		logger.info("MONAILabel:: Running Inference...");
		logger.info("MONAILabel:: Model: " + model + "; override: " + override + "; clicks:" + validateClicks
				+ "; Labels: " + labels);

		Path imagePatch = null;
		try {
			RequestInfer req = new RequestInfer();
			req.location[0] = bbox[0];
			req.location[1] = bbox[1];
			req.size[0] = bbox[2];
			req.size[1] = bbox[3];
			req.tile_size[0] = tileSize;
			req.tile_size[1] = tileSize;

			ROI roi = ROIs.createRectangleROI(bbox[0], bbox[1], bbox[2], bbox[3], null);
			String imageFile = imageData.getServerPath();
			if (imageFile.startsWith("file:/"))
				imageFile = imageFile.replace("file:/", "");
			logger.info("MONAILabel:: Image File: " + imageFile);

			String image = Utils.getNameWithoutExtension(imageFile);
			String sessionId = null;
			int offsetX = 0;
			int offsetY = 0;

			// check if image exists on server
			if (!MonaiLabelClient.imageExists(image) && (sessionId == null || sessionId.isEmpty())) {
				logger.info("MONAILabel:: Image does not exist on Server.");
				image = null;
				offsetX = req.location[0];
				offsetY = req.location[1];

				req.location[0] = req.location[1] = 0;
				req.size[0] = req.size[1] = 0;

				var fg = getClicks("Positive", imageData, roi, offsetX, offsetY);
				var bg = getClicks("Negative", imageData, roi, offsetX, offsetY);
				if (validateClicks) {
					if (fg.size() == 0 && bg.size() == 0) {
						Dialogs.showErrorMessage("MONAILabel",
								"Need atleast one Postive/Negative annotation/click point within the ROI");
						return;
					}
					if (roi.getBoundsHeight() < 128 || roi.getBoundsWidth() < 128) {
						Dialogs.showErrorMessage("MONAILabel",
								"Min Height/Width of ROI should be more than 128");
						return;
					}
				}

				req.params.addClicks(fg, true);
				req.params.addClicks(bg, false);

				imagePatch = java.nio.file.Files.createTempFile("patch", ".png");
				imageFile = imagePatch.toString();
				var requestROI = RegionRequest.createInstance(imageData.getServer().getPath(), 1, roi);
				ImageWriterTools.writeImageRegion(imageData.getServer(), requestROI, imageFile);
			}

			Document dom = MonaiLabelClient.infer(model, image, imageFile, sessionId, req);
			NodeList annotation_list = dom.getElementsByTagName("Annotation");
			int count = updateAnnotations(labels, annotation_list, roi, imageData, override, offsetX, offsetY);

			// Update hierarchy to see changes in QuPath's hierarchy
			QP.fireHierarchyUpdate(imageData.getHierarchy());
			logger.info("MONAILabel:: Annotation Done! => Total Objects Added: " + count);
		} finally {
			Utils.deleteFile(imagePatch);
		}
	}

	private int updateAnnotations(Set<String> labels, NodeList annotation_list, ROI roi,
			ImageData<BufferedImage> imageData, boolean override, int offsetX, int offsetY) {
		if (override) {
			List<PathObject> objs = imageData.getHierarchy().getFlattenedObjectList(null);
			for (int i = 0; i < objs.size(); i++) {
				String name = objs.get(i).getPathClass() != null ? objs.get(i).getPathClass().getName() : null;
				if (name != null && labels.contains(name)) {
					ROI r = objs.get(i).getROI();
					if (roi.contains(r.getCentroidX(), r.getCentroidY())) {
						imageData.getHierarchy().removeObjectWithoutUpdate(objs.get(i), false);
					}
				}
			}
		}

		int count = 0;
		for (int i = 0; i < annotation_list.getLength(); i++) {
			Node annotation = annotation_list.item(i);
			String annotationClass = annotation.getAttributes().getNamedItem("Name").getTextContent();
			// logger.info("Annotation Class: " + annotationClass);

			NodeList coordinates_list = annotation.getChildNodes();
			for (int j = 0; j < coordinates_list.getLength(); j++) {
				Node coordinates = coordinates_list.item(j);
				if (coordinates.getNodeType() != Node.ELEMENT_NODE) {
					continue;
				}

				NodeList coordinate_list = coordinates.getChildNodes();
				// logger.info("Total Coordinate: " + coordinate_list.getLength());

				ArrayList<Point2> pointsList = new ArrayList<>();
				for (int k = 0; k < coordinate_list.getLength(); k++) {
					Node coordinate = coordinate_list.item(k);
					if (coordinate.getAttributes() != null) {
						double px = offsetX
								+ Double.parseDouble(coordinate.getAttributes().getNamedItem("X").getTextContent());
						double py = offsetY
								+ Double.parseDouble(coordinate.getAttributes().getNamedItem("Y").getTextContent());
						pointsList.add(new Point2(px, py));
					}
				}
				if (pointsList.isEmpty()) {
					continue;
				}

				ImagePlane plane = ImagePlane.getPlane(0, 0);
				ROI polyROI = ROIs.createPolygonROI(pointsList, plane);
				PathObject annotationObject = PathObjects.createAnnotationObject(polyROI);

				PathClass pclass = PathClassFactory.getPathClass(annotationClass, Color.RED.getRGB());
				annotationObject.setPathClass(pclass);

				imageData.getHierarchy().addPathObjectWithoutUpdate(annotationObject);
				count++;
			}
		}

		return count;
	}
}
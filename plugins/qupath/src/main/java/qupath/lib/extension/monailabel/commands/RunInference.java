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
import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import javax.xml.parsers.ParserConfigurationException;

import org.controlsfx.dialog.ProgressDialog;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.w3c.dom.Document;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;
import org.xml.sax.SAXException;

import javafx.concurrent.Task;
import qupath.lib.common.GeneralTools;
import qupath.lib.extension.monailabel.MonaiLabelClient;
import qupath.lib.extension.monailabel.MonaiLabelClient.RequestInfer;
import qupath.lib.extension.monailabel.MonaiLabelClient.ResponseInfo;
import qupath.lib.extension.monailabel.Settings;
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
import qupath.lib.roi.PointsROI;
import qupath.lib.roi.ROIs;
import qupath.lib.roi.RectangleROI;
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

			if (roi == null || !(roi instanceof RectangleROI)) {
				Dialogs.showPlainMessage("Please create and select ROI", "Please create and select a Rectangle ROI before " +
						"running this method.\nThe \"Annotations\" function creates annotations within the selected rectangle.");
				return;
			}

			var uris = imageData.getServer().getURIs();
			String imageFile = GeneralTools.toPath(uris.iterator().next()).toString();
			String ext = GeneralTools.getExtension(imageFile).get().toLowerCase();
			boolean isWSI = !(ext.equals(".png") || ext.equals(".jpg") || ext.equals(".png"));
			logger.info("MONAILabel:: isWSI: " + isWSI + "; File: " + imageFile);

			// Select first RectangleROI if not selected explicitly
			if (isWSI && (roi == null || !(roi instanceof RectangleROI))) {
				List<PathObject> objs = imageData.getHierarchy().getFlattenedObjectList(null);
				for (int i = 0; i < objs.size(); i++) {
					var obj = objs.get(i);
					ROI r = obj.getROI();
					if (r instanceof RectangleROI) {
						roi = r;
						Dialogs.showWarningNotification("MONALabel",
								"ROI is NOT explicitly selected; using first Rectangle ROI from Hierarchy");
						imageData.getHierarchy().getSelectionModel().setSelectedObject(obj);
						break;
					}
				}
			}

			int[] bbox = Utils.getBBOX(roi);
			int tileSize = selectedTileSize;
			if (isWSI && bbox[2] == 0 && bbox[3] == 0 && selectedBBox != null) {
				bbox = selectedBBox;
			}

			ResponseInfo info = MonaiLabelClient.info();
			List<String> names = Arrays.asList(info.models.keySet().toArray(new String[0]));

			if (selectedModel == null || selectedModel.isEmpty()) {
				selectedModel = names.isEmpty() ? "" : names.get(0);
			}

			ParameterList list = new ParameterList();
			list.addChoiceParameter("Model", "Model Name", selectedModel, names);
			list.addTitleParameter("Parameters of selected ROI:");
			if (isWSI) {
				list.addStringParameter("Location", "Location (x,y,w,h)", Arrays.toString(bbox));
				list.addIntParameter("TileSize", "TileSize", tileSize);
			}

			if (Dialogs.showParameterDialog("MONAILabel", list)) {
				String model = (String) list.getChoiceParameterValue("Model");
				if (isWSI) {
					bbox = Utils.parseStringArray(list.getStringParameterValue("Location"));
					tileSize = list.getIntParameterValue("TileSize").intValue();
				} else {
					bbox = new int[] { 0, 0, 0, 0 };
					tileSize = selectedTileSize;
				}

				selectedModel = model;
				selectedBBox = bbox;
				selectedTileSize = tileSize;

				// runInference(model, info, bbox, tileSize, imageData, imageFile, isWSI);
				final int[] finalBbox = bbox;
				final int finalTileSize = tileSize;

				Task<Void> task = new Task<Void>() {
					@Override
					protected Void call() throws Exception {
						runInference(model, info, finalBbox, finalTileSize, imageData, imageFile, isWSI);
						return null;
					}
				};

				ProgressDialog progressDialog = new ProgressDialog(task);
				progressDialog.setTitle("MONAILabel");
				progressDialog.setHeaderText("Server-side processing is in progress, please wait...");
				progressDialog.setContentText("Annotations will be drawn immediately after the method ends.");
				progressDialog.initOwner(qupath.getStage());

				// Start the inference
				new Thread(task).start();

				task.setOnSucceeded(event -> {
					progressDialog.close();
				});
				task.setOnFailed(event -> {
					progressDialog.close();
					Throwable ex = task.getException();
					if (ex != null) {
						ex.printStackTrace();
						Dialogs.showErrorMessage("MONAILabel", ex);
					}
				});
			}

			imageData.getHierarchy().removeObject(imageData.getHierarchy().getSelectionModel().getSelectedObject(), true);
			imageData.getHierarchy().getSelectionModel().clearSelection();
		} catch (Exception ex) {
			ex.printStackTrace();
			Dialogs.showErrorMessage("MONAILabel", ex);
		}
	}

	public static ArrayList<Point2> getClicks(String name, ImageData<BufferedImage> imageData, ROI monaiLabelROI,
			int offsetX, int offsetY) {
		List<PathObject> objs = imageData.getHierarchy().getFlattenedObjectList(null);
		ArrayList<Point2> clicks = new ArrayList<Point2>();
		for (int i = 0; i < objs.size(); i++) {
			var obj = objs.get(i);
			String pname = obj.getPathClass() == null ? "" : obj.getPathClass().getName();
			if (name.isEmpty() || pname.equalsIgnoreCase(name)) {
				ROI r = obj.getROI();
				if (r instanceof PointsROI) {
					List<Point2> points = r.getAllPoints();
					for (Point2 p : points) {
						if (monaiLabelROI.contains(p.getX(), p.getY())) {
							clicks.add(new Point2(p.getX() - offsetX, p.getY() - offsetY));
						}
					}
				}
			}
		}

		logger.info("MONAILabel:: Total " + name + " clicks/points: " + clicks.size());
		return clicks;
	}

	public static void runInference(String model, ResponseInfo info, int[] bbox, int tileSize,
			ImageData<BufferedImage> imageData, String imageFile, boolean isWSI)
			throws SAXException, IOException, ParserConfigurationException, InterruptedException {
		logger.info("MONAILabel:: Running Inference...; model = " + model);

		boolean isNuClick = info.models.get(model).nuclick;
		boolean override = !isNuClick;
		boolean validateClicks = isNuClick;
		var labels = new HashSet<String>(Arrays.asList(info.models.get(model).labels.labels()));

		logger.info("MONAILabel:: Model: " + model + "; Labels: " + labels);

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

			String image = GeneralTools.getNameWithoutExtension(new File(imageFile));
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

				String im = imageFile.toLowerCase();
				if ((im.endsWith(".png") || im.endsWith(".jpg") || im.endsWith(".jpeg"))
						&& new File(imageFile).exists()) {
					logger.info("Simple Image.. will directly upload the same");
					offsetX = offsetY = 0;
					Dialogs.showWarningNotification("MONAILabel",
							"Ignoring ROI; Running Inference over full non-wsi Image");
				} else {
					if (bbox[2] == 0 && bbox[3] == 0) {
						Dialogs.showErrorMessage("MONAILabel",
								"Can not run WSI Inference on a remote image (Not exists in Datastore)");
						return;
					}

					imagePatch = java.nio.file.Files.createTempFile("patch", ".png");
					imageFile = imagePatch.toString();
					var requestROI = RegionRequest.createInstance(imageData.getServer().getPath(), 1, roi);
					ImageWriterTools.writeImageRegion(imageData.getServer(), requestROI, imageFile);
				}
			}

			ArrayList<Point2> fg = new ArrayList<>();
			ArrayList<Point2> bg = new ArrayList<>();
			if (isNuClick) {
				fg = getClicks("", imageData, roi, offsetX, offsetY);
			} else {
				fg = getClicks("Positive", imageData, roi, offsetX, offsetY);
				bg = getClicks("Negative", imageData, roi, offsetX, offsetY);
			}

			if (validateClicks) {
				if (fg.size() == 0 && bg.size() == 0) {
					Dialogs.showErrorMessage("MONAILabel",
							"Need atleast one Postive/Negative annotation/click point within the ROI");
					return;
				}
				if (roi.getBoundsHeight() < 128 || roi.getBoundsWidth() < 128) {
					Dialogs.showErrorMessage("MONAILabel", "Min Height/Width of ROI should be more than 128");
					return;
				}
			}
			req.params.addClicks(fg, true);
			req.params.addClicks(bg, false);
			req.params.max_workers = Settings.maxWorkersProperty().intValue();

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

	public static int updateAnnotations(Set<String> labels, NodeList annotation_list, ROI roi,
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
		} else {
			List<PathObject> objs = imageData.getHierarchy().getFlattenedObjectList(null);
			for (int i = 0; i < objs.size(); i++) {
				var obj = objs.get(i);
				ROI r = obj.getROI();
				if (r instanceof PointsROI) {
					String pname = obj.getPathClass() == null ? "" : obj.getPathClass().getName();
					if (pname.equalsIgnoreCase("Positive") || pname.equalsIgnoreCase("Negative")) {
						continue;
					}
					imageData.getHierarchy().removeObjectWithoutUpdate(obj, false);
				}
			}
			QP.fireHierarchyUpdate(imageData.getHierarchy());
		}

		int count = 0;
		for (int i = 0; i < annotation_list.getLength(); i++) {
			Node annotation = annotation_list.item(i);
			String annotationClass = annotation.getAttributes().getNamedItem("Name").getTextContent();
			int color = Color.RED.getRGB();
			Node colorNode = annotation.getAttributes().getNamedItem("Color");
			if (colorNode != null) {
				color = Integer.parseInt(colorNode.getTextContent().replaceFirst("#", ""), 16);
			// logger.info("Annotation Class: " + annotationClass + " Annotation Color: " + colorNode.getTextContent());
			}

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

				PathClass pclass = PathClassFactory.getPathClass(annotationClass, color);
				annotationObject.setPathClass(pclass);

				imageData.getHierarchy().addPathObjectWithoutUpdate(annotationObject);
				count++;
			}
		}

		return count;
	}
}

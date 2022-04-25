package qupath.lib.extension.monailabel.commands;

import java.awt.Color;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.w3c.dom.Document;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;

import com.google.common.io.Files;

import qupath.lib.extension.monailabel.MonaiLabelClient;
import qupath.lib.extension.monailabel.MonaiLabelClient.Model;
import qupath.lib.extension.monailabel.MonaiLabelClient.RequestInfer;
import qupath.lib.extension.monailabel.MonaiLabelClient.ResponseInfo;
import qupath.lib.geom.Point2;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.gui.dialogs.Dialogs;
import qupath.lib.images.ImageData;
import qupath.lib.objects.PathObject;
import qupath.lib.objects.PathObjects;
import qupath.lib.objects.classes.PathClass;
import qupath.lib.objects.classes.PathClassFactory;
import qupath.lib.plugins.parameters.ParameterList;
import qupath.lib.regions.ImagePlane;
import qupath.lib.roi.ROIs;
import qupath.lib.roi.RectangleROI;
import qupath.lib.roi.interfaces.ROI;
import qupath.lib.scripting.QP;

/**
 * Run Inference.
 * 
 * @author Sachidanand Alle
 *
 */
public class RunInference implements Runnable {
	private final static Logger logger = LoggerFactory.getLogger(RunInference.class);

	private QuPathGUI qupath;

	public RunInference(QuPathGUI qupath) {
		this.qupath = qupath;
	}

	@Override
	public void run() {
		var viewer = qupath.getViewer();
		var imageData = viewer.getImageData();
		var selected = imageData.getHierarchy().getSelectionModel().getSelectedObject();
		var roi = selected != null ? selected.getROI() : null;
		int[] bbox = getBBOX(roi);

		ResponseInfo info = MonaiLabelClient.info();
		List<String> names = new ArrayList<String>();
		Map<String, String[]> labels = new HashMap<String, String[]>();
		for (String n : info.models.keySet()) {
			names.add(n);
			labels.put(n, info.models.get(n).labels.labels());
		}

		ParameterList list = new ParameterList();
		list.addChoiceParameter("Model", "Model Name", names.isEmpty() ? "" : names.get(0), names);
		list.addIntParameter("X", "Location-x", bbox[0]);
		list.addIntParameter("Y", "Location-y", bbox[1]);
		list.addIntParameter("Width", "Width", bbox[2]);
		list.addIntParameter("Height", "Height", bbox[3]);

		if (Dialogs.showParameterDialog("MONAI Label - Pathology", list)) {
			String model = (String) list.getChoiceParameterValue("Model");
			bbox[0] = list.getIntParameterValue("X").intValue();
			bbox[1] = list.getIntParameterValue("Y").intValue();
			bbox[2] = list.getIntParameterValue("Width").intValue();
			bbox[3] = list.getIntParameterValue("Height").intValue();

			runInference(model, new HashSet<String>(Arrays.asList(labels.get(model))), bbox, imageData);
		}
	}

	ArrayList<Point2> getClicks(String name, ImageData<BufferedImage> imageData, ROI monaiLabelROI) {
		List<PathObject> objs = imageData.getHierarchy().getFlattenedObjectList(null);
		ArrayList<Point2> clicks = new ArrayList<Point2>();
		for (int i = 0; i < objs.size(); i++) {
			if (objs.get(i).getPathClass() != null && objs.get(i).getPathClass().getName() == name) {
				ROI r = objs.get(i).getROI();
				List<Point2> points = r.getAllPoints();
				for (Point2 p : points) {
					if (monaiLabelROI.contains(p.getX(), p.getY())) {
						clicks.add(p);
					}
				}
			}
		}
		return clicks;
	}

	private int[] getBBOX(ROI roi) {
		int x = 0, y = 0, w = 0, h = 0;
		if (roi != null && roi instanceof RectangleROI) {
			List<Point2> points = roi.getAllPoints();
			x = (int) points.get(0).getX();
			y = (int) points.get(0).getY();
			w = (int) points.get(2).getX() - x;
			h = (int) points.get(2).getY() - y;
		}
		return new int[] { x, y, w, h };
	}

	private void runInference(String model, Set<String> labels, int[] bbox, ImageData<BufferedImage> imageData) {
		logger.info("MONAILabel Annotation - Run Inference...");
		logger.info("Model: " + model);
		logger.info("Labels: " + labels);

		try {
			String image = Files.getNameWithoutExtension(imageData.getServerPath());

			RequestInfer req = new RequestInfer();
			req.location[0] = bbox[0];
			req.location[1] = bbox[1];
			req.size[0] = bbox[2];
			req.size[1] = bbox[3];

			ROI roi = ROIs.createRectangleROI(bbox[0], bbox[1], bbox[2], bbox[3], null);
			req.params.addClicks(getClicks("Positive", imageData, roi), true);
			req.params.addClicks(getClicks("Negative", imageData, roi), false);

			Document dom = MonaiLabelClient.infer(model, image, req);
			NodeList annotation_list = dom.getElementsByTagName("Annotation");
			int count = updateAnnotations(labels, annotation_list, roi, imageData);

			// Update hierarchy to see changes in QuPath's hierarchy
			QP.fireHierarchyUpdate(imageData.getHierarchy());
			logger.info("MONAILabel Annotation - Done! => Total Objects Added: " + count);
		} catch (Exception ex) {
			ex.printStackTrace();
		}
	}

	private int updateAnnotations(Set<String> labels, NodeList annotation_list, ROI roi,
			ImageData<BufferedImage> imageData) {
		List<PathObject> objs = imageData.getHierarchy().getFlattenedObjectList(null);
		for (int i = 0; i < objs.size(); i++) {
			String name = objs.get(i).getPathClass() != null ? objs.get(i).getPathClass().getName() : null;
			if (name != null && labels.contains(name)) {
				ROI r = objs.get(i).getROI();
				if (roi.contains(r.getCentroidX(), r.getCentroidY())) {
					imageData.getHierarchy().removeObjectWithoutUpdate(objs.get(i), false);
				}
			} else {
				logger.info("Name: " + name + " Not Found in: " + labels);
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
						double px = Double.parseDouble(coordinate.getAttributes().getNamedItem("X").getTextContent());
						double py = Double.parseDouble(coordinate.getAttributes().getNamedItem("Y").getTextContent());
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
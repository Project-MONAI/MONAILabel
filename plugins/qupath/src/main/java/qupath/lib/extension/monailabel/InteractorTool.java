package qupath.lib.extension.monailabel;

import java.util.ArrayList;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javafx.scene.input.MouseEvent;
import qupath.lib.common.GeneralTools;
import qupath.lib.extension.monailabel.MonaiLabelClient.ResponseInfo;
import qupath.lib.extension.monailabel.commands.RunInference;
import qupath.lib.gui.dialogs.Dialogs;
import qupath.lib.gui.viewer.tools.PointsTool;
import qupath.lib.objects.PathObject;
import qupath.lib.objects.PathROIObject;
import qupath.lib.plugins.parameters.ParameterList;
import qupath.lib.roi.PointsROI;

public class InteractorTool extends PointsTool {
	private final static Logger logger = LoggerFactory.getLogger(InteractorTool.class);

	private static String selectedModel;
	private static int selectedPatchSize = 128;
	private static ResponseInfo info;

	public void mousePressed(MouseEvent e) {
		var viewer = getViewer();
		if (viewer == null || viewer.getImageData() == null) {
			return;
		}

		logger.info("+++++++ Interaction Tool... mouse pressed...");
		super.mousePressed(e);

		PathObject currentObjectTemp = viewer.getSelectedObject();
		if (!(currentObjectTemp == null || currentObjectTemp instanceof PathROIObject))
			return;

		PathROIObject currentObject = (PathROIObject) currentObjectTemp;
		if (currentObject == null || !(currentObject.getROI() instanceof PointsROI))
			return;

		PointsROI roi = (PointsROI) currentObject.getROI();
		String model = selectedModel;
		int patchSize = selectedPatchSize;

		try {
			if (info == null) {
				info = MonaiLabelClient.info();
			}

			List<String> names = new ArrayList<String>();
			for (String n : info.models.keySet()) {
				if (info.models.get(n).nuclick) {
					names.add(n);
				}
			}
			if (names.size() == 1) {
				model = names.get(0);
			}

			var uris = viewer.getImageData().getServer().getURIs();
			String imageFile = GeneralTools.toPath(uris.iterator().next()).toString();
			String ext = GeneralTools.getExtension(imageFile).get().toLowerCase();
			boolean isWSI = !(ext.equals(".png") || ext.equals(".jpg") || ext.equals(".png"));
			logger.info("MONAILabel:: isWSI: " + isWSI + "; File: " + imageFile);

			if (model == null || model.isEmpty()) {
				ParameterList list = new ParameterList();
				list.addChoiceParameter("Model", "Model Name", names.get(0), names);
				if (isWSI) {
					list.addIntParameter("PatchSize", "PatchSize", patchSize);
				}
				list.addBooleanParameter("RememberMe", "Remember My Choice", true);

				if (!Dialogs.showParameterDialog("MONAILabel", list)) {
					return;
				}

				model = (String) list.getChoiceParameterValue("Model");
				patchSize = isWSI ? list.getIntParameterValue("PatchSize").intValue() : patchSize;
				if (list.getBooleanParameterValue("RememberMe").booleanValue()) {
					selectedModel = model;
					selectedPatchSize = patchSize;
				} else {
					selectedModel = null;
					selectedPatchSize = 128;
				}
			}

			int min_x = Integer.MAX_VALUE;
			int min_y = Integer.MAX_VALUE;
			int max_x = 0;
			int max_y = 0;
			for (var p : roi.getAllPoints()) {
				int x = (int) p.getX();
				int y = (int) p.getY();
				min_x = Math.min(x, min_x);
				min_y = Math.min(y, min_y);
				max_x = Math.max(x, max_x);
				max_y = Math.max(y, max_y);
			}

			int w = Math.max(max_x - min_x + 20, patchSize);
			int h = Math.max(max_y - min_y + 20, patchSize);
			int x = Math.max(0, min_x + (max_x - min_x) / 2 - w / 2);
			int y = Math.max(0, min_y + (max_y - min_y) / 2 - h / 2);
			int[] bbox = { x, y, w, h };

			RunInference.runInference(model, info, bbox, patchSize, viewer.getImageData(), imageFile, isWSI);
			// currentObject.setROI(ROIs.createPointsROI(viewer.getImagePlane()));
			viewer.getHierarchy().getSelectionModel().clearSelection();
		} catch (Exception ex) {
			ex.printStackTrace();
			Dialogs.showErrorMessage("MONAILabel", ex);

			selectedModel = null;
			selectedPatchSize = 128;
			info = null;
		}
	}
}

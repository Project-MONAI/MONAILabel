package qupath.lib.extension.monailabel;

import java.util.ArrayList;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javafx.scene.input.MouseEvent;
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

		try {
			if (info == null) {
				info = MonaiLabelClient.info();
				List<String> names = new ArrayList<String>();
				for (String n : info.models.keySet()) {
					if (info.models.get(n).nuclick) {
						names.add(n);
					}
				}
				int patchSize = selectedPatchSize;
				if (names.size() == 0) {
					return;
				}
				if (names.size() == 1) {
					selectedModel = names.get(0);
				}

				if (selectedModel == null || selectedModel.isEmpty()) {
					ParameterList list = new ParameterList();
					list.addChoiceParameter("Model", "Model Name", names.get(0), names);
					list.addIntParameter("PatchSize", "PatchSize", patchSize);

					if (!Dialogs.showParameterDialog("MONAILabel", list)) {
						return;
					}

					selectedModel = (String) list.getChoiceParameterValue("Model");
					selectedPatchSize = list.getIntParameterValue("PatchSize").intValue();
				}
			}

			int w = Math.max((int) roi.getBoundsWidth() + 20, selectedPatchSize);
			int h = Math.max((int) roi.getBoundsHeight() + 20, selectedPatchSize);
			int x = Math.max(0, (int) roi.getCentroidX() - w / 2);
			int y = Math.max(0, (int) roi.getCentroidY() - h / 2);
			int[] bbox = { x, y, w, h };

			RunInference.runInference(selectedModel, info, bbox, selectedPatchSize, viewer.getImageData());
			// currentObject.setROI(ROIs.createPointsROI(viewer.getImagePlane()));
			viewer.getHierarchy().getSelectionModel().clearSelection();
		} catch (Exception ex) {
			ex.printStackTrace();
			Dialogs.showErrorMessage("MONAILabel", ex);
		}
	}
}

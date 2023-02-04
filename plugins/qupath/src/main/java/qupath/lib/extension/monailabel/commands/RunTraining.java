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

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.gson.GsonBuilder;
import com.google.gson.ToNumberPolicy;

import javafx.beans.value.ChangeListener;
import javafx.beans.value.ObservableValue;
import javafx.collections.FXCollections;
import javafx.scene.Node;
import javafx.scene.control.ButtonType;
import javafx.scene.control.CheckBox;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Label;
import javafx.scene.control.Separator;
import javafx.scene.control.TextField;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.VBox;
import qupath.lib.extension.monailabel.MonaiLabelClient;
import qupath.lib.extension.monailabel.MonaiLabelClient.ResponseInfo;
import qupath.lib.gui.dialogs.Dialogs;
import qupath.lib.gui.dialogs.Dialogs.Builder;

class ConfigTable {
	private final static Logger logger = LoggerFactory.getLogger(RunTraining.class);
	private ResponseInfo info;
	private VBox vbox = new VBox();
	private GridPane gridPane;
	private ComboBox<String> models;
	private Map<String, GridPane> cache = new HashMap<>();
	private Map<String, Map<String, Object>> cachedParams = new HashMap<>();
	private Map<String, Map<String, Object>> cachedModifiedParams = new HashMap<>();

	public String selectedModel;
	public Map<String, Object> selectedParams;
	public Map<String, Object> modifiedParams;

	public ConfigTable(ResponseInfo info) {
		this.info = info;
	}

	public Node node() {
		List<String> names = new ArrayList<String>();
		for (String n : info.trainers.keySet()) {
			names.add(n);
		}

		models = new ComboBox<>(FXCollections.observableArrayList(names));
		models.getSelectionModel().select(0);
		models.getSelectionModel().selectedItemProperty().addListener((observable, oldValue, newValue) -> {
			logger.info("Old Value: " + oldValue + "; New Value: " + newValue);
			onSelectModel();
		});

		vbox.setSpacing(5);
		vbox.getChildren().add(new Label("Select Model:"));
		vbox.getChildren().add(models);
		vbox.getChildren().add(new Separator());

		onSelectModel();
		return vbox;
	}

	public void onSelectModel() {
		if (gridPane != null) {
			vbox.getChildren().remove(gridPane);
		}

		String model = models.getSelectionModel().getSelectedItem();
		selectedModel = model;

		if (cache.containsKey(model)) {
			vbox.getChildren().add(gridPane);
			selectedParams = cachedParams.get(model);
			modifiedParams = cachedModifiedParams.get(model);
			return;
		}

		gridPane = new GridPane();
		gridPane.setVgap(5);
		gridPane.setHgap(20);
		gridPane.setMinWidth(300);

		selectedParams = new HashMap<String, Object>();
		modifiedParams = new HashMap<String, Object>();

		int row = 0;
		for (String key : info.trainers.get(model).config.keySet()) {
			Object value = info.trainers.get(model).config.get(key);
			logger.info("Value: " + value + " => " + value.getClass().getSimpleName());

			Node nodeKey = new Label(key);
			Node nodeValue = null;

			if (value instanceof Boolean) {
				CheckBox c = new CheckBox();
				c.setSelected(((Boolean) value).booleanValue());
				c.selectedProperty().addListener(new ChangeListener<Boolean>() {
					@Override
					public void changed(ObservableValue<? extends Boolean> observable, Boolean oldValue,
							Boolean newValue) {
						logger.info(key + " => changed from " + oldValue + " to " + newValue);
						modifiedParams.put(key, newValue);
					}
				});
				nodeValue = c;
			} else if (value instanceof ArrayList) {
				int numCount = 0;
				List<Object> a = new ArrayList<>();
				for (Object v : (ArrayList<?>) value) {
					if (v instanceof Double || v instanceof Long) {
						numCount++;
					}
					a.add(v);
				}

				if (numCount == a.size()) {
					TextField t = new TextField(value.toString());
					t.textProperty().addListener((observable, oldValue, newValue) -> {
						logger.info(key + " => changed from " + oldValue + " to " + newValue);

						try {
							var nv = new GsonBuilder().setObjectToNumberStrategy(ToNumberPolicy.LONG_OR_DOUBLE).create()
									.fromJson(newValue.trim(), Object[].class);
							modifiedParams.put(key, nv);
						} catch (Exception e) {
						}
					});
					nodeValue = t;
				} else {
					ComboBox<String> c = new ComboBox<>(FXCollections.observableArrayList(a.toArray(new String[0])));
					c.getSelectionModel().select(0);
					c.getSelectionModel().selectedItemProperty().addListener((observable, oldValue, newValue) -> {
						logger.info(key + " => changed from " + oldValue + " to " + newValue);
						modifiedParams.put(key, newValue);
					});
					nodeValue = c;
				}
			} else {
				TextField t = new TextField(value.toString());
				t.textProperty().addListener((observable, oldValue, newValue) -> {
					logger.info(key + " => changed from " + oldValue + " to " + newValue);
					newValue = newValue.trim();
					if (!newValue.isBlank()) {
						if (value instanceof Double)
							modifiedParams.put(key, Double.parseDouble(newValue));
						else if (value instanceof Long) {
							modifiedParams.put(key, Long.parseLong(newValue));
						} else {
							modifiedParams.put(key, newValue);
						}
					}
				});
				nodeValue = t;
			}

			gridPane.add(nodeKey, 0, row);
			gridPane.add(nodeValue, 1, row);
			selectedParams.put(key, value);
			row++;
		}

		cache.put(model, gridPane);
		cachedParams.put(model, selectedParams);
		cachedModifiedParams.put(model, modifiedParams);
		vbox.getChildren().add(gridPane);
	}
}

public class RunTraining implements Runnable {
	private final static Logger logger = LoggerFactory.getLogger(RunTraining.class);

	public static boolean showDialog(String title, Node node) {
		return new Builder().buttons(ButtonType.OK, ButtonType.CANCEL).title(title).content(node).resizable()
				.showAndWait().orElse(ButtonType.NO) == ButtonType.OK;
	}

	@Override
	public void run() {
		try {
			ResponseInfo info = MonaiLabelClient.info();
			ConfigTable trainConfig = new ConfigTable(info);
			if (!showDialog("MONAILabel - Training", trainConfig.node())) {
				return;
			}

			String model = trainConfig.selectedModel;
			Map<String, Object> params = trainConfig.selectedParams;
			Map<String, Object> modified = trainConfig.modifiedParams;
			logger.info("Selected Model: " + model);
			logger.info("Selected Params: " + params);
			logger.info("Modified Params: " + modified);

			String p = new GsonBuilder().setObjectToNumberStrategy(ToNumberPolicy.LONG_OR_DOUBLE).create()
					.toJson(modified);
			logger.info("PARAMS: " + p);

			String res = MonaiLabelClient.train(model, p);
			logger.info("TRAINING:: resp = " + res);
			Dialogs.showInfoNotification("MONALabel", "Training job started...");
		} catch (Exception ex) {
			ex.printStackTrace();
			Dialogs.showErrorMessage("MONAILabel", ex);
		}
	}
}

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

import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.Type;
import java.net.URLEncoder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.w3c.dom.Document;
import org.xml.sax.SAXException;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonArray;
import com.google.gson.JsonDeserializationContext;
import com.google.gson.JsonDeserializer;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParseException;

import qupath.lib.geom.Point2;

public class MonaiLabelClient {
	private final static Logger logger = LoggerFactory.getLogger(MonaiLabelClient.class);

	public static class Labels {
		public String[] array;
		public Map<String, Integer> map;

		public String[] labels() {
			if (array != null) {
				return array;
			}
			return map.keySet().toArray(String[]::new);
		}
	}

	public static class LabelsDeserializer implements JsonDeserializer<Labels> {

		@Override
		public Labels deserialize(JsonElement paramJsonElement, Type paramType,
				JsonDeserializationContext paramJsonDeserializationContext) throws JsonParseException {

			Labels labels = new Labels();
			if (paramJsonElement.isJsonArray()) {
				JsonArray arr = paramJsonElement.getAsJsonArray();
				labels.array = new String[arr.size()];
				for (int i = 0; i < arr.size(); i++) {
					labels.array[i] = arr.get(i).getAsString();
				}
			} else {
				JsonObject o = paramJsonElement.getAsJsonObject();
				labels.map = new HashMap<String, Integer>();
				for (String key : o.keySet()) {
					labels.map.put(key, o.get(key).getAsInt());
				}
			}
			return labels;
		}

	}

	public static class Model {
		public String type;
		public int dimension;
		public String description;
		public Labels labels;
		public boolean pathology;
		public boolean nuclick;
	}

	public static class Strategy {
		public String description;
	}

	public static class ResponseInfo {
		public String name;
		public String description;
		public String version;
		public Labels labels;
		public Map<String, Model> models;
		public Map<String, Strategy> strategies;
	}

	public static class ImageInfo {
		public String image;
	}

	public static class LabelInfo {
		public String label;
	}

	public static class NextSampleInfo {
		public String id;
		public int[] bbox = { 0, 0, 0, 0 };
	}

	public static class InferParams {
		public List<List<Integer>> foreground = new ArrayList<>();
		public List<List<Integer>> background = new ArrayList<>();

		public void addClicks(ArrayList<Point2> clicks, boolean f) {
			List<List<Integer>> t = f ? foreground : background;
			for (int i = 0; i < clicks.size(); i++) {
				int x = (int) clicks.get(i).getX();
				int y = (int) clicks.get(i).getY();
				t.add(Arrays.asList(new Integer[] { x, y }));
			}
		}
	};

	public static class RequestInfer {
		public int level = 0;
		public int[] location = { 0, 0 };
		public int[] size = { 0, 0 };
		public int[] tile_size = { 1024, 1024 };
		public int min_poly_area = 30;
		public InferParams params = new InferParams();
	};

	public static ResponseInfo info() throws IOException, InterruptedException {
		String uri = "/info/";
		String res = RequestUtils.request("GET", uri, null);
		logger.info("MONAILabel:: INFO Response => " + res);

		Gson gson = new GsonBuilder().registerTypeAdapter(Labels.class, new LabelsDeserializer()).create();
		return gson.fromJson(res, ResponseInfo.class);
	}

	public static Document infer(String model, String image, String imageFile, String sessionId, RequestInfer req)
			throws SAXException, IOException, ParserConfigurationException, InterruptedException {

		String uri = "/infer/wsi_v2/" + URLEncoder.encode(model, "UTF-8") + "?output=asap";
		if (image != null && !image.isEmpty())
			uri += "&image=" + URLEncoder.encode(image, "UTF-8");
		if (sessionId != null && !sessionId.isEmpty())
			uri += "&session_id=" + URLEncoder.encode(sessionId, "UTF-8");

		String jsonBody = new Gson().toJson(req, RequestInfer.class);
		logger.info("MONAILabel:: Request BODY => " + jsonBody);

		String response;
		if (image == null) {
			var files = new HashMap<String, File>();
			files.put("file", new File(imageFile));

			var fields = new HashMap<String, String>();
			fields.put("wsi", jsonBody);

			response = RequestUtils.requestMultiPart("POST", uri, files, fields);
		} else {
			uri = uri.replace("/infer/wsi_v2/", "/infer/wsi/");
			response = RequestUtils.request("POST", uri, jsonBody);
		}

		DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
		DocumentBuilder builder = factory.newDocumentBuilder();
		InputStream inputStream = new ByteArrayInputStream(response.getBytes());
		Document dom = builder.parse(inputStream);
		return dom;
	}

	public static String train(String model, String params) throws IOException, InterruptedException {
		String uri = "/train/" + URLEncoder.encode(model, "UTF-8");
		return RequestUtils.request("POST", uri, params);
	}

	public static ImageInfo saveImage(String image, File imageFile, String params)
			throws IOException, InterruptedException {
		String uri = "/datastore/image?image=" + URLEncoder.encode(image, "UTF-8");

		var files = new HashMap<String, File>();
		files.put("file", imageFile);

		var fields = new HashMap<String, String>();
		fields.put("params", params);

		String res = RequestUtils.requestMultiPart("PUT", uri, files, fields);
		return new GsonBuilder().create().fromJson(res, ImageInfo.class);
	}

	public static LabelInfo saveLabel(String image, File label, String tag, String params)
			throws IOException, InterruptedException {
		String uri = "/datastore/label?image=" + URLEncoder.encode(image, "UTF-8");
		if (tag != null && !tag.isEmpty()) {
			uri += "&tag=" + tag;
		}

		var files = new HashMap<String, File>();
		files.put("label", label);

		var fields = new HashMap<String, String>();
		fields.put("params", params);

		String res = RequestUtils.requestMultiPart("PUT", uri, files, fields);
		return new GsonBuilder().create().fromJson(res, LabelInfo.class);
	}

	public static boolean imageExists(String image) {
		try {
			String uri = "/datastore/image?image=" + URLEncoder.encode(image, "UTF-8");
			String res = RequestUtils.request("HEAD", uri, null);

			logger.info("MONAILabel:: (Image Exists) Response => " + res);
			return true;
		} catch (IOException | InterruptedException e) {
			return false;
		}
	}

	public static NextSampleInfo nextSample(String strategy, String params) throws IOException, InterruptedException {
		String uri = "/activelearning/" + URLEncoder.encode(strategy, "UTF-8");
		logger.info("MONAILabel:: Next Sample Request (" + strategy + ") => " + params);

		String res = RequestUtils.request("POST", uri, params);
		return new GsonBuilder().create().fromJson(res, NextSampleInfo.class);
	}
}

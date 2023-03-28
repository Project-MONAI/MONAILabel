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

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.math.BigInteger;
import java.net.URI;
import java.net.URL;
import java.net.URLEncoder;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpRequest.BodyPublisher;
import java.net.http.HttpRequest.BodyPublishers;
import java.net.http.HttpResponse;
import java.net.http.HttpResponse.BodyHandlers;
import java.nio.channels.Channels;
import java.nio.channels.ReadableByteChannel;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.gson.GsonBuilder;

import javafx.scene.control.ButtonType;
import javafx.scene.control.Label;
import javafx.scene.control.PasswordField;
import javafx.scene.control.TextField;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.VBox;
import qupath.lib.gui.dialogs.Dialogs.Builder;

public class RequestUtils {
	private final static Logger logger = LoggerFactory.getLogger(RequestUtils.class);
	private static String username = null;
	private static String password = null;
	private static AuthInfo auth_info = null;
	private static AuthToken auth_token = null;

	private static class AuthInfo {
		public boolean enabled;
		public String client_id;
		public String realm;
	}

	private static class AuthToken {
		public String access_token;
		public String token_type;
	}

	private static void showLoginForm() {
		VBox vbox = new VBox();
		GridPane gridPane = new GridPane();
		TextField usernameField = new TextField();
		PasswordField passwordField = new PasswordField();

		if (username != null) {
			usernameField.setText(username);
			if (password != null)
				passwordField.setText(password);
		}

		gridPane.setVgap(5);
		gridPane.setHgap(20);
		gridPane.setMinWidth(300);

		gridPane.add(new Label("UserName"), 0, 0);
		gridPane.add(usernameField, 1, 0);
		usernameField.setPromptText("name");

		gridPane.add(new Label("Password"), 0, 1);
		gridPane.add(passwordField, 1, 1);
		passwordField.setPromptText("password");

		vbox.getChildren().add(gridPane);

		if (new Builder().buttons(ButtonType.OK, ButtonType.CANCEL).title("User Login").content(vbox).resizable()
				.showAndWait().orElse(ButtonType.NO) == ButtonType.OK) {
			username = usernameField.getText();
			password = passwordField.getText();
		} else {
			password = null;
		}
	}

	private static AuthToken getAuthToken() {
		if (username == null || username.isEmpty() || password == null || password.isEmpty()) {
			showLoginForm();
		}

		if (username == null || username.isEmpty() || password == null || password.isEmpty()) {
			logger.warn("Continuing without Login...");
			return null;
		}

		Map<Object, Object> params = new HashMap<>();
		params.put("username", username);
		params.put("password", password);

		String monaiServer = Settings.serverURLProperty().get();
		String requestURI = monaiServer + "/auth/token";
		var requestBuilder = HttpRequest.newBuilder().version(HttpClient.Version.HTTP_1_1).POST(toFormData(params))
				.header("Content-Type", "application/x-www-form-urlencoded").uri(URI.create(requestURI));

		var httpClient = HttpClient.newBuilder().build();
		HttpResponse<String> response;
		try {
			response = httpClient.send(requestBuilder.build(), BodyHandlers.ofString());
		} catch (IOException | InterruptedException e) {
			logger.info("Failed to send http request", e);
			username = null;
			return null;
		}

		logger.info("Auth Token:: Response (code): " + response.statusCode());
		// logger.info("Auth Token:: Response (body): " + response.body());

		if (response.statusCode() != 200) {
			username = null;
			return null;
		}

		var res = response.body();
		return new GsonBuilder().create().fromJson(res, AuthToken.class);
	}

	public static HttpRequest.BodyPublisher toFormData(Map<Object, Object> data) {
		var builder = new StringBuilder();
		for (Map.Entry<Object, Object> entry : data.entrySet()) {
			if (builder.length() > 0) {
				builder.append("&");
			}
			builder.append(URLEncoder.encode(entry.getKey().toString(), StandardCharsets.UTF_8));
			builder.append("=");
			builder.append(URLEncoder.encode(entry.getValue().toString(), StandardCharsets.UTF_8));
		}
		return HttpRequest.BodyPublishers.ofString(builder.toString());
	}

	public static boolean isAuthEnabled() {
		String monaiServer = Settings.serverURLProperty().get();
		String requestURI = monaiServer + "/auth/";
		var request = HttpRequest.newBuilder().version(HttpClient.Version.HTTP_1_1).GET().uri(URI.create(requestURI))
				.build();

		var httpClient = HttpClient.newBuilder().build();
		HttpResponse<String> response;
		try {
			response = httpClient.send(request, BodyHandlers.ofString());
		} catch (IOException | InterruptedException e) {
			logger.info("Failed to send http request", e);
			return false;
		}

		// logger.info("Auth Enabled:: Response (code): " + response.statusCode());
		// logger.info("Auth Enabled:: Response (body): " + response.body());

		if (response.statusCode() != 200) {
			return false;
		}

		var res = response.body();
		auth_info = new GsonBuilder().create().fromJson(res, AuthInfo.class);
		return auth_info.enabled;
	}

	public static boolean isValidToken() {
		if (auth_token == null) {
			return false;
		}

		String monaiServer = Settings.serverURLProperty().get();
		String requestURI = monaiServer + "/auth/token/valid";
		var request = HttpRequest.newBuilder().version(HttpClient.Version.HTTP_1_1).GET().uri(URI.create(requestURI))
				.header("Authorization", auth_token.token_type + " " + auth_token.access_token);

		var httpClient = HttpClient.newBuilder().build();
		HttpResponse<String> response;
		try {
			response = httpClient.send(request.build(), BodyHandlers.ofString());
		} catch (IOException | InterruptedException e) {
			logger.info("Failed to send http request", e);
			return false;
		}

		// logger.info("Valid Token:: Response (code): " + response.statusCode());
		// logger.info("Valid Token:: Response (body): " + response.body());

		if (response.statusCode() != 200) {
			return false;
		}
		return true;
	}

	private static HttpRequest.Builder addAuthHeader(HttpRequest.Builder builder) {
		if (isAuthEnabled()) {
			if (auth_token == null || !isValidToken()) {
				auth_token = getAuthToken();
			}
			if (auth_token != null) {
				builder = builder.header("Authorization", auth_token.token_type + " " + auth_token.access_token);
			}
		}
		return builder;
	}

	public static String request(String method, String uri, String body) throws IOException, InterruptedException {
		String monaiServer = Settings.serverURLProperty().get();
		String requestURI = monaiServer + uri;
		logger.info("MONAILabel:: Request URL => " + requestURI);

		var bodyPublisher = (body != null && !body.isEmpty()) ? HttpRequest.BodyPublishers.ofString(body)
				: HttpRequest.BodyPublishers.noBody();

		var requestBuilder = HttpRequest.newBuilder().version(HttpClient.Version.HTTP_1_1).method(method, bodyPublisher)
				.uri(URI.create(requestURI));
		requestBuilder = addAuthHeader(requestBuilder);

		var httpClient = HttpClient.newBuilder().build();
		var response = httpClient.send(requestBuilder.build(), BodyHandlers.ofString());
		if (response.statusCode() != 200) {
			logger.info("Error Response (code): " + response.statusCode());
			logger.info("Error Response (body): " + response.body());
			throw new IOException(response.toString());
		}
		return response.body();
	}

	public static void download(String uri, File file) throws IOException, InterruptedException {
		String monaiServer = Settings.serverURLProperty().get();
		String requestURI = monaiServer + uri;
		logger.info("MONAILabel:: Download URL => " + requestURI);

		var url = new URL(requestURI);
		var conn = url.openConnection();
		if (isAuthEnabled()) {
			if (auth_token == null || !isValidToken()) {
				auth_token = getAuthToken();
			}
			if (auth_token != null) {
				conn.setRequestProperty("Authorization", auth_token.token_type + " " + auth_token.access_token);
			}
		}

		ReadableByteChannel rbc = Channels.newChannel(conn.getInputStream());
		FileOutputStream fos = new FileOutputStream(file);
		fos.getChannel().transferFrom(rbc, 0, Long.MAX_VALUE);
		fos.close();
		rbc.close();
	}

	public static String requestMultiPart(String method, String uri, Map<String, File> files,
			Map<String, String> fields) throws IOException, InterruptedException {
		String monaiServer = Settings.serverURLProperty().get();
		String requestURI = monaiServer + uri;
		logger.info("MONAILabel:: MultiPart Request URL => " + requestURI);

		var multipartData = MultipartData.newBuilder().withCharset(StandardCharsets.UTF_8);

		// Add Files
		if (files != null && !files.isEmpty())
			for (var file : files.entrySet())
				multipartData.addFile(file.getKey(), file.getValue().toPath(),
						Files.probeContentType(file.getValue().toPath()));

		// Add fields
		if (fields != null && !fields.isEmpty())
			for (var field : fields.entrySet())
				multipartData.addText(field.getKey(), field.getValue());

		var mdata = multipartData.build();
		var requestBuilder = HttpRequest.newBuilder().version(HttpClient.Version.HTTP_1_1)
				.header("Content-Type", mdata.getContentType()).method(method, mdata.getBodyPublisher())
				.uri(URI.create(requestURI));
		requestBuilder = addAuthHeader(requestBuilder);

		var httpClient = HttpClient.newBuilder().build();
		var response = httpClient.send(requestBuilder.build(), BodyHandlers.ofString());
		if (response.statusCode() != 200) {
			logger.info("Error Response (code): " + response.statusCode());
			logger.info("Error Response (body): " + response.body());
			throw new IOException(response.toString());
		}
		return response.body();
	}

	public static class MultipartData {

		public static class Builder {

			private String boundary;
			private Charset charset = StandardCharsets.UTF_8;
			private List<MimedFile> files = new ArrayList<MimedFile>();
			private Map<String, String> texts = new LinkedHashMap<>();

			private Builder() {
				this.boundary = new BigInteger(128, new Random()).toString();
			}

			public Builder withCharset(Charset charset) {
				this.charset = charset;
				return this;
			}

			public Builder withBoundary(String boundary) {
				this.boundary = boundary;
				return this;
			}

			public Builder addFile(String name, Path path, String mimeType) {
				this.files.add(new MimedFile(name, path, mimeType));
				return this;
			}

			public Builder addText(String name, String text) {
				texts.put(name, text);
				return this;
			}

			public MultipartData build() throws IOException {
				MultipartData multipartData = new MultipartData();
				multipartData.boundary = boundary;

				var newline = "\r\n".getBytes(charset);
				var byteArrayOutputStream = new ByteArrayOutputStream();
				for (var f : files) {
					byteArrayOutputStream.write(("--" + boundary).getBytes(charset));
					byteArrayOutputStream.write(newline);
					byteArrayOutputStream.write(("Content-Disposition: form-data; name=\"" + f.name + "\"; filename=\""
							+ f.path.getFileName() + "\"").getBytes(charset));
					byteArrayOutputStream.write(newline);
					byteArrayOutputStream.write(("Content-Type: " + f.mimeType).getBytes(charset));
					byteArrayOutputStream.write(newline);
					byteArrayOutputStream.write(newline);
					byteArrayOutputStream.write(Files.readAllBytes(f.path));
					byteArrayOutputStream.write(newline);
				}
				for (var entry : texts.entrySet()) {
					byteArrayOutputStream.write(("--" + boundary).getBytes(charset));
					byteArrayOutputStream.write(newline);
					byteArrayOutputStream.write(
							("Content-Disposition: form-data; name=\"" + entry.getKey() + "\"").getBytes(charset));
					byteArrayOutputStream.write(newline);
					byteArrayOutputStream.write(newline);
					byteArrayOutputStream.write(entry.getValue().getBytes(charset));
					byteArrayOutputStream.write(newline);
				}
				byteArrayOutputStream.write(("--" + boundary + "--").getBytes(charset));

				multipartData.bodyPublisher = BodyPublishers.ofByteArray(byteArrayOutputStream.toByteArray());
				return multipartData;
			}

			public class MimedFile {

				public final String name;
				public final Path path;
				public final String mimeType;

				public MimedFile(String name, Path path, String mimeType) {
					this.name = name;
					this.path = path;
					this.mimeType = mimeType;
				}
			}
		}

		private String boundary;
		private BodyPublisher bodyPublisher;

		private MultipartData() {
		}

		public static Builder newBuilder() {
			return new Builder();
		}

		public BodyPublisher getBodyPublisher() throws IOException {
			return bodyPublisher;
		}

		public String getContentType() {
			return "multipart/form-data; boundary=" + boundary;
		}
	}
}

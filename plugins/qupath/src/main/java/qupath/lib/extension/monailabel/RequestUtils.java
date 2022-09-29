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
import java.io.IOException;
import java.math.BigInteger;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpRequest.BodyPublisher;
import java.net.http.HttpRequest.BodyPublishers;
import java.net.http.HttpResponse.BodyHandlers;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class RequestUtils {
	private final static Logger logger = LoggerFactory.getLogger(RequestUtils.class);

	public static String request(String method, String uri, String body) throws IOException, InterruptedException {
		String monaiServer = Settings.serverURLProperty().get();
		String requestURI = monaiServer + uri;
		logger.info("MONAILabel:: Request URL => " + requestURI);

		var bodyPublisher = (body != null && !body.isEmpty()) ? HttpRequest.BodyPublishers.ofString(body)
				: HttpRequest.BodyPublishers.noBody();

		var request = HttpRequest.newBuilder().version(HttpClient.Version.HTTP_1_1).method(method, bodyPublisher)
				.uri(URI.create(requestURI)).build();

		var httpClient = HttpClient.newBuilder().build();
		var response = httpClient.send(request, BodyHandlers.ofString()); // supporting string response only
		if (response.statusCode() != 200) {
			logger.info("Error Response (code): " + response.statusCode());
			logger.info("Error Response (body): " + response.body());
			throw new IOException(response.toString());
		}
		return response.body();
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
		var request = HttpRequest.newBuilder().version(HttpClient.Version.HTTP_1_1)
				.header("Content-Type", mdata.getContentType()).method(method, mdata.getBodyPublisher())
				.uri(URI.create(requestURI)).build();

		var httpClient = HttpClient.newBuilder().build();
		var response = httpClient.send(request, BodyHandlers.ofString()); // supporting string response only
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

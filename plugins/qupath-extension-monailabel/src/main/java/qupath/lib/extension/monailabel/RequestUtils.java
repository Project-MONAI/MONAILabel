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

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.Reader;
import java.net.HttpURLConnection;
import java.net.URL;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class RequestUtils {
	private final static Logger logger = LoggerFactory.getLogger(RequestUtils.class);

	public static String request(String method, String uri, String body) {
		try {
			String monaiServer = Settings.serverURLProperty().get();
			String requestURI = monaiServer + uri;
			logger.info("MONAI Label Annotation - URL => " + requestURI);

			HttpURLConnection connection = (HttpURLConnection) new URL(requestURI).openConnection();
			connection.setRequestMethod(method);
			connection.setRequestProperty("Content-Type", "application/json");
			connection.setDoInput(true);
			connection.setDoOutput(true);

			if (body != null && !body.isEmpty()) {
				connection.getOutputStream().write(body.getBytes("UTF-8"));
			}

			Reader in = new BufferedReader(new InputStreamReader(connection.getInputStream(), "UTF-8"));
			StringBuilder sb = new StringBuilder();
			for (int c; (c = in.read()) >= 0;)
				sb.append((char) c);
			return sb.toString();
		} catch (Exception ex) {
			ex.printStackTrace();
		}

		return null;
	}

}

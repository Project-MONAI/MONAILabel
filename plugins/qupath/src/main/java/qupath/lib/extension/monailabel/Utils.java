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

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import qupath.lib.geom.Point2;
import qupath.lib.roi.RectangleROI;
import qupath.lib.roi.interfaces.ROI;

public class Utils {
	public static String getNameWithoutExtension(String file) {
		String fileName = new File(file).getName();
		int dotIndex = fileName.lastIndexOf('.');
		return (dotIndex == -1) ? fileName : fileName.substring(0, dotIndex);
	}

	public static int[] getBBOX(ROI roi) {
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

	public static int[] parseStringArray(String str) {
		String[] fields = str.trim().replace("[", "").replace("]", "").split(",");
		int res[] = new int[fields.length];

		for (int i = 0; i < fields.length; i++)
			res[i] = Integer.parseInt(fields[i].trim());
		return res;
	}

	public static void deleteFile(Path path) {
		if (path != null) {
			try {
				Files.deleteIfExists(path);
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
	}

}

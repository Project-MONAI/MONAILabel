/*
 Copyright 2020 - 2021 MONAI Consortium
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

import qupath.lib.scripting.QP
import qupath.lib.geom.Point2
import qupath.lib.roi.PolygonROI
import qupath.lib.objects.PathAnnotationObject
import qupath.lib.images.servers.ImageServer

import org.apache.commons.io.FilenameUtils

def imageData = getCurrentImageData()
def image = FilenameUtils.getBaseName(imageData.getServerPath())

def pathObjects = getSelectedObjects()
if (pathObjects.isEmpty()) {
    Dialogs.showErrorMessage("MONAI Label", "Please select a parent object!")
    return
}

def roi = pathObjects[0].getROI()
def bbox = [[roi.x, roi.y], [roi.x2, roi.y2]]


def monailabel = 'http://127.0.0.1:8000'
def model = 'deepedit'
def server_url = monailabel + '/infer/wsi/' + model + '?image=' + image
print server_url

def foreground = '[]'
def strROI = '{"x": '+ (int)roi.x + ', "y": '+ (int)roi.y + ', "x2": '+ (int)roi.x2 + ', "y2": '+ (int)roi.y2 + '}'
def patch_size = '[4096, 4096]'
def min_poly_area = '1000'

def body = '{"level": 0, "patch_size": ' + patch_size + ', "min_poly_area": ' + min_poly_area + ', "roi": ' + strROI + ', "params": {"foreground": ' + foreground + '}}'
print body


def connection = new URL(server_url).openConnection()
connection.setDoInput(true)
connection.setDoOutput(true)
connection.setRequestMethod('POST')
connection.setRequestProperty('Content-Type' , 'application/json')
connection.getOutputStream().withWriter("UTF-8") { w ->
    w.write(body)
}


def responseBody
connection.getInputStream().withReader("UTF-8") { r ->
    responseBody = r.readLines().join("\n")
}

print connection.responseCode
print "Added new annotation objects"


def hierarchy = QP.getCurrentHierarchy()
def list = new XmlSlurper().parseText(responseBody)
def count = 0
list.Annotations.Annotation.each {
    def annotationClass = getPathClass(it.@Name.toString())
    def tmp_points_list = []
    it.Coordinates.Coordinate.each { vertex ->
        X = vertex.@X.toDouble()
        Y = vertex.@Y.toDouble()
        tmp_points_list.add(new Point2(X, Y))
    }

    def poly = new PolygonROI(tmp_points_list)
    def annotation = new PathAnnotationObject(poly)

    // Set the class here below
    annotation.setPathClass(annotationClass)
    hierarchy.addPathObject(annotation, false)
    count = count + 1
}

// Update hierarchy to see changes in QuPath's hierarchy
fireHierarchyUpdate()
print "Done! => Total Objects Added: " + count
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
def model = 'segmentation'
def server_url = monailabel + '/infer/' + model + '?image=' + image + '&wsi=true&output=image'
print server_url

final def boundary =  'abcd' + Long.toString(System.currentTimeMillis()) * 2 + 'dcba'
final def twoHyphens = '--'
final def lineEnd = '\r\n'
final def hash = 'test'

def json = '{"level": 0, "patch_size": [2048, 2048], "roi": '+ bbox + '}'
print json


def connection = new URL(server_url).openConnection()
connection.setDoInput(true)
connection.setDoOutput(true)
connection.setRequestMethod('POST')
connection.setRequestProperty('Content-Type' , 'multipart/form-data; boundary=' + boundary)

def outputStream = new DataOutputStream(connection.getOutputStream())
outputStream.writeBytes(twoHyphens + boundary + lineEnd);
outputStream.writeBytes('Content-Disposition: form-data; name="params"' + lineEnd)
outputStream.writeBytes(lineEnd)
outputStream.writeBytes(json)
outputStream.writeBytes(lineEnd)

outputStream.writeBytes(twoHyphens + boundary + twoHyphens + lineEnd)
outputStream.flush()
outputStream.close()


def responseBody
connection.getInputStream().withReader("UTF-8") { r ->
    responseBody = r.readLines().join("\n")
}

print connection.responseCode
print "Added new annotation objects"


def hierarchy = QP.getCurrentHierarchy()
def list = new XmlSlurper().parseText(responseBody)
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
}

// Update hierarchy to see changes in QuPath's hierarchy
fireHierarchyUpdate()
print "Done!"
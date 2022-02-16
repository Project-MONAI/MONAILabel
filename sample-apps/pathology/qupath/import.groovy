import qupath.lib.scripting.QP
import qupath.lib.geom.Point2
import qupath.lib.roi.PolygonROI
import qupath.lib.objects.PathAnnotationObject
import qupath.lib.images.servers.ImageServer

//Aperio Image Scope displays images in a different orientation
def rotated = false

def server = QP.getCurrentImageData().getServer()
def h = server.getHeight()
def w = server.getWidth()

// need to add annotations to hierarchy so qupath sees them
def hierarchy = QP.getCurrentHierarchy()

//Prompt user for exported aperio image scope annotation file
def path = server.getURIs().getAt(0).getPath();           // HERE
path = path.substring(0, path.lastIndexOf(".")) + ".xml"  // HERE
path = "/local/sachi/Data/Pathology/BCSS/TCGA-OL-A5RW-01Z-00-DX1.E16DE8EE-31AF-4EAF-A85F-DB3E3E2C3BFF.xml"
print path

def file = new File(path)
def text = file.getText()

def list = new XmlSlurper().parseText(text)

list.Annotations.Annotation.each {
    // Get the class from your XML
    def annotationClass = getPathClass(it.@Name.toString())

    def tmp_points_list = []
    it.Coordinates.Coordinate.each { vertex ->
        if (rotated) {
            X = vertex.@Y.toDouble()
            Y = h - vertex.@X.toDouble()
        }
        else {
            X = vertex.@X.toDouble()
            Y = vertex.@Y.toDouble()
        }
        tmp_points_list.add(new Point2(X, Y))
    }

    def roi = new PolygonROI(tmp_points_list)
    def annotation = new PathAnnotationObject(roi)


    // Set the class here below
    annotation.setPathClass(annotationClass)

    hierarchy.addPathObject(annotation, false)
    print "Added new object"
}

// Update hierarchy to see changes in QuPath's hierarchy
fireHierarchyUpdate()

print "Done!"
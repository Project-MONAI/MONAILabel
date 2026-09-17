package org.monailabel.qupath

import java.awt.image.BufferedImage
import qupath.lib.objects.PathObjects
import qupath.lib.roi.RoiTools

/** Native selection guides, exact crop footprints, and scoped object replacement. */
class SelectionRegion {
    static final String ROLE = 'MONAILabel.SelectionGuide'
    static boolean guide(object) { object.pathClass == null && object.getMetadata()[ROLE] == 'true' }
    static List labels(imageData) { imageData.hierarchy.annotationObjects.findAll { !guide(it) } }
    static void markSelectedGuides(imageData) {
        imageData.hierarchy.selectionModel.selectedObjects.findAll {
            it.isAnnotation() && it.pathClass == null && it.ROI?.isArea()
        }.each { it.getMetadata().put(ROLE, 'true') }
    }

    static Map capture(imageData) {
        def selected = imageData.hierarchy.selectionModel.selectedObjects
        if (selected.size() != 1) throw new IllegalArgumentException('Select exactly one box or area to annotate. The full image was not sent.')
        def object = selected.first()
        def region = captureObject(object, imageData.server.width, imageData.server.height)
        if (object.pathClass == null) object.getMetadata().put(ROLE, 'true')
        return region
    }
    static Map captureObject(object, int imageWidth, int imageHeight) {
        def roi = object.ROI
        if (!object.isAnnotation() || roi == null || !roi.isArea() || roi.z != 0 || roi.t != 0)
            throw new IllegalArgumentException('Select an area annotation on this 2D image.')
        int x = Math.max(0, (int)Math.floor(roi.boundsX)), y = Math.max(0, (int)Math.floor(roi.boundsY))
        int right = Math.min(imageWidth, (int)Math.ceil(roi.boundsX + roi.boundsWidth))
        int bottom = Math.min(imageHeight, (int)Math.ceil(roi.boundsY + roi.boundsHeight))
        int width = right - x, height = bottom - y
        if (width <= 0 || height <= 0) throw new IllegalArgumentException('The selected region is outside the image.')
        def canvas = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY)
        def g = canvas.createGraphics()
        g.translate(-x, -y); g.setColor(java.awt.Color.WHITE); g.fill(RoiTools.getShape(roi)); g.dispose()
        byte[] pixels = (byte[])canvas.raster.dataBuffer.data
        List runs = []
        int start = -1
        for (int i = 0; i <= pixels.length; i++) {
            boolean inside = i < pixels.length && pixels[i] != 0
            if (inside && start < 0) start = i
            if (!inside && start >= 0) { runs.add([start, i]); start = -1 }
        }
        if (!runs) throw new IllegalArgumentException('The selection contains no image pixels.')
        if (runs.size() > 65536) throw new IllegalArgumentException('The selected outline is too complex. Select a smaller region.')
        def scope = [x: x, y: y, width: width, height: height]
        if (!(runs.size() == 1 && runs[0] == [0, pixels.length])) scope.runs = runs
        return [scope: scope, object: object, pixels: pixels]
    }
    static byte[] merge(byte[] current, byte[] prediction, List ids, int width, Map region = null) {
        if (prediction.length != current.length) throw new IllegalArgumentException('Returned mask dimensions differ from the sample.')
        byte[] merged = current.clone()
        for (int i = 0; i < merged.length; i++) {
            if (region) {
                int x = i % width - (int)region.scope.x, y = (int)(i / width) - (int)region.scope.y
                if (x < 0 || y < 0 || x >= region.scope.width || y >= region.scope.height || region.pixels[y * (int)region.scope.width + x] == 0) continue
            }
            if (ids.contains(Byte.toUnsignedInt(merged[i]))) merged[i] = 0
            if (ids.contains(Byte.toUnsignedInt(prediction[i]))) {
                if (merged[i] != 0) throw new IllegalArgumentException('Prediction overlaps another locally edited class.')
                merged[i] = prediction[i]
            }
        }
        return merged
    }
    static List replacement(imageData, byte[] merged, List labels, List ids, Map region) {
        int width = imageData.server.width, height = imageData.server.height
        byte[] inside = new byte[width * height]
        for (int y = 0; y < region.scope.height; y++) for (int x = 0; x < region.scope.width; x++) {
            if (region.pixels[y * (int)region.scope.width + x] == 0) continue
            int i = (y + (int)region.scope.y) * width + x + (int)region.scope.x
            if (ids.contains(Byte.toUnsignedInt(merged[i]))) inside[i] = merged[i]
        }
        return outsideObjects(imageData, labels, ids, region) + MaskObjects.decode(inside, width, height, labels)
    }
    static List outsideObjects(imageData, List labels, List ids, Map region) {
        int width = imageData.server.width, height = imageData.server.height
        byte[] footprint = new byte[width * height]
        for (int y = 0; y < region.scope.height; y++) for (int x = 0; x < region.scope.width; x++) {
            if (region.pixels[y * (int)region.scope.width + x] != 0)
                footprint[(y + (int)region.scope.y) * width + x + (int)region.scope.x] = 1
        }
        def cut = MaskObjects.decode(footprint, width, height, [[id: 1, name: 'Selection']])[0].ROI
        def targetNames = labels.findAll { ids.contains(it.id as int) }.collect { it.name }
        List objects = []
        imageData.hierarchy.annotationObjects.each { object ->
            if (guide(object) || !targetNames.contains(MaskObjects.labelName(object)) || !object.ROI.geometry.intersects(cut.geometry)) {
                objects.add(object)
            } else {
                def remainder = RoiTools.combineROIs(object.ROI, cut, RoiTools.CombineOp.SUBTRACT)
                if (!remainder.isEmpty()) {
                    def preserved = PathObjects.createAnnotationObject(remainder, object.pathClass)
                    preserved.setName(object.name); preserved.getMetadata().putAll(object.getMetadata())
                    preserved.setLocked(object.isLocked()); objects.add(preserved)
                }
            }
        }
        return objects
    }
}

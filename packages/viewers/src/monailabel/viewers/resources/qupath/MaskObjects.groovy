package org.monailabel.qupath

import java.awt.image.BufferedImage
import java.security.MessageDigest
import qupath.lib.analysis.images.ContourTracing
import qupath.lib.analysis.images.SimpleImages
import qupath.lib.objects.classes.PathClass
import qupath.lib.roi.RoiTools

/** Pixel-exact transfer between the service's row-major masks and native editable ROIs. */
class MaskObjects {
    static String labelName(object) { object.pathClass?.baseClass?.toString() }
    static List decode(byte[] mask, int width, int height, List labels) {
        if (mask.length != width * height) throw new IllegalArgumentException('Mask dimensions differ from the sample.')
        float[] pixels = new float[mask.length]
        for (int i = 0; i < mask.length; i++) pixels[i] = Byte.toUnsignedInt(mask[i])
        def objects = ContourTracing.createAnnotations(SimpleImages.createFloatImage(pixels, width, height), null, 1, -1)
        def names = labels.collectEntries { [(it.id as int): it.name] }
        for (def object : objects) {
            int id = Integer.parseInt(object.name)
            if (!names.containsKey(id)) throw new IllegalArgumentException('Unknown mask label ' + id)
            object.setPathClass(PathClass.fromString(names[id]))
            object.setName(names[id])
        }
        return objects
    }
    static byte[] encode(Collection objects, int width, int height, List labels) {
        def names = labels.collectEntries { [(it.name): (it.id as int)] }
        def image = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY)
        byte[] result = new byte[width * height]
        // Render each label independently to detect conflicting classes rather than overwrite.
        def grouped = objects.groupBy { object ->
            def name = labelName(object)
            if (!names.containsKey(name) || names[name] == 0)
                throw new IllegalArgumentException('Classify every annotation with a project foreground label before submitting.')
            if (!object.ROI.isArea()) throw new IllegalArgumentException('Only area annotations can be submitted as segmentation.')
            if (object.ROI.z != 0 || object.ROI.t != 0) throw new IllegalArgumentException('This connection supports a single 2D image plane.')
            name
        }
        grouped.each { name, items ->
            def g = image.createGraphics()
            g.setBackground(java.awt.Color.BLACK); g.clearRect(0, 0, width, height)
            g.setColor(java.awt.Color.WHITE)
            items.each { g.fill(RoiTools.getShape(it.ROI)) }
            g.dispose()
            byte[] pixels = (byte[])image.raster.dataBuffer.data
            for (int i = 0; i < result.length; i++) if (pixels[i] != 0) {
                if (result[i] != 0) throw new IllegalArgumentException('Different classes overlap. Resolve them before submitting.')
                result[i] = (byte)(int)names[name]
            }
        }
        return result
    }
    static String signature(Collection objects) {
        def text = objects.collect { [it.getID(), it.pathClass?.toString(), it.ROI?.geometry?.toText()] }.toString()
        return MessageDigest.getInstance('SHA-256').digest(text.getBytes('UTF-8')).encodeHex().toString()
    }
}

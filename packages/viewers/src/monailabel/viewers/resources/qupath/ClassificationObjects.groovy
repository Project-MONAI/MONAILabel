package org.monailabel.qupath

import qupath.lib.objects.PathObjects
import qupath.lib.objects.classes.PathClass
import qupath.lib.roi.RoiTools

/** Object identities, independent categories, and undo-safe classification application. */
class ClassificationObjects {
    static Map prepare(imageData, Map project, boolean selectedOnly, List targets) {
        def ids = project.labels.findAll { (it.id as int) != 0 }.collectEntries { [(it.name): it.id as int] }
        def names = targets ?: ids.keySet()
        def source = selectedOnly ? imageData.hierarchy.selectionModel.selectedObjects : imageData.hierarchy.annotationObjects
        def originals = source.findAll { it.isAnnotation() && names.contains(MaskObjects.labelName(it)) }
        if (!originals) throw new IllegalArgumentException('No labeled annotation objects to classify. Annotate nuclei first, or select labeled objects.')
        def objects = [], requests = []
        for (def original : originals) {
            // Older masks can store disjoint nuclei as a single multipart annotation.
            def parts = RoiTools.splitROI(original.ROI)
            for (def roi : parts) {
                def object = PathObjects.createAnnotationObject(roi, original.pathClass)
                object.setName(original.name); object.getMetadata().putAll(original.getMetadata())
                object.setLocked(original.isLocked())
                def region = SelectionRegion.captureObject(object, imageData.server.width, imageData.server.height)
                objects.add(object)
                requests.add([id: object.getID().toString(), label_id: ids[MaskObjects.labelName(object)], region: region.scope])
            }
        }
        if (objects.size() > 128) throw new IllegalArgumentException('Classify up to 128 objects at a time. Select fewer nuclei and say "classify selected nuclei".')
        return [originals: originals, objects: objects, requests: requests]
    }
    static List apply(imageData, Map project, Map asset, Map proposal, Map plan, List categories) {
        if (proposal.project_id != project.id || proposal.asset_id != asset.id || proposal.base_revision != asset.revision)
            throw new IllegalStateException('Classification belongs to another sample or revision.')
        def results = proposal.results.collectEntries { [(it.object_id): it.category] }
        if (results.size() != plan.objects.size() || proposal.results.size() != results.size() ||
                results.keySet() != plan.requests.collect { it.id }.toSet())
            throw new IllegalArgumentException('Classification results do not match the requested objects.')
        def replacements = []
        for (def object : plan.objects) {
            String category = results[object.getID().toString()]
            if (category != null && !categories.contains(category)) throw new IllegalArgumentException('Unknown classification category.')
            String base = MaskObjects.labelName(object)
            object.setPathClass(PathClass.fromString(base + ': ' + (category ?: 'Unclassified')))
            object.setName(base + ' · ' + (category ?: 'Unclassified'))
            object.getMetadata().put('MONAILabel.ClassificationProposal', proposal.id)
            object.getMetadata().put('MONAILabel.Classification', category ?: '')
            replacements.add(object)
        }
        return imageData.hierarchy.annotationObjects.findAll { !plan.originals.contains(it) } + replacements
    }
}

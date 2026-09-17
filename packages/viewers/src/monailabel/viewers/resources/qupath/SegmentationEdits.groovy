package org.monailabel.qupath

/** Plan local object edits before changing the hierarchy or undo history. */
class SegmentationEdits {
    static List clear(imageData, Map project, Map asset, Map action, Map region, json) {
        if (action.project_id != project.id || action.asset_id != asset.id || action.base_revision != asset.revision)
            throw new IllegalStateException('Clear request belongs to another sample or revision. Reload before editing.')
        def ids = action.label_ids.collect { it as int }
        def labels = project.labels.findAll { ids.contains(it.id as int) && (it.id as int) != 0 }
        if (!ids || labels.size() != ids.toSet().size())
            throw new IllegalArgumentException('Clear request must name project foreground labels.')
        if (action.image_region != null) {
            if (!region || !json.toJsonTree(action.image_region).equals(json.toJsonTree(region.scope + [runs: region.scope.runs])))
                throw new IllegalStateException('Clear scope differs from the requested selection.')
            return SelectionRegion.outsideObjects(imageData, project.labels, ids, region)
        }
        if (region) throw new IllegalStateException('The requested selection scope was not returned.')
        def names = labels.collect { it.name }
        return imageData.hierarchy.annotationObjects.findAll { !names.contains(MaskObjects.labelName(it)) }
    }
}

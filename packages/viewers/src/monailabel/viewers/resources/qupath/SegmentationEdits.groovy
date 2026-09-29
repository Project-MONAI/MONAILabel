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

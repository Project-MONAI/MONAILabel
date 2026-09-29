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

import javafx.application.Platform
import javafx.scene.control.MenuItem
import qupath.lib.gui.QuPathGUI
import qupath.lib.gui.extensions.QuPathExtension
import qupath.lib.common.Version

/** A native QuPath entry point; all annotation work goes through the backend API. */
class MonaiLabelExtension implements QuPathExtension {
    AnnotationPanel panel
    void installExtension(QuPathGUI qupath) {
        def item = new MenuItem('MONAI Label · Annotation assistant')
        item.setOnAction { show(qupath) }
        qupath.getMenu('Extensions', true).items.add(item)
        if (System.getenv('MONAILABEL_VIEWER_SESSION')) Platform.runLater { show(qupath) }
    }
    void show(QuPathGUI qupath) {
        if (panel == null) panel = new AnnotationPanel(qupath)
        panel.show()
    }
    String getName() { 'MONAI Label' }
    String getDescription() { 'Prompt annotation and revision-aware review submission' }
    Version getQuPathVersion() { Version.parse('0.7.0') }
    Version getVersion() { Version.parse('0.1.0') }
}

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

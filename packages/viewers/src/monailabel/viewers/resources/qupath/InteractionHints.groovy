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

import javafx.scene.input.MouseEvent
import javafx.scene.input.MouseButton
import qupath.lib.gui.viewer.overlays.PathOverlay
import java.awt.Color
import java.awt.BasicStroke
import java.awt.geom.Ellipse2D
import java.awt.geom.Rectangle2D

/** Source-pixel prompts, independent of the annotation hierarchy. */
class InteractionHints {
    final AnnotationPanel panel
    List<Map> objects = []
    String mode = 'navigate'
    List start
    def viewer
    InteractionHints(AnnotationPanel panel) { this.panel = panel }
    String target() { panel.targetChoice.editor.text.trim() }
    List<Map> visible() { objects.findAll { it.target.equalsIgnoreCase(target()) } }
    boolean active() { panel.imageData && viewer?.imageData == panel.imageData && panel.modelChoice.value?.interaction }
    void install() {
        viewer = panel.qupath.viewer
        objects = new ArrayList((List)(panel.imageData.getProperty('MONAILabel.InputHints') ?: []))
        viewer.view.addEventFilter(javafx.scene.input.KeyEvent.KEY_PRESSED, { event ->
            if (active() && event.code == javafx.scene.input.KeyCode.ESCAPE) {
                navigate(); panel.controls()
            }
        } as javafx.event.EventHandler)
        viewer.customOverlayLayers.add({ g, region, downsample, image, complete ->
            if (!active() || image != panel.imageData) return
            def graphics = g.create()
            try {
                graphics.setStroke(new BasicStroke((float)(2 * downsample)))
                visible().each { hint ->
                    def p = hint.coordinates[0]
                    graphics.setColor(hint.kind == 'box' ? new Color(34, 200, 223) : hint.positive ? new Color(22, 163, 74) : new Color(220, 38, 38))
                    if (hint.kind == 'point') {
                        double r = 5 * downsample
                        graphics.fill(new Ellipse2D.Double(p[1] - r, p[0] - r, r * 2, r * 2))
                    } else {
                        def end = hint.coordinates[1]
                        graphics.draw(new Rectangle2D.Double(p[1], p[0], end[1] - p[1], end[0] - p[0]))
                    }
                }
            } finally { graphics.dispose() }
        } as PathOverlay)
        viewer.view.addEventFilter(MouseEvent.ANY, { event ->
            if (!active() || panel.busy || mode == 'navigate' || !target() || !(panel.canAnnotate || panel.canReview)) return
            if (event.button != MouseButton.PRIMARY && !event.primaryButtonDown) return
            if (!(event.eventType in [MouseEvent.MOUSE_PRESSED, MouseEvent.MOUSE_DRAGGED, MouseEvent.MOUSE_RELEASED, MouseEvent.MOUSE_CLICKED])) return
            event.consume()
            def local = viewer.view.sceneToLocal(event.sceneX, event.sceneY)
            def point = viewer.componentPointToImagePoint(local.x, local.y, null, false)
            List p = [point.y, point.x]
            if (p[0] < 0 || p[1] < 0 || p[0] >= panel.imageData.server.height || p[1] >= panel.imageData.server.width) { start = null; return }
            if (event.eventType == MouseEvent.MOUSE_PRESSED) {
                if (mode == 'box') start = p
                else {
                    def nearby = visible().find { h -> h.kind == 'point' && Math.hypot(h.coordinates[0][0] - p[0], h.coordinates[0][1] - p[1]) < 7 * viewer.downsampleFactor }
                    if (nearby) objects.remove(nearby)
                    else if (objects.size() < 128) objects.add([id: UUID.randomUUID().toString(), target: target(), kind: 'point', coordinates: [p], positive: mode == 'positive', selected: false])
                    changed()
                }
            } else if (event.eventType == MouseEvent.MOUSE_RELEASED && start) {
                def a = start; start = null
                if (Math.abs(a[0] - p[0]) < 1 || Math.abs(a[1] - p[1]) < 1) return
                objects.removeAll { it.kind == 'box' && it.target.equalsIgnoreCase(target()) }
                objects.add([id: UUID.randomUUID().toString(), target: target(), kind: 'box', coordinates: [[Math.min(a[0], p[0]), Math.min(a[1], p[1])], [Math.max(a[0], p[0]), Math.max(a[1], p[1])]], positive: true, selected: false])
                changed()
            }
        } as javafx.event.EventHandler)
    }
    void changed() {
        panel.imageData.setProperty('MONAILabel.InputHints', new ArrayList(objects))
        viewer.repaint()
    }
    void navigate() { mode = 'navigate'; start = null; viewer?.repaint() }
    void apply(Map action) {
        if (action.asset_id != panel.asset.id || action.base_revision != panel.asset.revision || !panel.backend.json.toJsonTree(action.expected).equals(panel.backend.json.toJsonTree(objects)))
            throw new IllegalStateException('The viewer inputs changed. Retry the request; your annotations are preserved.')
        if (action.client_action == 'set_interaction_mode') {
            if (action.model_id) panel.modelChoice.selectionModel.select(panel.modelChoice.items.find { it.id == action.model_id })
            if (action.target) panel.targetChoice.editor.text = action.target
            mode = action.mode
        } else {
            objects.removeAll { it.id in action.remove || it.id in action.upsert.collect { h -> h.id } }
            objects.addAll(action.upsert)
        }
        changed(); panel.controls()
    }
}

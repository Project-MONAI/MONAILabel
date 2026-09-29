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
import javafx.geometry.Insets
import javafx.scene.Scene
import javafx.scene.control.*
import javafx.scene.input.KeyCode
import javafx.scene.layout.*
import javafx.stage.Stage
import javafx.stage.Screen
import qupath.lib.gui.QuPathGUI
import qupath.lib.objects.classes.PathClass
import qupath.lib.images.servers.ImageServerProvider
import qupath.lib.images.ImageData
import qupath.lib.projects.Projects
import qupath.lib.projects.ProjectIO
import java.awt.image.BufferedImage
import java.nio.file.Files
import java.nio.file.Path
import java.util.concurrent.Executors

class AnnotationPanel {
    final QuPathGUI qupath
    final Stage stage = new Stage()
    final Tab tab = new Tab('MONAI Label')
    final Button popButton = new Button('Pop out')
    VBox pane
    boolean floating = false
    final TextArea history = new TextArea()
    final TextArea prompt = new TextArea()
    final ComboBox<Map> modelChoice = new ComboBox<>()
    final ComboBox<String> targetChoice = new ComboBox<>()
    final HBox inputTools = new HBox(4)
    final Separator inputSeparator = new Separator(javafx.geometry.Orientation.VERTICAL)
    final SplitMenuButton updateButton = new SplitMenuButton()
    final HBox interactionToolbar = new HBox(6)
    final InteractionHints hints = new InteractionHints(this)
    String updateScope = 'selected_region'
    final Label status = new Label('Connecting…')
    final Button sendButton = new Button('Send prompt')
    final Button cancelButton = new Button('Cancel job')
    final Button submitButton = new Button('Submit annotation for review')
    final Button draftButton = new Button('Save QuPath draft')
    final executor = Executors.newSingleThreadExecutor { runnable ->
        def thread = new Thread(runnable, 'monailabel-qupath'); thread.daemon = true; thread
    }
    Backend backend
    Map project, asset
    List models = []
    List submissionRegions = []
    List savedRegions = []
    boolean wholeImageDraft = false
    Object imageData, entry
    String jobId, proposalId, conversationId
    boolean busy = false
    boolean canAnnotate = false
    boolean canReview = false
    boolean reviewMode = false
    final TextField reviewComment = new TextField()
    final Button goodButton = new Button('Good')
    final Button badButton = new Button('Bad / needs changes')
    final Button correctedButton = new Button('Good with corrections')
    final FlowPane reviewButtons = new FlowPane(8, 6, goodButton, badButton, correctedButton)
    final List<List> undo = []
    final List<List> redo = []
    List classificationCategories = ['Tumor', 'Immune', 'Stromal']

    AnnotationPanel(QuPathGUI qupath) {
        this.qupath = qupath
        stage.initOwner(qupath.stage)
        stage.title = 'MONAI Label · Pathology assistant'
        history.editable = false; history.wrapText = true
        history.setId('monailabel-history')
        prompt.wrapText = true; prompt.prefRowCount = 4; prompt.setId('monailabel-prompt')
        prompt.promptText = 'Annotate the selected region for nuclei\nOr: annotate the full image for nuclei'
        prompt.setOnKeyPressed { event ->
            if (event.controlDown && event.code == KeyCode.ENTER) { send(); event.consume() }
        }
        targetChoice.editable = true; targetChoice.maxWidth = Double.MAX_VALUE
        targetChoice.editor.promptText = 'Choose or type a target label'; targetChoice.setAccessibleText('Target label')
        targetChoice.editor.textProperty().addListener { observable, previous, current -> hints.navigate(); controls() }
        modelChoice.valueProperty().addListener { observable, previous, current ->
            if (current?.heading) {
                int index = modelChoice.items.indexOf(current)
                int direction = modelChoice.items.indexOf(previous) <= index ? 1 : -1
                int next = index + direction
                while (next >= 0 && next < modelChoice.items.size() && modelChoice.items[next].heading) next += direction
                if (next >= 0 && next < modelChoice.items.size()) modelChoice.selectionModel.select(next)
                else modelChoice.selectionModel.select(previous)
                return
            }
            configureModel()
        }
        updateButton.text = '▶ Update region'; updateButton.setAccessibleText('Update segmentation')
        updateButton.style = '-fx-base: #116b87; -fx-text-base-color: white; -fx-mark-color: white;'
        updateButton.setOnAction { send(null, [viewer_scope: updateScope]) }
        [['Update region', 'selected_region'], ['Update image', 'full']].each { label, scope ->
            def item = new MenuItem(label)
            item.setOnAction { updateScope = scope; updateButton.text = '▶ ' + label; send(null, [viewer_scope: scope]) }
            updateButton.items.add(item)
        }
        inputSeparator.prefHeight = 20
        def spacer = new Region(); HBox.setHgrow(spacer, Priority.ALWAYS)
        interactionToolbar.children.addAll(inputTools, inputSeparator, spacer, updateButton)
        interactionToolbar.alignment = javafx.geometry.Pos.CENTER_LEFT
        sendButton.setOnAction { send() }
        cancelButton.disable = true
        cancelButton.setOnAction {
            if (jobId) {
                // The annotation worker may be polling; use a separate short-lived task for cancellation.
                Thread.startDaemon { try { backend.request('/api/jobs/' + jobId + '/cancel', [:]) }
                    catch (Exception error) { Platform.runLater { log(error.message) } } }
            }
        }
        draftButton.setOnAction { try { saveDraft(); log('Saved QuPath draft. It is not submitted for review.') }
            catch (Exception error) { log(error.message) } }
        submitButton.setOnAction { submit() }
        def split = new SplitPane(history, prompt); split.orientation = javafx.geometry.Orientation.VERTICAL
        split.setDividerPositions(0.78)
        reviewComment.promptText = 'Review comment (optional)'
        reviewComment.setId('monailabel-review-comment')
        goodButton.setId('monailabel-review-good'); badButton.setId('monailabel-review-bad')
        correctedButton.setId('monailabel-review-corrected')
        goodButton.setOnAction { reviewDecision('accepted') }
        badButton.setOnAction { reviewDecision('changes_requested') }
        correctedButton.setOnAction { reviewDecision('accepted') }
        pane = new VBox(10, new HBox(10, new Label('Annotation assistant'), popButton), new Label('Model'), modelChoice, targetChoice, interactionToolbar, split,
            new HBox(8, sendButton, cancelButton), draftButton, submitButton, reviewComment, reviewButtons, status)
        pane.padding = new Insets(14); VBox.setVgrow(split, Priority.ALWAYS)
        pane.addEventFilter(javafx.scene.input.KeyEvent.KEY_PRESSED, { event ->
            if (event.code == KeyCode.ESCAPE) { hints.navigate(); controls() }
        } as javafx.event.EventHandler)
        status.wrapText = true; modelChoice.maxWidth = Double.MAX_VALUE
        stage.scene = new Scene(new VBox(), 470, 760)
        stage.minWidth = 360; stage.minHeight = 480
        stage.setOnShown {
            def bounds = Screen.getScreensForRectangle(qupath.stage.x, qupath.stage.y, qupath.stage.width, qupath.stage.height)[0].visualBounds
            stage.x = bounds.maxX - stage.width; stage.y = bounds.minY + 24
        }
        stage.setOnCloseRequest { event -> event.consume(); dock() }
        popButton.setOnAction { if (floating) dock(); else popOut() }
        tab.closable = false; tab.setId('monailabel-tab'); tab.content = pane
        qupath.analysisTabPane.tabs.add(tab)
        qupath.analysisTabPane.selectionModel.select(tab)
        background({ connect() }, { data -> opened(data) })
    }
    void show() {
        if (floating) { stage.show(); stage.toFront() }
        else qupath.analysisTabPane.selectionModel.select(tab)
    }
    void popOut() {
        tab.content = null; stage.scene.root = pane; floating = true; popButton.text = 'Dock in QuPath'
        def restore = new Button('Show annotation window'); restore.setOnAction { show() }
        tab.content = new VBox(12, new Label('Annotation assistant is in a separate window.'), restore)
        stage.show(); stage.toFront()
    }
    void dock() {
        stage.hide(); stage.scene.root = new VBox(); tab.content = pane; floating = false
        popButton.text = 'Pop out'; qupath.analysisTabPane.selectionModel.select(tab)
    }
    void log(String text) { status.text = text; history.appendText('Assistant: ' + text + '\n') }
    void background(Closure work, Closure done) {
        busy = true; controls()
        executor.submit {
            try {
                def value = work()
                Platform.runLater {
                    busy = false
                    try { done(value) } catch (Exception error) { log(error.message ?: error.toString()) }
                    finally { controls() }
                }
            } catch (Exception error) {
                Platform.runLater { busy = false; jobId = null; controls(); log(error.message ?: error.toString()) }
            }
        }
    }
    void configureModel() {
        hints.navigate(); inputTools.children.clear()
        def model = modelChoice.value
        def kinds = model?.interaction?.inputs ?: [:]
        updateScope = model?.interaction ? 'full' : 'selected_region'
        updateButton.text = model?.interaction ? '▶ Update image' : '▶ Update region'
        [['positive', 'positive_point', '+ Point', 'M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18 M8 12h8 M12 8v8'], ['negative', 'negative_point', '− Point', 'M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18 M8 12h8'], ['box', 'box', 'Box', 'M5 3H3v2 M8 3h8 M19 3h2v2 M21 8v8 M21 19v2h-2 M16 21H8 M5 21H3v-2 M3 16V8']].each { mode, kind, title, path ->
            if (!kinds.containsKey(kind)) return
            def icon = new javafx.scene.shape.SVGPath(); icon.content = path
            icon.fill = javafx.scene.paint.Color.TRANSPARENT; icon.stroke = javafx.scene.paint.Color.web('#536779'); icon.strokeWidth = 1.8
            icon.scaleX = 0.8; icon.scaleY = 0.8
            def button = new ToggleButton(); button.graphic = icon; button.userData = mode
            button.tooltip = new Tooltip(title); button.setAccessibleText(title); button.prefWidth = 30; button.prefHeight = 30
            button.setOnAction { hints.mode = hints.mode == mode ? 'navigate' : mode; controls() }
            inputTools.children.add(button)
        }
        inputSeparator.visible = !inputTools.children.empty; inputSeparator.managed = inputSeparator.visible
        String text = targetChoice.editor.text
        targetChoice.items.setAll(model?.supported_targets != null ? model.supported_targets : (project?.labels ?: []).findAll { it.id }.collect { it.name })
        targetChoice.editor.text = text
        controls()
    }
    void controls() {
        modelChoice.disable = busy
        targetChoice.disable = busy || !(canAnnotate || canReview)
        updateButton.disable = busy || !imageData || !targetChoice.editor.text.trim() || !(canAnnotate || canReview)
        inputTools.children.each { button -> button.disable = updateButton.disable; button.selected = hints.mode == button.userData }
        sendButton.disable = busy || !imageData
        submitButton.disable = busy || !canAnnotate || !imageData
        submitButton.visible = canAnnotate && !reviewMode; submitButton.managed = submitButton.visible
        draftButton.disable = busy || !imageData
        cancelButton.disable = !jobId
        reviewButtons.visible = canReview; reviewButtons.managed = canReview
        reviewComment.visible = canReview; reviewComment.managed = canReview
        [goodButton, badButton, correctedButton].each { it.disable = busy || !canReview || !asset?.annotation_id }
    }
    Map connect() {
        backend = new Backend()
        reviewMode = backend.settings.mode == 'review'
        def pid = backend.settings.project_id
        def aid = backend.settings.asset_id
        def project = backend.request('/api/projects/' + pid)
        def asset = backend.request('/api/assets/' + aid)
        if (asset.project_id != pid || asset.kind != 'image2d') throw new IllegalArgumentException('Select a 2D pathology image for QuPath.')
        def models = backend.request('/api/projects/' + pid + '/models?asset_id=' + aid)
        def permissions = backend.request('/api/projects/' + pid + '/permissions')
        def units = backend.request('/api/projects/' + pid + '/review-units')
        def root = Path.of(System.getenv('MONAILABEL_QUPATH_DATA'), pid, aid)
        Files.createDirectories(root)
        def suffix = asset.name.lastIndexOf('.') >= 0 ? asset.name.substring(asset.name.lastIndexOf('.')) : '.png'
        def image = root.resolve('source' + suffix)
        if (!Files.exists(image)) Files.write(image, (byte[])backend.request('/api/assets/' + aid + '/image', null, true))
        return [project: project, asset: asset, models: models, permissions: permissions,
            root: root, image: image, units: units.findAll { it.asset_id == aid }]
    }
    void opened(Map data) {
        project = data.project; asset = data.asset; models = data.models
        savedRegions = data.units.findAll { it.scope.kind == 'region' }.collect { it.scope.region }
        canAnnotate = data.permissions.roles.any { it in ['manager', 'annotator'] }
        canReview = data.permissions.roles.any { it in ['manager', 'reviewer'] }
        List choices = []
        models.groupBy { it.catalog_group ?: 'Other models' }.each { group, members ->
            choices.add([name: group, heading: true, divider: !choices.empty])
            choices.addAll(members)
        }
        modelChoice.cellFactory = { ignored -> new ListCell<Map>() {
            @Override
            protected void updateItem(Map item, boolean empty) {
                super.updateItem(item, empty)
                text = empty || item == null ? null : item.name
                disable = !empty && item?.heading == true
                style = item?.heading ? ('-fx-font-weight: bold;' + (item.divider ? '-fx-border-color: #b8c5d0; -fx-border-width: 1 0 0 0;' : '')) : ''
            }
        } }
        modelChoice.buttonCell = new ListCell<Map>() {
            @Override
            protected void updateItem(Map item, boolean empty) {
                super.updateItem(item, empty)
                text = empty || item == null ? null : item.name
            }
        }
        modelChoice.items.setAll(choices)
        def selected = choices.find { it.id && it.id == project.annotation_model_id } ?: choices.find { it.id }
        modelChoice.selectionModel.select(selected)
        def projectFile = data.root.resolve('project.qpproj').toFile()
        def nativeProject
        boolean restoring = projectFile.exists()
        if (restoring) nativeProject = ProjectIO.loadProject(projectFile, BufferedImage)
        else {
            nativeProject = Projects.createProject(data.root.toFile(), BufferedImage)
            def server = ImageServerProvider.buildServer(data.image.toString(), BufferedImage)
            entry = nativeProject.addImage(server.builder); entry.setImageName(asset.name)
            entry.saveImageData(new ImageData(server, ImageData.ImageType.BRIGHTFIELD_H_E))
            server.close(); nativeProject.syncChanges()
        }
        nativeProject.setPathClasses(project.labels.findAll { it.id != 0 }.collect { PathClass.fromString(it.name) })
        entry = nativeProject.imageList[0]
        qupath.setProject(nativeProject)
        qupath.openImageEntry(entry)
        imageData = qupath.imageData
        hints.install()
        if (project.labels.findAll { it.id }.size() == 1) targetChoice.editor.text = project.labels.find { it.id }.name
        configureModel()
        submissionRegions = new ArrayList((List)(imageData.getProperty('MONAILabel.SubmissionRegions') ?: []))
        wholeImageDraft = imageData.getProperty('MONAILabel.WholeImageDraft') == true
        classificationCategories = (List)(imageData.getProperty('MONAILabel.ClassificationCategories') ?: ['Tumor', 'Immune', 'Stromal'])
        if (imageData.server.width != asset.spatial_shape[1] || imageData.server.height != asset.spatial_shape[0])
            throw new IllegalArgumentException('QuPath image dimensions differ from the backend sample.')
        imageData.setImageType(ImageData.ImageType.BRIGHTFIELD_H_E)
        controls()
        log('Connected to ' + project.name + ' · ' + asset.name + '. ' +
            (reviewMode ? 'Review the annotation, make corrections and submit Good or Needs changes.' : 'Ask me to annotate a region or the full image. You can name a model in your message.') + ' Ctrl+Enter sends.')
        if (restoring && imageData.getProperty('MONAILabel.BaseRevision') != null) {
            int savedRevision = (int)imageData.getProperty('MONAILabel.BaseRevision')
            if (savedRevision != asset.revision) {
                asset = new HashMap(asset); asset.revision = savedRevision
                log('This draft is based on an older backend revision. Submission will require resolving that conflict.')
            }
            proposalId = (String)imageData.getProperty('MONAILabel.ProposalID')
        }
        if (restoring) log('Restored the saved QuPath draft; backend revision is ' + asset.revision + '.')
        else if (asset.annotation_id) {
            background({ backend.request('/api/annotations/' + asset.annotation_id + '/mask.bin', null, true) },
                { mask -> applyMask((byte[])mask); saveDraft(); log('Loaded submitted annotation revision ' + asset.revision + '.') })
        }
    }
    void checkImage() {
        if (qupath.imageData != imageData) throw new IllegalStateException('Return to the connected sample before using this assistant.')
    }
    void remember() {
        undo.add(new ArrayList(imageData.hierarchy.annotationObjects)); if (undo.size() > 20) undo.remove(0)
        redo.clear()
    }
    void replaceObjects(Collection objects) {
        imageData.hierarchy.removeObjects(new ArrayList(imageData.hierarchy.annotationObjects), false)
        imageData.hierarchy.addObjects(objects)
    }
    void applyMask(byte[] mask) {
        checkImage(); remember()
        def guides = imageData.hierarchy.annotationObjects.findAll { SelectionRegion.guide(it) }
        replaceObjects(guides + MaskObjects.decode(mask, imageData.server.width, imageData.server.height, project.labels))
    }
    void saveDraft() {
        checkImage(); imageData.setProperty('MONAILabel.BaseRevision', asset.revision)
        imageData.setProperty('MONAILabel.ProposalID', proposalId)
        imageData.setProperty('MONAILabel.SubmissionRegions', new ArrayList(submissionRegions))
        imageData.setProperty('MONAILabel.WholeImageDraft', wholeImageDraft)
        entry.saveImageData(imageData); qupath.project.syncChanges()
    }
    void send(String continuedMessage = null, Map prepared = null) {
        if (busy) return
        try {
            checkImage()
            String message = prepared?.viewer_scope ? "Update " + targetChoice.editor.text.trim() : continuedMessage ?: prompt.text.trim(); if (!message) return
            history.appendText('You: ' + message + '\n'); prompt.clear()
            Map classificationPlan = prepared?.plan
            SelectionRegion.markSelectedGuides(imageData)
            def selectedObjects = imageData.hierarchy.selectionModel.selectedObjects
            def region = selectedObjects.size() == 1 && selectedObjects.first().isAnnotation() && selectedObjects.first().ROI?.isArea() ? SelectionRegion.capture(imageData) : null
            def signature = MaskObjects.signature(imageData.hierarchy.annotationObjects)
            String inputSignature = backend.json.toJson(hints.objects)
            byte[] currentMask = null
            String maskError = null
            try { currentMask = MaskObjects.encode(SelectionRegion.labels(imageData), imageData.server.width, imageData.server.height, project.labels) }
            catch (IllegalArgumentException error) { maskError = error.message }
            def context = [asset_id: asset.id, base_revision: asset.revision,
                viewer_actions: ['set_interaction_mode', 'edit_spatial_prompts', 'clear_segments', 'classify_objects', 'undo', 'redo', 'save_draft'] + (canAnnotate && !reviewMode ? ['submit'] : []) + (canReview ? ['review_annotation'] : [])]
            if (prepared?.plan) context.classification = [base_revision: asset.revision, categories: classificationCategories, objects: classificationPlan.requests]
            if (region) context.image_region = region.scope
            context.image_tiling = [tile_size: 256, overlap: 32]
            context.interaction_target = targetChoice.editor.text.trim()
            context.interaction_mode = hints.mode
            context.spatial_objects = new ArrayList(hints.objects)
            if (modelChoice.value?.interaction) context.image_tiling = null
            def selection = modelChoice.selectionModel.selectedItem
            if (selection?.id) context.model_id = selection.id
            background({
                def reply = prepared?.viewer_scope ? backend.request('/api/projects/' + project.id + '/viewer-inference', [context: context, scope: prepared.viewer_scope]) : backend.request('/api/projects/' + project.id + '/assistant', [message: message, context: context, conversation_id: conversationId, request_id: UUID.randomUUID().toString().replace('-', ''), continue_tool: prepared?.continue_tool])
                if (reply.conversation_id) conversationId = reply.conversation_id
                Platform.runLater {
                    if (reply.data?.model_id) {
                        def model = modelChoice.items.find { it.id == reply.data.model_id }
                        if (model) modelChoice.selectionModel.select(model)
                    }
                    if (reply.data?.project && reply.data.project.id == project.id) {
                        project = reply.data.project
                        qupath.project.setPathClasses(project.labels.findAll { (it.id as int) != 0 }.collect { PathClass.fromString(it.name) })
                    }
                    log(reply.message)
                }
                if (!reply.job_id) return [reply: reply]
                jobId = reply.job_id; Platform.runLater { controls() }
                Map job
                while (true) {
                    job = backend.request('/api/jobs/' + jobId)
                    def current = job
                    Platform.runLater { status.text = current.status + ' · ' + current.progress_message }
                    if (!(job.status in ['queued', 'running'])) break
                    Thread.sleep(500)
                }
                jobId = null
                if (job.status != 'succeeded') throw new IOException(job.error ?: job.status)
                if (job.result.classification_id)
                    return [classification: backend.request('/api/classification-proposals/' + job.result.classification_id)]
                if (!job.result.proposal_id) return [reply: reply]
                def proposal = backend.request('/api/proposals/' + job.result.proposal_id)
                def mask = backend.request('/api/proposals/' + proposal.id + '/mask.bin', null, true)
                return [proposal: proposal, mask: mask]
            }, { result ->
                checkImage()
                if (result.reply?.data?.client_action in ['set_interaction_mode', 'edit_spatial_prompts']) {
                    hints.apply(result.reply.data); return
                }
                if (result.reply?.data?.client_action == 'review_annotation') {
                    def action = result.reply.data
                    if (action.asset_id != asset.id || action.base_revision != asset.revision ||
                        MaskObjects.signature(imageData.hierarchy.annotationObjects) != signature)
                        throw new IllegalStateException('The annotation changed while interpreting the review. Retry it.')
                    reviewDecision(action.verdict, action.comment ?: '')
                    return
                }
                if (result.reply?.data?.client_action in ['configure_classification', 'viewer_edit']) {
                    if (MaskObjects.signature(imageData.hierarchy.annotationObjects) != signature)
                        throw new IllegalStateException('Your objects changed while interpreting the prompt. Retry it.')
                    def data = result.reply.data
                    if (!(canAnnotate || canReview) ||
                        (data.client_action == 'configure_classification' && !canAnnotate) ||
                        (data.operation == 'submit' && !canAnnotate))
                        throw new IllegalStateException('Permission for this viewer action is required.')
                    if (data.client_action == 'configure_classification') {
                        def plan = ClassificationObjects.prepare(imageData, project, data.selected_only as boolean, [])
                        def dialog = new TextInputDialog((data.categories ?: classificationCategories).join(', '))
                        dialog.initOwner(qupath.stage); dialog.title = 'Classification categories'
                        dialog.headerText = 'Name the categories for these objects'
                        dialog.contentText = 'Comma-separated categories (the model can abstain):'
                        def answer = dialog.showAndWait()
                        if (!answer.isPresent()) { log('Classification cancelled.'); return }
                        classificationCategories = answer.get().split(',').collect { it.trim() }.findAll { it }
                        if (classificationCategories.size() < 2 || classificationCategories.size() > 16 ||
                            classificationCategories.collect { it.toLowerCase() }.toSet().size() != classificationCategories.size())
                            throw new IllegalArgumentException('Choose 2–16 unique classification categories.')
                        imageData.setProperty('MONAILabel.ClassificationCategories', new ArrayList(classificationCategories))
                        send(message, [plan: plan, continue_tool: data.continue_tool])
                    } else {
                        if (data.project_id != project.id || data.asset_id != asset.id || data.base_revision != asset.revision)
                            throw new IllegalStateException('Viewer action belongs to another sample or revision.')
                        switch (data.operation) {
                            case 'undo': case 'redo':
                                def source = data.operation == 'undo' ? undo : redo
                                def destination = data.operation == 'undo' ? redo : undo
                                if (!source) { log('No assistant change to ' + data.operation + '.'); return }
                                destination.add(new ArrayList(imageData.hierarchy.annotationObjects))
                                replaceObjects(source.remove(source.size() - 1)); log('Restored the previous annotation objects.'); break
                            case 'save_draft': saveDraft(); log('Saved QuPath draft.'); break
                            case 'submit': submit(); break
                            default: throw new IllegalArgumentException('Unsupported viewer action.')
                        }
                    }
                    return
                }
                if (result.classification) {
                    if (MaskObjects.signature(imageData.hierarchy.annotationObjects) != signature)
                        throw new IllegalStateException('Your objects changed during classification. The result was not applied.')
                    def objects = ClassificationObjects.apply(imageData, project, asset, result.classification, classificationPlan, classificationCategories)
                    remember(); replaceObjects(objects)
                    def counts = result.classification.results.countBy { it.category ?: 'Unclassified' }
                    log('Candidate classifications: ' + counts.collect { name, count -> name + ': ' + count }.join(', ') +
                        '. Inspect and edit native object classes. Say "undo" to restore, or save your QuPath draft. Classification review/training is not yet supported by the backend.')
                    return
                }
                if (result.reply?.data?.client_action == 'clear_segments') {
                    if (!(canAnnotate || canReview)) throw new IllegalStateException('Annotation or review permission is required.')
                    if (MaskObjects.signature(imageData.hierarchy.annotationObjects) != signature)
                        throw new IllegalStateException('Your objects changed while requesting the edit. Retry the clear command.')
                    def clearRegion = result.reply.data.image_region != null ? region : null
                    def objects = SegmentationEdits.clear(imageData, project, asset, result.reply.data, clearRegion, backend.json)
                    if (objects == new ArrayList(imageData.hierarchy.annotationObjects)) {
                        log('No matching segments to clear.'); return
                    }
                    remember(); replaceObjects(objects)
                    if (region && objects.contains(region.object))
                        imageData.hierarchy.selectionModel.setSelectedObject(region.object)
                    log('Cleared the requested segments. Selection guides are preserved. Say "undo" to restore them; save your draft to keep this edit.')
                    return
                }
                if (result.proposal) {
                    if (backend.json.toJson(hints.objects) != inputSignature) throw new IllegalStateException('Input hints changed. Run Update again; your draft is preserved.')
                    if (MaskObjects.signature(imageData.hierarchy.annotationObjects) != signature)
                        throw new IllegalStateException('Your objects changed during inference. The proposal was not applied.')
                    if (result.proposal.asset_id != asset.id || result.proposal.base_revision != asset.revision)
                        throw new IllegalStateException('Proposal belongs to another sample or revision.')
                    def selected = result.proposal.label_ids.collect { it as int }
                    if (currentMask == null) throw new IllegalStateException(maskError ?: 'Cannot encode the current annotations.')
                    def appliedRegion = result.proposal.image_region ? region : null
                    if (result.proposal.image_region && !region) throw new IllegalStateException('No matching viewer region.')
                    if (appliedRegion && !backend.json.toJsonTree(result.proposal.image_region).equals(backend.json.toJsonTree(region.scope + [runs: region.scope.runs])))
                        throw new IllegalStateException('Returned proposal scope differs from the requested selection.')
                    byte[] merged = SelectionRegion.merge(currentMask, (byte[])result.mask, selected, imageData.server.width, appliedRegion)
                    if (appliedRegion) {
                        def replacement = SelectionRegion.replacement(imageData, merged, project.labels, selected, appliedRegion)
                        remember(); replaceObjects(replacement)
                        imageData.hierarchy.selectionModel.setSelectedObject(region.object)
                        if (!submissionRegions.any { backend.json.toJsonTree(it).equals(backend.json.toJsonTree(region.scope)) })
                            submissionRegions.add(new HashMap(region.scope))
                        log('Applied nuclei/structure labels inside the selected region. The selection guide and outside annotations are preserved.')
                    } else {
                        applyMask(merged)
                        wholeImageDraft = true; submissionRegions.clear()
                        log('Added editable annotation objects. Inspect and correct them before submitting the complete annotation for review.')
                    }
                    proposalId = result.proposal.id
                }
            })
        } catch (Exception error) { log(error.message ?: error.toString()) }
    }
    void reviewDecision(String verdict, String suppliedComment = null) {
        if (busy || !canReview || !asset?.annotation_id) return
        try {
            checkImage()
            String comment = suppliedComment == null ? reviewComment.text : suppliedComment
            if (verdict == 'accepted') {
                byte[] mask = MaskObjects.encode(SelectionRegion.labels(imageData), imageData.server.width, imageData.server.height, project.labels)
                def metadata = [base_revision: asset.revision, covered_labels: project.labels.collect { it.id }, comment: comment]
                String header = backend.json.toJson(metadata)
                // JSON header values must be ASCII even for non-English comments.
                header = header.toCharArray().collect { ch -> ((int)ch) > 127 ? String.format('\\u%04x', (int)ch) : ch.toString() }.join('')
                background({ backend.request('/api/assets/' + asset.id + '/review-complete', mask, false, ['X-MONAILABEL-REVIEW': header]) }, { annotation ->
                    asset = asset + [revision: annotation.revision, annotation_id: annotation.id]
                    proposalId = null; saveDraft(); reviewComment.clear(); controls()
                    log('Review saved: Good · revision ' + annotation.revision + '. Open the next pending case in the web review queue.')
                })
            } else {
                background({ backend.request('/api/annotations/' + asset.annotation_id + '/decision', [verdict: 'changes_requested', comment: comment]) }, { decision ->
                    reviewComment.clear(); log('Review saved: Needs changes · revision ' + decision.revision + '. Open the next pending case in the web review queue.')
                })
            }
        } catch (Exception error) { log(error.message ?: error.toString()) }
    }
    void submit() {
        if (reviewMode) { log('Use Good or Needs changes to submit your review and corrections.'); return }
        if (busy || !canAnnotate) return
        try {
            checkImage()
            byte[] mask = MaskObjects.encode(SelectionRegion.labels(imageData),
                imageData.server.width, imageData.server.height, project.labels)
            List regions = new ArrayList(submissionRegions)
            if (!wholeImageDraft && !regions) {
                SelectionRegion.markSelectedGuides(imageData)
                def selected = imageData.hierarchy.selectionModel.selectedObjects
                if (selected.size() == 1 && SelectionRegion.guide(selected.first()))
                    regions.add(SelectionRegion.capture(imageData).scope)
                else regions.addAll(savedRegions)
            }
            if (!wholeImageDraft && regions) {
                def metadata = [base_revision: asset.revision, regions: regions,
                    covered_labels: project.labels.collect { it.id }]
                if (proposalId) metadata.proposal_id = proposalId
                background({ backend.submitRegions(asset.id, metadata, mask) }, { annotation ->
                    asset = asset + [revision: annotation.revision, annotation_id: annotation.id]
                    savedRegions = (savedRegions + regions).unique { backend.json.toJson(it) }
                    proposalId = null; submissionRegions.clear(); saveDraft()
                    log('Submitted ' + regions.size() + ' region(s) for independent review. Unrelated regions are preserved.')
                })
                return
            }
            def params = 'base_revision=' + asset.revision + project.labels.collect { '&covered_labels=' + it.id }.join('')
            if (proposalId) params += '&proposal_id=' + proposalId
            background({ backend.request('/api/assets/' + asset.id + '/review-mask?' + params, mask) }, { annotation ->
                asset = new HashMap(asset); asset.revision = annotation.revision; asset.annotation_id = annotation.id
                proposalId = null; submissionRegions.clear(); wholeImageDraft = false
                saveDraft(); log('Submitted revision ' + annotation.revision + ' for reviewer approval.')
            })
        } catch (Exception error) { log(error.message ?: error.toString()) }
    }
}

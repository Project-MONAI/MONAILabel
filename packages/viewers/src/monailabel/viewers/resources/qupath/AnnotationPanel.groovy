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
    final ComboBox<String> modelChoice = new ComboBox<>()
    final Label status = new Label('Connecting…')
    final Button sendButton = new Button('Send prompt')
    final Button cancelButton = new Button('Cancel job')
    final Button submitButton = new Button('Submit complete annotation for review')
    final Button draftButton = new Button('Save QuPath draft')
    final executor = Executors.newSingleThreadExecutor { runnable ->
        def thread = new Thread(runnable, 'monailabel-qupath'); thread.daemon = true; thread
    }
    Backend backend
    Map project, asset
    List models = []
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
    final HBox reviewButtons = new HBox(8, goodButton, badButton, correctedButton)
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
        pane = new VBox(10, new HBox(10, new Label('Annotation assistant'), popButton), new Label('Model'), modelChoice, split,
            new HBox(8, sendButton, cancelButton), draftButton, submitButton, reviewComment, reviewButtons, status)
        pane.padding = new Insets(14); VBox.setVgrow(split, Priority.ALWAYS)
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
    void controls() {
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
        def root = Path.of(System.getenv('MONAILABEL_QUPATH_DATA'), pid, aid)
        Files.createDirectories(root)
        def suffix = asset.name.lastIndexOf('.') >= 0 ? asset.name.substring(asset.name.lastIndexOf('.')) : '.png'
        def image = root.resolve('source' + suffix)
        if (!Files.exists(image)) Files.write(image, (byte[])backend.request('/api/assets/' + aid + '/image', null, true))
        return [project: project, asset: asset, models: models, permissions: permissions,
            root: root, image: image]
    }
    void opened(Map data) {
        project = data.project; asset = data.asset; models = data.models
        canAnnotate = data.permissions.roles.any { it in ['manager', 'annotator'] }
        canReview = data.permissions.roles.any { it in ['manager', 'reviewer'] }
        modelChoice.items.setAll(['Automatic'] + models.collect { it.name })
        modelChoice.selectionModel.select(0)
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
        entry.saveImageData(imageData); qupath.project.syncChanges()
    }
    void send(String continuedMessage = null, Map prepared = null) {
        if (busy) return
        try {
            checkImage()
            String message = continuedMessage ?: prompt.text.trim(); if (!message) return
            history.appendText('You: ' + message + '\n'); prompt.clear()
            Map classificationPlan = prepared?.plan
            SelectionRegion.markSelectedGuides(imageData)
            def selectedObjects = imageData.hierarchy.selectionModel.selectedObjects
            def region = selectedObjects.size() == 1 && selectedObjects.first().isAnnotation() && selectedObjects.first().ROI?.isArea() ? SelectionRegion.capture(imageData) : null
            def signature = MaskObjects.signature(imageData.hierarchy.annotationObjects)
            byte[] currentMask = null
            String maskError = null
            try { currentMask = MaskObjects.encode(SelectionRegion.labels(imageData), imageData.server.width, imageData.server.height, project.labels) }
            catch (IllegalArgumentException error) { maskError = error.message }
            def context = [asset_id: asset.id, base_revision: asset.revision,
                viewer_actions: ['clear_segments', 'classify_objects', 'undo', 'redo', 'save_draft'] + (canAnnotate && !reviewMode ? ['submit'] : []) + (canReview ? ['review_annotation'] : [])]
            if (prepared) context.classification = [base_revision: asset.revision, categories: classificationCategories, objects: classificationPlan.requests]
            if (region) context.image_region = region.scope
            context.image_tiling = [tile_size: 256, overlap: 32]
            int selection = modelChoice.selectionModel.selectedIndex
            if (selection > 0) context.model_id = models[selection - 1].id
            background({
                def reply = backend.request('/api/projects/' + project.id + '/assistant', [message: message, context: context, conversation_id: conversationId, request_id: UUID.randomUUID().toString().replace('-', ''), continue_tool: prepared?.continue_tool])
                conversationId = reply.conversation_id
                Platform.runLater {
                    if (reply.data?.model_id) {
                        int index = models.findIndexOf { it.id == reply.data.model_id }
                        if (index >= 0) modelChoice.selectionModel.select(index + 1)
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
                    def objects = SegmentationEdits.clear(imageData, project, asset, result.reply.data, region, backend.json)
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
                        log('Applied nuclei/structure labels inside the selected region. The selection guide and outside annotations are preserved.')
                    } else {
                        applyMask(merged)
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
            def params = 'base_revision=' + asset.revision + project.labels.collect { '&covered_labels=' + it.id }.join('')
            if (proposalId) params += '&proposal_id=' + proposalId
            background({ backend.request('/api/assets/' + asset.id + '/review-mask?' + params, mask) }, { annotation ->
                asset = new HashMap(asset); asset.revision = annotation.revision; asset.annotation_id = annotation.id
                proposalId = null; saveDraft(); log('Submitted revision ' + annotation.revision + ' for reviewer approval.')
            })
        } catch (Exception error) { log(error.message ?: error.toString()) }
    }
}

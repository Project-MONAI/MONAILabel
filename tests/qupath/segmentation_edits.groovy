// Run with an installed, current extension: QuPath script tests/qupath/segmentation_edits.groovy
import java.awt.image.BufferedImage
import qupath.lib.images.ImageData
import qupath.lib.images.servers.ImageServerProvider
import qupath.lib.objects.PathObjects
import qupath.lib.objects.classes.PathClass
import qupath.lib.regions.ImagePlane
import qupath.lib.roi.ROIs
import org.monailabel.qupath.Backend
import org.monailabel.qupath.MaskObjects
import org.monailabel.qupath.SelectionRegion
import org.monailabel.qupath.SegmentationEdits

def file = File.createTempFile('monailabel-clear-', '.png')
def data
try {
    javax.imageio.ImageIO.write(new BufferedImage(64, 48, BufferedImage.TYPE_INT_RGB), 'PNG', file)
    data = new ImageData(ImageServerProvider.buildServer(file.toString(), BufferedImage))
    def plane = ImagePlane.getDefaultPlane()
    def object = { x, y, w, h, name -> PathObjects.createAnnotationObject(
        ROIs.createRectangleROI(x, y, w, h, plane), name == null ? null : PathClass.fromString(name)) }
    def guide = object(20, 15, 16, 12, null)
    def nucleus = object(22, 17, 3, 3, 'Nuclei')
    def outside = object(2.25, 3.25, 4, 5, 'Nuclei')
    def crossed = object(18, 16, 5, 4, 'Nuclei')
    def tissue = object(30, 17, 3, 3, 'Tissue')
    def unrelated = object(50, 40, 3, 3, null)
    def objects = [guide, nucleus, outside, crossed, tissue, unrelated]
    data.hierarchy.addObjects(objects)
    data.hierarchy.selectionModel.setSelectedObject(guide)
    def region = SelectionRegion.capture(data)
    def project = [id:'project', labels:[[id:0,name:'Background'],[id:1,name:'Nuclei'],[id:2,name:'Tissue']]]
    def asset = [id:'asset', revision:3]
    def action = [project_id:project.id,asset_id:asset.id,base_revision:3,label_ids:[1],image_region:null]
    def remaining = SegmentationEdits.clear(data, project, asset, action, null, Backend.json)
    assert remaining.toSet() == [guide,tissue,unrelated].toSet()
    assert data.hierarchy.annotationObjects.toSet() == objects.toSet() // Planning has no side effects.
    remaining = SegmentationEdits.clear(data, project, asset, action + [label_ids:[1,2]], null, Backend.json)
    assert remaining.toSet() == [guide,unrelated].toSet()

    action.image_region = region.scope + [runs:region.scope.runs]
    remaining = SegmentationEdits.clear(data, project, asset, action, region, Backend.json)
    assert remaining.containsAll([guide,outside,tissue,unrelated])
    assert !remaining.contains(nucleus) && !remaining.contains(crossed)
    def valid = { items -> items.findAll { it.pathClass != null } }
    def before = MaskObjects.encode(valid(objects),64,48,project.labels)
    def cleared = MaskObjects.encode(valid(remaining),64,48,project.labels)
    for (int y=0;y<48;y++) for (int x=0;x<64;x++) {
        int i=y*64+x
        boolean inside = x>=20 && x<36 && y>=15 && y<27
        assert cleared[i] == (inside && before[i]==1 ? 0 : before[i])
    }
    // Applying the plan and restoring the retained objects is the panel's undo operation.
    data.hierarchy.removeObjects(new ArrayList(data.hierarchy.annotationObjects),false)
    data.hierarchy.addObjects(remaining)
    data.hierarchy.removeObjects(new ArrayList(data.hierarchy.annotationObjects),false)
    data.hierarchy.addObjects(objects)
    assert Arrays.equals(before, MaskObjects.encode(valid(data.hierarchy.annotationObjects),64,48,project.labels))

    for (def invalid : [action + [base_revision:2], action + [asset_id:'other'],
                        action + [label_ids:[0]], action + [image_region:null]]) {
        boolean rejected = false
        try { SegmentationEdits.clear(data, project, asset, invalid, region, Backend.json) }
        catch (IllegalArgumentException | IllegalStateException expected) { rejected=true }
        assert rejected
    }
    println('QUPATH_SEGMENT_CLEAR_SCOPE_AND_UNDO_OK')
    data.server.close(); file.delete(); System.exit(0)
} catch(Throwable error) {
    error.printStackTrace(); data?.server?.close(); file.delete(); System.exit(1)
}

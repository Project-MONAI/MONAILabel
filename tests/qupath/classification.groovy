// Run with the current extension: QuPath script tests/qupath/classification.groovy
import java.awt.image.BufferedImage
import qupath.lib.images.ImageData
import qupath.lib.images.servers.ImageServerProvider
import qupath.lib.objects.PathObjects
import qupath.lib.regions.ImagePlane
import qupath.lib.roi.ROIs
import org.monailabel.qupath.ClassificationObjects
import org.monailabel.qupath.MaskObjects

def file = File.createTempFile('monailabel-classify-', '.png')
def data
try {
    javax.imageio.ImageIO.write(new BufferedImage(64, 48, BufferedImage.TYPE_INT_RGB), 'PNG', file)
    data = new ImageData(ImageServerProvider.buildServer(file.toString(), BufferedImage))
    def project = [id:'project',labels:[[id:0,name:'Background'],[id:1,name:'Nuclei']]]
    def asset = [id:'asset',revision:3]
    byte[] mask = new byte[64*48]
    for (int y=4;y<9;y++) for (int x=3;x<8;x++) mask[y*64+x]=1
    for (int y=16;y<23;y++) for (int x=32;x<37;x++) mask[y*64+x]=1
    def original = MaskObjects.decode(mask,64,48,project.labels)
    def guide = PathObjects.createAnnotationObject(ROIs.createRectangleROI(1,1,40,30,ImagePlane.getDefaultPlane()))
    data.hierarchy.addObjects(original + [guide])
    def signature = MaskObjects.signature(data.hierarchy.annotationObjects)
    def plan = ClassificationObjects.prepare(data,project,false,[])
    assert plan.requests.size() == 2 // Split disconnected nuclei in a multipart mask object.
    assert MaskObjects.signature(data.hierarchy.annotationObjects) == signature
    def proposal = [id:'proposal',project_id:project.id,asset_id:asset.id,base_revision:3,
        results:[[object_id:plan.requests[0].id,category:'Epithelial'],[object_id:plan.requests[1].id,category:null]]]
    def result = ClassificationObjects.apply(data,project,asset,proposal,plan,['Epithelial','Immune'])
    assert result.contains(guide)
    def classified = result.findAll { it.pathClass != null }
    assert classified.size() == 2
    assert classified.collect { MaskObjects.labelName(it) }.toSet() == ['Nuclei'].toSet()
    assert classified.collect { it.pathClass.toString() }.toSet() == ['Nuclei: Epithelial','Nuclei: Unclassified'].toSet()
    assert Arrays.equals(mask,MaskObjects.encode(classified,64,48,project.labels))
    assert original.every { it.pathClass.toString() == 'Nuclei' } // Undo objects remain unmodified.
    data.hierarchy.removeObjects(new ArrayList(data.hierarchy.annotationObjects),false)
    data.hierarchy.addObjects(result)
    data.hierarchy.selectionModel.setSelectedObject(classified[0])
    assert ClassificationObjects.prepare(data,project,true,[]).requests.size() == 1
    for (def bad : [proposal + [base_revision:2], proposal + [asset_id:'other'],
        proposal + [results:[[object_id:'missing',category:'Immune']]]]) {
        boolean rejected=false
        try { ClassificationObjects.apply(data,project,asset,bad,plan,['Epithelial','Immune']) }
        catch(IllegalArgumentException | IllegalStateException expected) { rejected=true }
        assert rejected
    }
    data.hierarchy.removeObjects(new ArrayList(data.hierarchy.annotationObjects),false)
    data.hierarchy.addObjects(original + [guide])
    assert MaskObjects.signature(data.hierarchy.annotationObjects) == signature
    println('QUPATH_CLASSIFICATION_IDENTITY_MASK_AND_UNDO_OK')
    data.server.close(); file.delete(); System.exit(0)
} catch(Throwable error) {
    error.printStackTrace(); data?.server?.close(); file.delete(); System.exit(1)
}

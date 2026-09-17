import qupath.lib.gui.prefs.PathPrefs
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.StandardCopyOption

def userDirectory = PathPrefs.userPathProperty().get()
if (!userDirectory) {
    userDirectory = args[1]
    Files.createDirectories(Path.of(userDirectory))
    PathPrefs.userPathProperty().set(userDirectory)
}
def extensions = Path.of(userDirectory, 'extensions')
Files.createDirectories(extensions)
Files.copy(Path.of(args[0]), extensions.resolve('monailabel-qupath.jar'), StandardCopyOption.REPLACE_EXISTING)
println 'MONAI Label extension installed.'

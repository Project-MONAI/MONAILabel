import org.codehaus.groovy.control.CompilerConfiguration
import org.codehaus.groovy.control.CompilationUnit

def config = new CompilerConfiguration()
config.setTargetDirectory(new File(args[1]))
def unit = new CompilationUnit(config)
new File(args[0]).listFiles().findAll { it.name.endsWith('.groovy') && !(it.name in ['build.groovy', 'install.groovy']) }.each { unit.addSource(it) }
unit.compile()

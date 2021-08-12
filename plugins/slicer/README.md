## MONAILabel Plugin for 3D Slicer

<img src="https://raw.githubusercontent.com/Project-MONAI/MONAILabel/main/plugins/slicer/MONAILabel/Screenshots/1.png" width="800"/>

## Installing MONAILabel Plugin

Pick one of the following options to install MONAILabel Plugin for 3D Slicer

### Install 3D Slicer Preview Version with in-built Plugin

- Download and Install [3D Slicer](https://download.slicer.org/) **Preview version**
- Go to **View** -> **Extension Manager** -> **Active Learning** -> **MONAI Label**
- Install MONAI Label plugin
- _**Restart**_ 3D Slicer

> To update the plugin to latest version, you have to uninstall existing 3D Slicer version and download + install
> new preview version of 3D Slicer again.

### Install Plugin in Developer Mode

- `git clone git@github.com:Project-MONAI/MONAILabel.git`
- Open 3D Slicer: Go to **Edit** -> **Application Settings** -> **Modules** -> **Additional Module Paths**
- Add New Module Path: _<FULL_PATH>_/plugins/slicer/MONAILabel
- _**Restart**_ 3D Slicer

### Plugin Settings

User can change some default behavior for the plugin.  
Go to **Edit** -> **Application Settings** -> **MONAI Label**
<img src="https://raw.githubusercontent.com/Project-MONAI/MONAILabel/main/plugins/slicer/MONAILabel/Screenshots/3.png" width="400"/>

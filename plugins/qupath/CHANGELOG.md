## Version 0.3.0-SNAPSHOT
*In progress*

* Now separated from QuPath in its own repository!
* Log on/off OMERO servers with different credentials
* View the connection types of different OMERO servers and their status (public/private, connected/not connected)
* Browse any OMERO servers from within QuPath and open any project/dataset/image from the browser
* Retrieve OMERO project/dataset/image metadata (`More info..`)
* Advanced OMERO server search
* Import/send ROIs from/to the original image hosted on OMERO

> **Important!** This uses the OMERO web API: only RGB images are supported & are converted to JPEG before reaching QuPath. It is intended for viewing and annotating images; the JPEG compression may make it unsuitable for some kinds of analysis.
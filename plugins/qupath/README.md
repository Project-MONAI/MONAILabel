# QuPath MONAILabel extension

Download QuPath from: https://qupath.github.io/ and then install monailabel plugin using one of the following methods.

## From Binaries

Download [qupath-extension-monailabel-0.3.0.jar](https://github.com/Project-MONAI/MONAILabel/releases/download/data/qupath-extension-monailabel-0.3.0.jar)
and **drag the jar** on top of the running QuPath application window (black screen area) to install the extension.

## Building from source

You can build the latest extension jar using [OpenJDK 11](https://openjdk.java.net/) or later
with [gradle](https://gradle.org/install/)

```bash
gradle clean build
```

The output extension jar will be under `build/libs`. You can **drag the jar** file on top of QuPath to install the
extension.


## Using Plugin

- Make sure MONAILabel Server URL is correctly through `Preferences`.
- Open Sample Whole Slide Image in QuPath (which is shared as studies for MONAILabel server)
- Add/Select Rectangle ROI to run annotations using MONAI Label models.
- For Interative model (e.g. DeepEdit) you can choose to provide `Positive` and `Negative` points through Annotation panel.

![image](../../docs/images/qupath.jpg)

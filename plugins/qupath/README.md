<!--
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
-->

# QuPath MONAILabel extension

Download QuPath from: https://qupath.github.io/ and then install monailabel plugin using one of the following methods.

## From Binaries

Download [qupath-extension-monailabel-0.3.0.jar](https://github.com/Project-MONAI/MONAILabel/releases/download/data/qupath-extension-monailabel-0.3.0.jar)
and **drag the jar** on top of the running QuPath application window (black screen area) to install the extension.
If you are have previously installed then make sure to **_remove/uninstall_** the extension before updating.

> Development in progress.  If you are using latest MONAI Label, please build from source or use latest QuPath plugin available from [here](https://github.com/Project-MONAI/MONAILabel/releases/tag/data) and vice versa.

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

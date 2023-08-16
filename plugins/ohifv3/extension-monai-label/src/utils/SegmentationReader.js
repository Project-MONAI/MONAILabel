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

import nrrd from 'nrrd-js';
import pako from 'pako';
import readImageArrayBuffer from 'itk/readImageArrayBuffer';
import writeArrayBuffer from 'itk/writeArrayBuffer';
import config from 'itk/itkConfig';

const pkgJSON = require('../../package.json');
const itkVersion = pkgJSON.dependencies.itk.substring(1);
config.itkModulesPath = 'https://unpkg.com/itk@' + itkVersion; // HACK to use ITK from CDN

export default class SegmentationReader {
  static parseNrrdData(data) {
    let nrrdfile = nrrd.parse(data);

    // Currently gzip is not supported in nrrd.js
    if (nrrdfile.encoding === 'gzip') {
      const buffer = pako.inflate(nrrdfile.buffer).buffer;

      nrrdfile.encoding = 'raw';
      nrrdfile.data = new Uint16Array(buffer);
      nrrdfile.buffer = buffer;
    }

    const image = nrrdfile.buffer;
    const header = nrrdfile;
    delete header.data;
    delete header.buffer;

    return {
      header,
      image,
    };
  }

  static saveFile(blob, filename) {
    if (window.navigator.msSaveOrOpenBlob) {
      window.navigator.msSaveOrOpenBlob(blob, filename);
    } else {
      const a = document.createElement('a');
      document.body.appendChild(a);

      const url = window.URL.createObjectURL(blob);
      a.href = url;
      a.download = filename;
      a.click();

      setTimeout(() => {
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
      }, 0);
    }
  }

  // GZIP write not supported by nrrd-js (so use ITK save with compressed = true)
  static serializeNrrdCompressed(header, image, filename) {
    const nrrdBuffer = SegmentationReader.serializeNrrd(header, image);

    const reader = readImageArrayBuffer(null, nrrdBuffer, 'temp.nrrd');
    reader.then(function(response) {
      const writer = writeArrayBuffer(
        response.webWorker,
        true,
        response.image,
        filename
      );
      writer.then(function(response) {
        SegmentationReader.saveFile(new Blob([response.arrayBuffer]), filename);
        console.debug('File downloaded: ' + filename);
      });
    });
  }

  static serializeNrrd(header, image, filename) {
    let nrrdOrg = Object.assign({}, header);
    nrrdOrg.buffer = image;
    nrrdOrg.data = new Uint16Array(image);

    const nrrdBuffer = nrrd.serialize(nrrdOrg);
    if (filename) {
      SegmentationReader.saveFile(new Blob([nrrdBuffer]), filename);
      console.debug('File downloaded: ' + filename);
    }
    return nrrdBuffer;
  }
}

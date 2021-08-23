import axios from 'axios';

export default class MonaiLabelClient {
  constructor(server_url) {
    this.server_url = new URL(server_url);
  }

  async info() {
    let url = new URL('info', this.server_url);
    return await MonaiLabelClient.api_get(url.toString());
  }

  async segmentation(model, image) {
    return this.infer(model, image, {});
  }

  async deepgrow(model, image, foreground, background) {
    const params = {
      foreground: foreground,
      background: background,
    };

    return this.infer(model, image, params);
  }

  async infer(model, image, params, result_extension = '.nrrd') {
    let url = new URL('infer/' + encodeURIComponent(model), this.server_url);
    url.searchParams.append('image', JSON.stringify(image));
    url.searchParams.append('output', 'image');
    url = url.toString();

    if (result_extension) {
      params.result_extension = result_extension;
      params.result_dtype = 'uint16';
      params.result_compress = false;
    }

    return await MonaiLabelClient.api_post(
      url,
      params,
      null,
      true,
      'arraybuffer'
    );
  }

  async next_sample(stategy = 'random') {
    const url = new URL(
      'activelearning/' + encodeURIComponent(stategy),
      this.server_url
    ).toString();

    return await MonaiLabelClient.api_post(url, {}, null, false, 'json');
  }

  async save_label(params, image, label) {
    let url = new URL('datastore/label', this.server_url);
    url.searchParams.append('image', JSON.stringify(image));
    url = url.toString();

    const data = MonaiLabelClient.constructFormDataFromArray(
      params,
      label,
      'label',
      'label.bin'
    );

    return await MonaiLabelClient.aput(url, data, 'json');
  }

  static constructFormDataFromArray(params, data, name, fileName) {
    let formData = new FormData();
    formData.append('params', JSON.stringify(params));
    formData.append(name, data, fileName);
    return formData;
  }

  static constructFormData(params, files) {
    let formData = new FormData();
    formData.append('params', JSON.stringify(params));

    if (files) {
      if (!Array.isArray(files)) {
        files = [files];
      }
      for (let i = 0; i < files.length; i++) {
        formData.append(files[i].name, files[i].data, files[i].fileName);
      }
    }
    return formData;
  }

  static constructFormOrJsonData(params, files) {
    return files ? MonaiLabelClient.constructFormData(params, files) : params;
  }

  static api_get(url) {
    console.info('GET:: ' + url);
    return axios
      .get(url)
      .then(function(response) {
        console.info(response);
        return response;
      })
      .catch(function(error) {
        return error;
      })
      .finally(function() {});
  }

  static api_post(
    url,
    params,
    files,
    form = true,
    responseType = 'arraybuffer'
  ) {
    const data = form
      ? MonaiLabelClient.constructFormData(params, files)
      : MonaiLabelClient.constructFormOrJsonData(params, files);
    return MonaiLabelClient.apost(url, data, responseType);
  }

  static apost(url, data, responseType) {
    console.info('POST:: ' + url);
    return axios
      .post(url, data, {
        responseType: responseType,
        headers: {
          accept: ['application/json', 'multipart/form-data'],
        },
      })
      .then(function(response) {
        console.info(response);
        return response;
      })
      .catch(function(error) {
        return error;
      })
      .finally(function() {});
  }

  static api_put(url, params, files, form = false, responseType = 'json') {
    const data = form
      ? MonaiLabelClient.constructFormData(params, files)
      : MonaiLabelClient.constructFormOrJsonData(params, files);
    return MonaiLabelClient.aput(url, data, responseType);
  }

  static aput(url, data, responseType = 'json') {
    console.info('PUT:: ' + url);
    return axios
      .put(url, data, {
        responseType: responseType,
        headers: {
          accept: ['application/json', 'multipart/form-data'],
        },
      })
      .then(function(response) {
        console.info(response);
        return response;
      })
      .catch(function(error) {
        return error;
      });
  }
}

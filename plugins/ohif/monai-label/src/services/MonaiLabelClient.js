import axios from 'axios';

export default class MonaiLabelClient {
  constructor(server_url) {
    this.server_url = new URL(server_url);
  }

  getInfoURL() {
    let model_url = new URL('info/', this.server_url);
    return model_url.toString();
  }

  getLogsURL(lines = 200) {
    let log_url = new URL('logs/', this.server_url);
    log_url.searchParams.append('lines', String(lines));
    return log_url.toString();
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
    url.searchParams.append('image', image);
    url.searchParams.append('output', 'image');
    url = url.toString();

    if (result_extension) {
      params.result_extension = result_extension;
      params.result_dtype = 'uint16';
      params.result_compress = false;
    }

    const files = null;
    const responseType = 'arraybuffer';
    return await MonaiLabelClient.api_post(url, params, files, responseType);
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
    console.info('POST:: ' + url);
    const data = form
      ? MonaiLabelClient.constructFormData(params, files)
      : MonaiLabelClient.constructFormOrJsonData(params, files);

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
    console.info('PUT:: ' + url);
    const data = form
      ? MonaiLabelClient.constructFormData(params, files)
      : MonaiLabelClient.constructFormOrJsonData(params, files);

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

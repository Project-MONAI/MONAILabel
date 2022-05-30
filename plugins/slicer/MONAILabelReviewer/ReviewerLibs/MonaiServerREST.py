import requests
import logging
import datetime
from requests.structures import CaseInsensitiveDict
from urllib.parse import quote_plus

'''
MonaiServerREST provides the REST endpoints to the MONAIServer
'''

class MonaiServerREST:
    
    def __init__(self, serverUrl : str):
        self.serverUrl = serverUrl

    def getServerUrl(self) -> str:
        return self.serverUrl

    def getCurrentTime(self) -> datetime:
        return datetime.datetime.now()
        
    def requestDataStoreInfo(self) -> dict:
      download_uri = f"{self.serverUrl}/datastore/?output=all"
      
      try:
        response = requests.get(download_uri, timeout=5)
      except Exception as exception:
        logging.warning("{}: Request for DataStoreInfo failed due to '{}'".format(self.getCurrentTime(), exception))
        return None
      if (response.status_code != 200):
          logging.warning("{}: Request for datastore-info failed (url: '{}'). Response code is {}".format(self.getCurrentTime(), download_uri, response.status_code))
          return None
      
      return response.json()

    def getDicomDownloadUri(self, image_id : str) -> str: 
        download_uri = f"{self.serverUrl}/datastore/image?image={quote_plus(image_id)}"
        logging.info("{}: REST: request dicom image '{}'".format(self.getCurrentTime(), download_uri))
        return download_uri

    def requestSegmentation(self, image_id : str, tag = "final") -> requests.models.Response:
        download_uri = f"{self.serverUrl }/datastore/label?label={quote_plus(image_id)}&tag={quote_plus(tag)}"
        logging.info("{}: REST: request segmentation '{}'".format(self.getCurrentTime(), download_uri))
        
        try:
            response = requests.get(download_uri, timeout=5)
        except Exception as exception:
            logging.warning("{}: Segmentation request (image id: '{}') failed due to '{}'".format(self.getCurrentTime(), image_id, exception))
            return None
        if(response.status_code != 200):
            logging.warn("{}: Segmentation request (image id: '{}') failed due to response code: '{}'".format(self.getCurrentTime(), image_id, response.status_code))
            return None

        return response

    def checkServerConnection(self) -> bool:
        if not self.serverUrl:
            self.serverUrl = "http://127.0.0.1:8000"
        url = self.serverUrl.rstrip("/")
       
        try:
            response = requests.get(url, timeout=5)
        except Exception as exception:
            logging.warning("{}: Connection to Monai Server failed due to '{}'".format(self.getCurrentTime(), exception))
            return False
        if (response.status_code != 200):
            logging.warn("{}: Server connection Failed. (response code = {}) ".format(self.getCurrentTime(), response.status_code))
            return False
        
        logging.info("{}: Successfully connected to server (server url: '{}').".format(self.getCurrentTime(), url))
        return True

    def updateLabeInfo(self, image_id : str, params : str) -> int:
        url = f"{self.serverUrl}/datastore/updatelabelinfo?image={quote_plus(image_id)}"
        headers = CaseInsensitiveDict()
        headers["Content-Type"] = "application/x-www-form-urlencoded"
        headers["accept"] = "application/json"
        
        try:
            response = requests.put(url, headers=headers, data=params)
        except Exception as exception:
            logging.warning("{}: Update meta data (image id: '{}') failed due to '{}'".format(self.getCurrentTime(), image_id, exception))
            return None
        if (response.status_code != 200):
            logging.warn("{}: Update meta data (image id: '{}') failed due to response code = {}) ".format(self.getCurrentTime(), image_id, response.status_code))
            return response.status_code

        logging.info("{}: Meta data was updated successfully (image id: '{}').".format(self.getCurrentTime(), image_id))
        return response.status_code
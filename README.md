# MONAI-label
Draft Repository for MONAI Label


## UseCase
![alt text](https://www.websequencediagrams.com/cgi-bin/cdraw?lz=dGl0bGUgU2ltcGxlIFVzZWNhc2UKClJlc2VhcmNoZXItPgACCjogcGlwIGluc3RhbGwgbW9uYWlsYWJlbAAWGWRldmVsb3BzIG15X2FwcApub3RlIG92ZXIgAEkMdXNpbmcgdGVtcGxhdGUgYXBwIHByb3ZpZGVzIFxuMS4gdHJhaW5cbjIuIGluZmVyIGZvciBwcmVfABMFZWQgbW9kZWxzXG4zLiBhY3RpdmUgbGVhcm4gc3RyYXRlZ2llcwCBRA1NT05BSUxhYmVsOiBzdGFydF8AgUYFXwCBRwUoAIElBiwgZGF0YXNldCkAgX4OYWRpb2xvZ2lzdDogc2hhcmUAggEGIACCAwUgc2VydmVyIGlwCgAeCy0-M0RTbGljZXI6IGNvbmZpZ3VyZQCBfwwASAxGZXcAHgdzIGFyZSBhdgCCXwVsZSB0byBzZWxlY3QgXG5uZXR3b3JrLwCBWgd5L2RldmljZSBldGMuLgoAZwgAgVwObmV4dF9zYW1wbGUoAIIgBkxlYXJuaW5nKQoAgggKAIEhDAAnBiBkZXRhaWxzAEwLAIFGCmZldGNoAE4IbG9jYWwvcmVtb3RlKQBvF3J1bgCDPwZlbmNlKGRlZXBncm93LCBhdXRvc2VnbWVudAB3GACEZAcAgkEYcnJlY3RzAB8HAIFtFnN1Ym1pdCgAhS8FAIFnDgCEBwwAhGkFKG5ldwCCJwcpCgo&s=rose)



## Dry Run basic flow (without slicer)
- get /info/ (you get list of pre-trained models available) and other configs you can see for client
- post /activelearning/next_sample (you get sample image details)
- post /inference/segmentation_spleen (image is the one which u got in step 2)
- post /activelearning/save_label (not complete implemented yet.. but it will eventually save the label)
- post /train (manually start training.. otherwise plan is to add a config to support auto-start after N samples saved)
- If slicer is on remote it will use /download/ api to get it from server (u get the download link in sample details) (edited) 

# MONAI-label
Draft Repository for MONAI Label

## API Status
### App
- [x] List All Apps | GET http://127.0.0.1:8000/apps
- [x] App Details | GET http://127.0.0.1:8000/app/{app}
- [ ] Upload App | PUT http://127.0.0.1:8000/app/{app}
- [ ] Reload App | PATCH http://127.0.0.1:8000/app/{app}
- [ ] Delete App | DELETE http://127.0.0.1:8000/app/{app}

### Dataset
- [x] List All Datasets | GET http://127.0.0.1:8000/apps
- [x] Dataset Details | GET http://127.0.0.1:8000/app/{app}
- [ ] Upload Dataset | PUT http://127.0.0.1:8000/app/{app}
- [ ] Delete Dataset | DELETE http://127.0.0.1:8000/app/{app}

### AppEngine
- [ ] Run Inference | POST http://127.0.0.1:8000/inference/{app}
- [ ] Run Training | POST http://127.0.0.1:8000/inference/{app}

### Logs
- [x] Get Logs | GET http://127.0.0.1:8000/logs/
- [x] Get App Logs | POST http://127.0.0.1:8000/logs/{app}

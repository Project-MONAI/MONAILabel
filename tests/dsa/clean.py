import girder_client

apiUrl = "http://127.0.0.1:8080/api/v1"
apiToken = "mSQ5zHJk9j3FIi7MSM4LydSub0FLqm6eyqok399v"


gc = girder_client.GirderClient(apiUrl=apiUrl)
gc.authenticate(apiKey=apiToken)

jobList = gc.get("/job", parameters={"limit": 0})
for job in jobList:
    print(f"Remove job: {job['_id']}")
    gc.delete("/job/%s" % (job["_id"]))

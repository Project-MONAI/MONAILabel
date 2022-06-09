import json
import sys

import dateutil.parser
import girder_client
from tqdm import tqdm

apiUrl = "http://10.117.23.243:8080/api/v1"
apiToken = "mSQ5zHJk9j3FIi7MSM4LydSub0FLqm6eyqok399v"
resourceRoot = "collection/HistomicsTK Tests/TCGA/"
outputCSV = "summary.csv"

gc = girder_client.GirderClient(apiUrl=apiUrl)
gc.authenticate(apiKey=apiToken)

itemList = [
    ("TCGA-06-0130-01Z-00-DX1.0391b65f-4e1d-4444-abb0-e5804606d461.svs", 6000, 7350),
    ("TCGA-AA-3663-01Z-00-DX1.9AEDC003-2062-4876-8993-A5CEE4DDE1A9.svs", 10000, 9869),
    ("TCGA-HC-7080-01Z-00-DX1.c979be6a-e7c9-4840-8555-6f34499dd2bf.svs", 13429, 16750),
    ("TCGA-06-0138-01Z-00-DX3.2767efef-7d5f-40ff-9b36-5329d0fa6829.svs", 16000, 25013),
    ("TCGA-28-1746-01Z-00-DX1.06f187d2-b5e8-4b37-bb23-a707d0059944.svs", 24001, 26093),
    ("TCGA-DU-7010-01Z-00-DX1.542F36CC-9685-4780-94EB-B664CECFF09D.svs", 27888, 32276),
    ("TCGA-EM-A2CJ-01Z-00-DX1.D6F4716C-D6C7-4087-9B17-E1D89A3EEA8F.svs", 31723, 38601),
    ("TCGA-27-1831-01Z-00-DX4.b8a6fef5-9ba3-40b4-b32a-31485dbaa153.svs", 40001, 39991),
    ("TCGA-AP-A0LT-01Z-00-DX1.74C269EA-3118-4E65-AAFE-C1D186EAC207.svs", 51930, 38999),
    ("TCGA-HU-8244-01Z-00-DX1.EF15C805-A823-46EA-B737-2EC4A8C5C278.svs", 55775, 44812),
    ("TCGA-06-0195-01Z-00-DX2.5327662a-89b0-4297-ac6e-7af80f06cb3a.svs", 67917, 44541),
    ("TCGA-HT-7620-01Z-00-DX5.E88271BE-B362-4F17-96DF-31E421AA3143.svs", 75695, 47533),
    ("TCGA-FF-8047-01Z-00-DX1.75aa745c-bbe3-4869-a37b-c18ee50c14d5.svs", 76739, 55047),
    ("TCGA-BB-A6UO-01Z-00-DX1.11D049DC-EFC3-47EB-B390-A694BFD304A2.svs", 91632, 53467),
    ("TCGA-85-8072-01Z-00-DX1.3a0ad5a6-c93e-428c-94e7-809ceaf01ef1.svs", 80576, 69789),
    ("TCGA-V4-A9E7-01Z-00-DX1.465EFC95-3B6C-4836-A8BC-0A4F0BBFA601.svs", 81672, 78369),
    ("TCGA-C5-A3HD-01Z-00-DX1.11EECACD-371A-4B16-A21A-8E2A2258D3A9.svs", 135360, 53378),
    ("TCGA-OR-A5J6-01Z-00-DX1.C3F415F4-B679-433F-B8C2-33ED940272FB.svs", 91631, 88418),
    ("TCGA-26-1799-01Z-00-DX1.630B7217-0B01-4CDD-8ABF-0EC4CF293476.svs", 116825, 77258),
    ("TCGA-V1-A8WV-01Z-00-DX1.1419FDEA-BA02-42C4-9FF0-6F1F284BC6F3.svs", 113543, 88075),
    ("TCGA-AX-A1C4-01Z-00-DX1.237A4C5C-E87E-4904-83F2-B76196A247F0.svs", 123759, 89134),
    ("TCGA-DX-A6YR-01Z-00-DX1.8329CE17-C02B-4C56-8D02-54F40D95D624.svs", 137448, 88062),
    ("TCGA-DJ-A2PP-01Z-00-DX1.5BC2A5F2-1918-44E9-9544-1972974BA7BC.svs", 129472, 102134),
    ("TCGA-D8-A1JK-01Z-00-DX1.3190C919-A403-460D-9F6C-D2AB5FD3FD05.svs", 163743, 87914),
    ("TCGA-50-5068-01Z-00-DX2.0492A5C6-09CB-424B-BE20-10A1CBEA2E57.svs", 169320, 92215),
    ("TCGA-T7-A92I-01Z-00-DX1.3B036C1D-F8A7-475F-9830-C0972AD3889F.svs", 102912, 164096),
    ("TCGA-5N-A9KM-01Z-00-DX1.5197F750-D17F-459B-B74D-846F5F50F7B7.svs", 119040, 152832),
    ("TCGA-P3-A6T4-01Z-00-DX1.5DC1C4B4-7BB2-44AE-8D7A-FFFA3CB4BE63.svs", 203183, 97499),
    ("TCGA-T3-A92N-01Z-00-DX2.A08786DD-AF48-4551-BF71-E41C371C97C7.svs", 102656, 197888),
    ("TCGA-OL-A6VO-01Z-00-DX1.291D54D6-EBAF-4622-BD42-97AA5997F014.svs", 126464, 199936),
    ("TCGA-OL-A66J-01Z-00-DX1.661F7F70-E4D4-4875-B8C4-556F7927F3BA.svs", 130304, 247552),
    ("TCGA-02-0010-01Z-00-DX4.07de2e55-a8fe-40ee-9e98-bcb78050b9f7.svs", 32001, 38474),
]
itemNames = [n[0] for n in itemList]


def get_sec(time_str):
    h, m, s = time_str.split(":")
    return round(float(h) * 3600 + float(m) * 60 + float(s), 2)


jobList = gc.get(
    "/job",
    parameters={
        "types": json.dumps(
            ["projectmonai/monailabel-dsa:latest#MONAILabelAnnotation", "dsarchive/histomicstk:latest#NucleiDetection"]
        ),
        "limit": 0,
    },
)
jobSummary = {}

for job in tqdm(jobList):
    ca = job["kwargs"]["container_args"]
    model = (
        "segmentation_nuclei"
        if "segmentation_nuclei" in ca
        else ("deepedit_nuclei" if "deepedit_nuclei" in ca else "NucleiDetection")
    )
    try:
        file = "TCGA" + str(ca).split("TCGA")[1].split("svs")[0] + "svs"
        size = 1024 if "1024, 1024" in str(ca) else (4096 if "4096, 4096" in str(ca) else -1)
        stat2 = next(t for t in job["timestamps"] if t.get("status") == 2)
        stat3 = next(t for t in job["timestamps"] if t.get("status") == 3)

        if file not in itemNames:
            continue

        # print(f"JOB {size} x {model} :: {job['_id']}")
        jobFull = gc.get(f"/job/{job['_id']}")
        logs = []
        for l in jobFull["log"]:
            logs.extend(l.split("\n"))

        tiles = 0
        fg = 0
        detect = 0
        nuclei = 0
        for l in logs:
            if "Number of foreground tiles" in l:
                s = l.split("=")[1].strip()
                tiles = int(s.split(" ")[0])
            if "Tile foreground fraction computation time" in l:
                s = l.split("=")[1].strip()
                fg = get_sec(s) if ":" in s else float(s)
            if "Total Annotation Fetch time" in l or "Nuclei detection time" in l:
                s = l.split("=")[1].strip()
                detect = get_sec(s) if ":" in s else float(s)
            if "Number of nuclei = " in l:
                s = l.split("=")[1].strip()
                nuclei = int(s.split(" ")[0])
            if '"count":' in l:
                s = l.split(":")[1].strip().strip(",")
                nuclei = int(s)

    except Exception as e:
        continue

    duration = dateutil.parser.parse(stat3["time"]).timestamp() - dateutil.parser.parse(stat2["time"]).timestamp()

    # if model != 'NucleiDetection':
    #    duration /= 2

    duration = round(duration, 3)
    fg = round(fg, 3)
    detect = round(detect, 3)
    others = round(duration - fg - detect, 3)

    if file not in jobSummary:
        jobSummary[file] = {}
    if size not in jobSummary[file]:
        jobSummary[file][size] = {}
    if model not in jobSummary[file][size]:
        jobSummary[file][size][model] = {
            "Tiles": tiles,
            "FG": fg,
            "Detect": detect,
            "Others": others,
            "Total": duration,
            "Nuclei": nuclei,
        }

dataList = []
sizes = [1024, 4096, -1]
models = ["NucleiDetection", "segmentation_nuclei", "deepedit_nuclei"]
metrics = ["Tiles", "FG", "Detect", "Others", "Total", "Nuclei"]

csv_lines = []
header = ["name", "w", "h", "megapixels"]
for s in sizes:
    for m in models:
        for n in metrics:
            if n in ("Tiles", "FG") and s != -1:
                continue
            header.append(f"{s if s > 0 else 'wsi'}_{m}_{n}")

# print(header)
csv_lines.append(",".join(header))

for name, w, h in itemList:
    item = gc.get("resource/lookup", parameters={"path": f"{resourceRoot}{name}"})
    itemId = item["_id"]
    if len(sys.argv) == 2 and sys.argv[1] == "--purge":
        print(f"Purging annotations for {itemId} ({name})")
        gc.delete("annotation/item/%s" % (itemId))
        continue
    if not jobSummary.get(name):
        print(f"Not Ready yet!! Skip... {name}")
        continue

    row = [name, w, h, int(w * h // 1e6)]
    for s in sizes:
        for m in models:
            j = jobSummary.get(name, {}).get(s, {}).get(m, {})
            print(f"{name} => {s} => {m} => {j}")
            for n in metrics:
                if n in ("Tiles", "FG") and s != -1:
                    continue
                row.append(j.get(n, ""))

    # print(row)
    csv_lines.append(",".join([str(r) for r in row]))

with open(outputCSV, "w") as fp:
    for l in csv_lines:
        fp.write(l)
        fp.write("\n")

if len(sys.argv) == 2 and sys.argv[1] == "--purge":
    sys.exit(0)

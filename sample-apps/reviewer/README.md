# MONAILabel Reviewer - Lightweight Review App

A minimal, GPU-free MONAILabel application designed specifically for image segmentation review workflows.

## Overview

MONAILabel Reviewer is a streamlined application that enables radiologists and clinicians to:
- **Review and validate** segmentation masks created by AI models
- **Rate difficulty levels** of segmentations (Easy/Medium/Hard)
- **Add comments** with version control for annotations
- **Generate reports** with review statistics
- **Review existing labels** without loading MONAI Label AI tasks

**Key Features**:
- ✅ Zero AI model dependencies (no GPU required)
- ✅ Lightweight server with reviewer-specific aggregate endpoints under `/review`
- ✅ Version control for segmentations
- ✅ Support for multiple reviewers
- ✅ Review filtering and reporting

## Installation

### Prerequisites

```bash
# Python 3.8+
pip install monailabel
```

### Server Startup

#### Start Review Server (Lightweight Mode)

```bash
monailabel start_server \
  --app sample-apps/reviewer \
  --studies /path/to/review-dataset \
  --conf mode review
```

Recommended dataset layout:

```text
/path/to/review-dataset/
  case-001.nrrd
  case-002.nii.gz
  labels/
    final/
      case-001.seg.nrrd
      case-002.seg.nrrd
```

The server starts in review mode and keeps standard datastore routes for binary image and label access.

### Slicer Integration

1. Open **3D Slicer**
2. Go to **Edit > Application Settings > Modules**
3. Click **Add**
4. Navigate to `/workspace/MONAILabel/plugins/slicer/`
5. Select **MONAILabelReviewer** folder
6. Click **OK**

The reviewer module will now appear in the **MONAI Label** section.

## Quick Start

### 1. Connect to Server

1. Open 3D Slicer and load the **MONAILabelReviewer** module
2. Enter your MONAI Label server URL in the connection dialog
3. Click **Connect**
4. Click **Load** to load all images and segmentations

The reviewer UI normalizes the server URL before connecting, so accidental trailing `/` characters are removed automatically.

### 2. Review Segmentations

**Reviewer Mode** (Default):
- Use **Previous/Next** buttons to navigate images
- Use **Easy/Medium/Hard** buttons to rate difficulty
- Click **Approve** to mark segmentation as approved
- Click **Flag** to mark for revision
- Enter **Comments** about any issues or improvements
- Filters: Check which images to show (Approved/Flagged/Pending)

**Basic Mode** (Simplified):
- Skip advanced features
- Stream through segmentations quickly
- Simple navigation only

### 3. Edit Segmentations (Optional)

1. Click the segmentation dropdown to select version
2. Use Slicer's **Edit** tool to modify the mask
3. Click **Save as new version** to create a revised segmentation
4. **Approve** the new version

### 4. Generate Reports

```bash
# Generate JSON report
curl "http://localhost:8000/review/report?fmt=json" > review_report.json

# Generate CSV report
curl "http://localhost:8000/review/report?fmt=csv" > review_report.csv

# Generate HTML report
curl "http://localhost:8000/review/report?fmt=html" > review_report.html
```

## Architecture

### Server Components

```
monailabel/
├── endpoints/
│   └── datastore_review.py    # Reviewer aggregate endpoints

sample-apps/reviewer/
├── app.py                     # Lightweight review-only app
├── client.py                  # Simplified review client
├── main.py                    # App loader entrypoint
└── lib/
  └── config.py             # Review app configuration

plugins/slicer/MONAILabelReviewer/
└── ...                        # 3D Slicer reviewer module
```

### Review Data Model

Review metadata is stored through the standard datastore label-info APIs, and label binaries continue to use MONAI Label datastore version tags such as `final` and reviewer-created versions.

## API Reference

### Review Server Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/review/cases` | List reviewable images with summary metadata |
| `GET` | `/datastore/image` | Download image volume |
| `GET` | `/datastore/label` | Download segmentation (any version) |
| `GET` | `/datastore/label/info` | Get label metadata |
| `PUT` | `/datastore/label/info` | Update label metadata |
| `PUT` | `/datastore/label` | Save new/updated label |
| `GET` | `/review/versions` | List available versions |
| `GET` | `/review/report` | Generate review report |

### Request/Response Examples

#### List Images

```bash
GET /review/cases?status_filter=approved
Response:
{
  "summary": {
    "total": 100,
    "approved": 45,
    "flagged": 5,
    "pending": 50
  },
  "results": [
    {
      "id": "CT_abdomen_001",
      "name": "CT_abdomen_001.nii.gz",
      "status": "approved",
      "review_count": 2,
      "last_reviewed": "2024-01-15T10:30:00",
      "reviewer": "Dr. Smith"
    }
  ]
}
```

#### Update Label Info

```bash
PUT /datastore/label/info?label=CT_abdomen_001&tag=final
Content-Type: application/x-www-form-urlencoded

info={"status":"approved","level":"medium","comment":"Good segmentation","reviewer_name":"Dr. John Smith"}
```

#### Download Label (with Version Tag)

```bash
GET /datastore/label?label=CT_abdomen_001&tag=version_2
Response: Binary NIfTI/NRRD file
```

## Configuration

### Environment Variables

```bash
# Server Configuration
MONAI_LABEL_SERVER=http://localhost:8000
MONAI_LABEL_STUDIES=/path/to/images

# Reviewer Configuration
MONAI_LABEL_REVIEWER_NAME=Dr. John Smith
MONAI_LABEL_REVIEWER_EMAIL=dr.smith@example.com

MONAI_LABEL_REVIEW_MAX_HISTORY=10

# Mode
MONAI_LABEL_REVIEW_MODE=review
```

### Example Config File

Create `.env` file in the reviewer app directory:

```bash
cat > .env << EOF
MONAI_LABEL_SERVER=http://localhost:8000
MONAI_LABEL_REVIEWER_NAME=Dr. Sarah Johnson
MONAI_LABEL_REVIEWER_EMAIL=sarah.j@hospital.edu
EOF
```

## Workflow Examples

### Active Learning Workflow

1. **Initial Run**: AI model labels all images
2. **First Review Cycle**:
   - Connect to server
   - Review all images
   - Tag impossibly/hard cases as `approved`
   - Flag ambiguous cases for re-annotation
3. **Update Model**: Use flagged cases as training data
4. **Iterate**: Repeat until satisfactory model

## Troubleshooting

### Server Won't Start

```bash
# Check if app directory exists
ls sample-apps/reviewer/

# Validate python syntax
python -m py_compile sample-apps/reviewer/*.py

# Check for missing files
find sample-apps/reviewer -type f
```

### Client Can't Connect

```bash
# Test server connectivity
curl http://localhost:8000

# Check server logs
monailabel logs --tail 50

# Verify server URL
echo $MONAI_LABEL_SERVER
```

If remote image downloads fail while masks still load, check that the server URL is correct. Current reviewer builds also normalize a trailing `/` automatically before sending requests.

## Performance

- **Server Startup**: ~10 seconds (no AI models)
- **Image Load**: ~0.5 seconds per image
- **Label Download**: ~0.3 seconds
- **Memory Usage**: ~2GB (CPU only, no GPU)

## Limitations

- No AI model training (review only)
- Requires readable images in supported format
- Version control limited to annotated images
- Largest reports depend on available disk space

## Contributing

Contributions welcome! Areas for improvement:

- Additional review filters (date range, reviewer)
- Export to DICOM SEG
- Integration with PACS systems
- Multi-reviewer consensus workflow
- Automated quality metrics

## License

Apache License 2.0 - Same as MONAI Label

## Acknowledgements

- MONAI Consortium
- rAiDiance (original MONAILabelReviewer)
- 3D Slicer community

## Support

- Documentation: https://monai.readthedocs.io/projects/label/en/latest/
- GitHub Issues: https://github.com/Project-MONAI/MONAILabel/issues
- MONAI Discord: https://discord.gg/projectmonai

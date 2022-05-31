# Purpose
Radiologists have different levels of experience reading X-ray images.
Therefore, agreement of several radiologists on X-ray segmentations (especially in difficult cases) is required to increase the overall quality of a data set, which is then used to model a neural network.
MONAILABELReviewer is a tool for research groups to check the quality of the segmentation of their data sets.

![distributedWork](https://user-images.githubusercontent.com/30056369/158844144-94769304-cd4c-4630-ac6a-3dc124a9fc22.png)


# Import MONAILABELReviewer into 3DSlicer
1. Select "Edit"
2. Select "Application Settings"
3. Select "Modules"
4. Select "Add"
5. Within Browser select folder "MONAILabelReviewer"

![ImportReviewerIntoSlicer](https://user-images.githubusercontent.com/30056369/158845199-1f723b8b-a64e-4bdc-8596-974e952569d9.png)

# MonaiLabelReviewer UI
MonaiLabelReviewer has two usage modes, "Reviewer Mode" and "Basic Mode". The latter can be enabled by checking the checkbox next to "Basic Mode".
"Reviewer Mode" provides advanced features such as filer options and segmentation classification by difficulty (see subsection "UI in Reviewer Mode" for further description)

# UI in basic mode
1. If checkbox is selected, "basic mode" is activated (just for streaming through the segmentations)
2. Progress bar displays how many images have already been segmented in total.
3. Progress bar displays how many images have already been segmented by the selected annotator
4. Combobox: Selection of annotator (if option "All" is selected, the dataset includes segmentations of all annotators)
5. Slide bar: Displays currently index of image within the selected dataset
6. Lines which displays the meta data: imageId, annotator's name, date
7. Segmentation selection box: Hide/Show-option of segmentation layers

![UiBasicMode](https://user-images.githubusercontent.com/30056369/158844598-cd6a0ea9-2e2f-4da6-b2e7-7900c8e00b83.png)


#Required extensions in MonaiLabel

In order to persist the information created by MonaiLabelReviewer during the review process, an additional rest endpoint needs to be introduced into MonaiLabel.
In particular, the following methods (see below) need to be added in the datastore.py file.
(We will apply these changes in the monai community, so the workflow in MonaiLabelReviewer will be available without any additional manual changes in MonaiLabel.)

├── MONAILAIBEL
    ├── monailabel
        ├── endpoints
            ├── datastore.py

```
@router.put("/updatelabelinfo", summary="Update label info")
async def api_update_label_info(image: str, params: str = Form("{}")):
    return update_label_info(image, params)

def update_label_info(id: str, params: str = Form("{}")):
  save_params: Dict[str, Any] = json.loads(params) if params else {}
  instance: MONAILabelApp = app_instance()
  instance.datastore().update_image_info(id, save_params)
  return {}
```

# UI in Reviewer mode
1. If checkbox is not selected, "Reviewer Mode" is activated (the enables all feature for reviewing the segmentations)
2. Selection of reviewer's name or add new reviewer
3. Progress bar displays how many images have already been approved in total.
4. Progress bar displays how many images of selected Annotator have already been approved
5. Buttons (Easy, Medium, Hard) allows reviewer to classify the difficulty of segmentation to Easy, Medium, Hard
6. "Approve" Button: After the reviewer approves the segmentation, it can be included in the neural network modelling dataset
7. "Flag" Button allows the reviewer to mark a segmentation for later evaluation by another radiologist
8. Comment field: Reviewer can add comment into that box regarding the segmentation. If commented review is flagged additionally, the annotator can improve the segmentation according to comment
9. Filter options allows the reviewer to select a subset of image data set (not segmented, segmented, flagged, approved)

![UiReviewerMode](https://user-images.githubusercontent.com/30056369/158844810-27848c54-29d5-4d74-b1f2-27e38e92b150.png)


# Search  by Image Id
After entering a list of comma-separated image IDs in the left field, the right field displays a list of IDs of the corresponding found images.
That data set can be reviewed using the "Next"-"Previous"-Button.

![MonaiLabelReviewer SearchField](https://user-images.githubusercontent.com/30056369/159154537-0f97f004-0c61-4b63-947b-b7b55a3e61b1.png)
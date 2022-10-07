# Configure variables for bidsify functions

# Strings to exclude from filename
OMIT_REGEX = r"|".join(["posdisp", "scout", "localizer", "localiser", "3.*pl.*loc", "request_card", "screen_save"])

# Filter criteria for DICOM tags. Records with DICOM tags matching these regexes will be invalidated with the given
# comment. List of dictionaries, each with three keys:
#     *   tag :       column in which to match the regex
#     *   regex :     records where the tag matches this regex will be invalidated
#     *   comment :   for the 'comment' column, explaining why the record is invalid
TAG_FILTERS = [
    {"tag": "Modality", "regex": r"^(?!(MR)).*", "comment": "Not MR"},
    {"tag": "BodyPartExamined", "regex": r"PELVIS|TSPINE|LSPINE|BLADDER|HAND", "comment": "Non brain"},
    {
        "tag": "SeriesDescription",
        "regex": r"|".join(["posdisp", "scout", "localizer", "localiser", "3.*pl.*loc", "request_card", "screen_save"]),
        "comment": "SeriesDescription contains blacklisted string",
    },
]

# Default heuristics for converting DICOM tags to BIDS entities
HEURISTICS = [
    ({"tag": "SeriesDescription", "regex": r"t1"}, [{"entity": "suffix", "value": "T1w"}]),
    ({"tag": "SeriesDescription", "regex": r"mprage|mp-rage"}, [{"entity": "suffix", "value": "T1w"}]),
    ({"tag": "SeriesDescription", "regex": r"t1map"}, [{"entity": "suffix", "value": "T1map"}]),
    ({"tag": "SeriesDescription", "regex": r"t2"}, [{"entity": "suffix", "value": "T2w"}]),
    ({"tag": "SeriesDescription", "regex": r"t2\*"}, [{"entity": "suffix", "value": "T2star"}]),
    ({"tag": "SeriesDescription", "regex": r"flair|dark"}, [{"entity": "suffix", "value": "FLAIR"}]),
    (
        {"tag": "SeriesDescription", "regex": r"t2.*tirm"},
        [{"entity": "suffix", "value": "FLAIR"}, {"entity": "acquisition", "value": "TIRM"}],
    ),
    ({"tag": "SeriesDescription", "regex": r"pd"}, [{"entity": "suffix", "value": "PD"}]),
    ({"tag": "SeriesDescription", "regex": r"pd.*t2"}, [{"entity": "suffix", "value": "PDT2"}]),
    ({"tag": "SeriesDescription", "regex": r"angio"}, [{"entity": "suffix", "value": "angio"}]),
    ({"tag": "ContrastBolusAgent", "regex": r"^(?!\s*$).+"}, [{"entity": "contrast_enhancement", "value": "gad"}]),
    ({"tag": "SeriesDescription", "regex": r"dwi|dti|diff"}, [{"entity": "suffix", "value": "dwi"}]),
    (
        {"tag": "SeriesDescription", "regex": r"adc|apparent|average.dc"},
        [{"entity": "suffix", "value": "dwi"}, {"entity": "reconstruction", "value": "ADC"}],
    ),
    (
        {"tag": "SeriesDescription", "regex": r"trace"},
        [{"entity": "suffix", "value": "dwi"}, {"entity": "reconstruction", "value": "TRACEW"}],
    ),
    (
        {"tag": "SeriesDescription", "regex": r"fractional"},
        [{"entity": "suffix", "value": "dwi"}, {"entity": "reconstruction", "value": "FA"}],
    ),
]

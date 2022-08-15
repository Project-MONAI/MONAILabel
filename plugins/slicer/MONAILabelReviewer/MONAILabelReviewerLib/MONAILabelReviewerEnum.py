class SegStatus:
    def __init__(self):
        self.NOT_SEGMENTED = "not segmented"
        self.APPROVED = "approved"
        self.SEGMENTED = "segmented"
        self.FLAGGED = "flagged"


class Level:
    def __init__(self):
        self.EASY = "easy"
        self.MEDIUM = "medium"
        self.HARD = "hard"


class Label:
    def __init__(self):
        self.ORIGINAL = "original"
        self.FINAL = "final"
        self.VERSION = "version"

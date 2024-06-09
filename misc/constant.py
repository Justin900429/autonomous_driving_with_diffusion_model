from enum import Enum

COLOR_LIST = [
    (13, 36, 250),
    (23, 129, 226),
    (166, 230, 185),
    (146, 15, 39),
    (207, 214, 108),
    (209, 69, 61),
    (181, 221, 146),
    (244, 41, 112),
    (154, 162, 254),
    (174, 6, 136),
]


class GuidanceType(Enum):
    NO_GUIDANCE = 0
    FREE_GUIDANCE = 1
    CLASSIFIER_GUIDANCE = 2

from enum import Enum


class GuidanceType(Enum):
    NO_GUIDANCE = 0
    FREE_GUIDANCE = 1
    CLASSIFIER_GUIDANCE = 2

from typing import List, Tuple
from enum import Enum


class AnnotationResult(str, Enum):
    """
    Outcome of user annotation in the keyword validation step. (When keywords
    are being added.)
    """
    Correct = "correct"
    Incorrect = "incorrect"
    AccidentallyCorrect = "accidentaly_correct"


AnnotatedRandomSample = List[Tuple[str, AnnotationResult]]

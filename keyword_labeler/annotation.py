from enum import Enum
import random
import re
from typing import List, Optional, Tuple
import json
import string
from pathlib import Path
import datetime

from dataclasses import dataclass
from keyword_labeler.dataset import Dataset, WordIndex
from keyword_labeler.util.util import query_user_yes_no
from keyword_labeler.util.highlight import red_highlight, blue_highlight, green_highlight, yellow_highlight
from keyword_labeler.annotated_random_sample import AnnotationResult, AnnotatedRandomSample


# The Annotation class handles displaying random samples of the dataset to the
# user and collects feedback/annotation from the user.


@dataclass
class Annotation():
    annotated_random_sample: AnnotatedRandomSample
    keywords: List[str]
    overall_confidence: float

    def __init__(self, annotated_random_sample: AnnotatedRandomSample, keywords: List[str]):
        self.annotated_random_sample = annotated_random_sample
        self.keywords = keywords

        if annotated_random_sample:
            self.overall_confidence = sum(
                1
                for _, annotation
                in annotated_random_sample
                if annotation == AnnotationResult.Correct
            ) / len(annotated_random_sample)
        else:
            self.overall_confidence = 0

    @classmethod
    def combine(cls, annotation_1, annotation_2) -> "Annotation":
        return Annotation(
            annotation_1.annotated_random_sample + annotation_2.annotated_random_sample,
            list(set(annotation_1.keywords + annotation_2.keywords))
        )

    @classmethod
    def minimum_required_annotation_count(cls, occurence_ratio: float) -> int:
        if occurence_ratio < 0.002:
            return 1
        elif occurence_ratio < 0.005:
            return 3
        elif occurence_ratio < 0.01:
            return 5
        elif occurence_ratio < 0.02:
            return 7
        else:
            return 10

    def required_sample_count(self, word_index: WordIndex) -> int:
        return Annotation.minimum_required_annotation_count(
            word_index.get_occurence_ratio(self.keywords)
        )

    def get_sample_count(self) -> int:
        return len(self.annotated_random_sample)

    @classmethod
    def annotate(
        cls,
        training_set: Dataset,
        current_label: str,
        keywords: List[str],
        all_keywords_to_emphasize: List[str],
        sample_count: int,
        logging_path: Path,
    ) -> Optional["Annotation"]:
        lookuped_training_samples = training_set.get_texts_by_keywords(keywords)

        if not lookuped_training_samples:
            return

        random_sample = _get_random_sample(lookuped_training_samples)
        random_sample_emphasized_keywords = [
            _emphasize_keywords(text, keywords, all_keywords_to_emphasize)
            for text
            in random_sample
        ]

        print(blue_highlight(
            f"Are these texts categorized as '{current_label}' correctly? (y=yes / n=no / s=skip)"
        ))
        random_sample_annotation = _query_user_for_sample_annotation(
            random_sample_emphasized_keywords, sample_count, logging_path
        )
        print()

        annotated_random_sample = list(
            zip(random_sample[:len(random_sample_annotation)], random_sample_annotation)
        )

        return Annotation(annotated_random_sample, keywords)

    def get_overall_confidence(self):
        return self.overall_confidence

    def get_keywords(self) -> List[str]:
        return self.keywords

    def get_annotated_random_sample(self) -> AnnotatedRandomSample:
        return self.annotated_random_sample

    def to_string(
        self,
        text_group_keywords_to_emphasize: Optional[List[str]] = None,
        all_keywords_to_emphasize: Optional[List[str]] = None
    ):
        print(_emphasize_keywords(json.dumps(
            self.annotated_random_sample,
            sort_keys=True,
            indent=4,
            separators=(',', ': '),
        ), text_group_keywords_to_emphasize, all_keywords_to_emphasize))


def _emphasize_keywords(
        text: str,
        text_group_keywords_to_emphasize: Optional[List[str]] = None,
        all_keywords_to_emphasize: Optional[List[str]] = None
    ) -> str:
    if text_group_keywords_to_emphasize is not None:
        for keyword in text_group_keywords_to_emphasize:
            text = re.sub(rf"([{string.punctuation}\s]){keyword.lower()}([{string.punctuation}\s])", rf"\1{red_highlight(keyword)}\2", text.lower())

    if all_keywords_to_emphasize is not None:
        for keyword in all_keywords_to_emphasize:
            text = re.sub(rf"([{string.punctuation}\s]){keyword.lower()}([{string.punctuation}\s])", rf"\1{yellow_highlight(keyword)}\2", text.lower())

    return text


def _get_random_sample(training_samples: List[str]) -> List[str]:
    return random.sample(training_samples, k=len(training_samples))


def _query_user_for_sample_annotation(
    sample: List[str],
    count: int,
    logging_path: Path,
    start_index: int = 0,
) -> List[AnnotationResult]:
    result = []

    for index, text in enumerate(sample[start_index:][:count], start=start_index + 1):
        user_query_result = _query_user_for_annotation_result(f"{index}. / {len(sample)}: {text}")

        with logging_path.open('a') as fp:
            timestamp = datetime.datetime.now().strftime("%d. %B %Y %I:%M%p")
            fp.write(timestamp + " " + text + "\n")

        if user_query_result is None:
            break
        result.append(user_query_result)

    return result


def _query_user_for_annotation_result(query: str) -> Optional[AnnotationResult]:
    print(query, "(y=yes/n=no/s=skip)")

    correct = {"correct", "c", "yes", "y"}
    incorrect = {"incorrect", "i", "no", "n"}
    skip = {"skip", "s"}

    while True:
        choice = input().lower()
        if choice in correct:
            return AnnotationResult.Correct
        elif choice in incorrect:
            return AnnotationResult.Incorrect
        elif choice in skip:
            return None
        else:
            print("Please respond with 'correct/c/yes/y' or 'incorrect/i/no/n' or 'skip/s'")


def _query_user_should_keep_keyword(training_set: Dataset, keyword: str, all_keywords_to_emphasize, print_prefix: str, logging_path: Path) -> bool:
    random_sample_emphasized_keywords = [
        _emphasize_keywords(text, [keyword], all_keywords_to_emphasize)
        for text
        in _get_random_sample(
            training_set.get_texts_by_keywords([keyword])
        )
    ]
    if not random_sample_emphasized_keywords:
        return False

    def _print(index):
        print_string = (
            f"{print_prefix}{random_sample_emphasized_keywords[index]} " +
            "(y=yes=keep/n=no=discard/m=more)"
        )
        print(print_string)
        with logging_path.open('a') as fp:
            timestamp = datetime.datetime.now().strftime("%d. %B %Y %I:%M%p")
            fp.write(timestamp + " " + print_string + "\n")

    _print(0)
    index = 1

    keep = {"keep", "k", "yes", "y"}
    discard = {"discard", "d", "no", "n"}
    more_samples = {"more_samples", "more", "m"}

    while True:
        choice = input().lower()
        if choice in keep:
            return True
        elif choice in discard:
            return False
        elif choice in more_samples:
            try:
                if len(random_sample_emphasized_keywords) <= index:
                    return False
                _print(index)
                index += 1
            except IndexError:
                print("No more samples available.")
        else:
            print("Please respond with 'keep/k/yes/y' or 'discard/d/no/n' or 'more/m'")

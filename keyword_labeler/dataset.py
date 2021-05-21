from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict
import dataclasses
import datetime
import time
from pathlib import Path

from pydantic.dataclasses import dataclass

from keyword_labeler.util.loadable_json import LoadableJson
from keyword_labeler.annotated_random_sample import AnnotatedRandomSample


# An implementation of indexed annotated text datasets.
# Indexing is done on individual tokens.
# Supports reading and writing the fasttext file format, where:
#   - The dataset is in a text file.
#   - Each line of the file contains one text document.
#   - At the begining of each line may be present annotation labels, prefixed
#     with `__label__`. Annotation labels may be followed by a floating point
#     number between 0 and 1, which is the label's confidence value.


@dataclass
class DatasetLabel:
    label: str
    confidence: float


@dataclass
class DatasetRow:
    labels: List[DatasetLabel]
    text: str

    @classmethod
    def from_fasttext_line(cls, fasttext_line: str) -> "DatasetRow":
        # Only first occurence of label in fasttext_line is used.
        label_set = set()

        labels = []

        while fasttext_line[:9] == "__label__":
            fasttext_line_split = fasttext_line.split(" ", 2)

            try:
                if fasttext_line_split[0] not in label_set:
                    labels.append(DatasetLabel(
                        label=fasttext_line_split[0][9:],
                        confidence=float(fasttext_line_split[1]),
                    ))
                    fasttext_line = fasttext_line_split[2]

                    label_set.add(fasttext_line_split[0])
            except ValueError:
                labels.append(DatasetLabel(
                    label=fasttext_line_split[0][9:],
                    confidence=1,
                ))
                if len(fasttext_line_split) == 3:
                    fasttext_line = fasttext_line_split[1] + " " + fasttext_line_split[2]
                else:
                    fasttext_line = fasttext_line_split[1]

        return DatasetRow(labels=labels, text=fasttext_line)


@dataclass
class WordIndex(LoadableJson):
    # word_index: {"word" -> [occurence_1, occurence_2, ...]}
    word_index: Dict[str, Set[int]]
    dataset_size: int

    def __init__(self, data: List[DatasetRow]):
        self.dataset_size = len(data)

        self.word_index = defaultdict(lambda: set())
        for index, row in enumerate(data):
            words = row.text.split()
            for word in words:
                self.word_index[word].add(index)

    def get_occurence_count(self, keywords: List[str]) -> int:
        if not isinstance(keywords, list):
            raise TypeError("Must be list")

        try:
            return len(set.union(*[
                self.word_index[keyword]
                for keyword
                in keywords
                if keyword in self.word_index
            ]))
        except TypeError:
            # if keywords is empty or no keyword is in word_index
            return 0

    def get_occurence_ratio(self, keywords: List[str]) -> float:
        occurence_count = self.get_occurence_count(keywords)
        if occurence_count == 0:
            return float("nan")
        else:
            return occurence_count / self.dataset_size

    def get_occurences(self, keywords: List[str]) -> str:
        occurences = self.get_occurence_count(keywords)

        return f"{occurences / self.dataset_size * 100:>{4}.1f}% ({occurences}/{self.dataset_size})"

    def get_average_overlap_count(self, keyword: str, prediction_repeat_counts: Dict[int, int]) -> int:
        return sum(
            prediction_repeat_counts[index]
            for index
            in self.word_index[keyword]
        ) / self._get_overlap_cardinality(keyword, prediction_repeat_counts)

    def get_overlap_percentage(self, keywords: List[str], prediction_repeat_counts: Dict[int, int]) -> float:
        try:
            return (
                100 *
                self._get_overlap_cardinality(keywords, prediction_repeat_counts) /
                self.get_occurence_count(keywords)
            )
        except ZeroDivisionError:
            return 0

    def _get_overlap_cardinality(self, keywords: List[str], prediction_repeat_counts: Dict[int, int]) -> int:
        keywords_prediction_repeat_counts = self.get_keyword_prediction_repeat_counts(keywords)

        indices = set.union(*[self.word_index[keyword] for keyword in keywords])

        return sum(
            1 if prediction_repeat_counts[index] > keywords_prediction_repeat_counts[index] else 0
            for index
            in indices
        )

    def get_keyword_prediction_repeat_counts(self, keywords) -> Dict[int, int]:
        result = defaultdict(lambda: 0)

        # if a keyword is present multiple times, don't include this
        for keyword in set(keywords):
            for index in self.word_index[keyword]:
                result[index] += 1

        return result

    def get_keyword_predicted_labels(self, keywords: List[str], label: str) -> List[List[str]]:
        return [
            [label] if any(index in self.word_index[keyword] for keyword in keywords) else []
            for index
            in range(self.dataset_size)
        ]


@dataclass
class Dataset:
    data: List[DatasetRow]
    word_index: WordIndex

    @classmethod
    def from_fasttext_file(cls, file_path: Path) -> "Dataset":
        if not file_path.is_file():
            raise ValueError("Not an existing file:", file_path)

        fasttext_document = file_path.read_text().splitlines()

        data = [
            DatasetRow.from_fasttext_line(fasttext_line)
            for fasttext_line
            in fasttext_document
        ]

        return Dataset.from_data(data)

    @classmethod
    def from_data(cls, data: List[DatasetRow]) -> "Dataset":
        return Dataset(
            data=data,
            word_index=WordIndex(data),
        )

    def get_word_index(self) -> WordIndex:
        return self.word_index

    def get_texts_by_keywords(self, keywords) -> List[str]:
        try:
            indices = set.union(*[
                self.word_index.word_index[keyword]
                for keyword
                in keywords
            ])
        except TypeError:
            return []

        texts = self.get_texts()

        return [
            texts[index]
            for index
            in indices
        ]

    def get_annotated_fasttext_document(self) -> str:
        return "\n".join(
            (
                " ".join(
                    f"__label__{dataset_label.label} {dataset_label.confidence:.4f}"
                    for dataset_label
                    in dataset_row.labels
                    )
                + dataset_row.text
            )
            for dataset_row
            in self.data
        )

    def get_texts(self) -> List[str]:
        return [
            dataset_row.text
            for dataset_row
            in self.data
        ]

    def get_labels(self) -> List[List[str]]:
        return [
            [
                label.label
                for label
                in dataset_row.labels
            ]
            for dataset_row
            in self.data
        ]

    def filter_labels_by_confidence_threshold(self, confidence_threshold: float) -> "Dataset":
        return Dataset.from_data(
            [
                DatasetRow(
                    labels=[
                        DatasetLabel(
                            label=dataset_label.label,
                            confidence=dataset_label.confidence,
                        )
                        for dataset_label
                        in dataset_row.labels
                        if dataset_label.confidence > confidence_threshold
                        ],
                    text=dataset_row.text,
                )
                for dataset_row
                in self.data
            ]
        )

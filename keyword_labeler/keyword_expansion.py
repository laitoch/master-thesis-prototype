from typing import Dict, List, Optional, Tuple
import dataclasses
import datetime
import time
import itertools
import re
import string

from pydantic.dataclasses import dataclass

from keyword_labeler.util.loadable_json import LoadableJson
from keyword_labeler.dataset import Dataset, DatasetRow, DatasetLabel, WordIndex
from keyword_labeler.annotation import AnnotatedRandomSample, Annotation
from keyword_labeler.evaluator import MultiLabelEvaluator


# This file contains classes that represent keyword groups for all existing labels.
# The data can be saved and loaded in the json format.


class LabelExistsError(Exception):
    pass


@dataclass
class KeywordGroup:
    keywords: List[str]
    enabled: bool
    created_timestamp: int
    created_timestamp_readable: str
    sample_size: int
    annotated_random_sample: AnnotatedRandomSample
    confidence: float

    @classmethod
    def create(self, keywords, enabled, annotated_random_sample, confidence) -> "KeywordGroup":
        return KeywordGroup(
            keywords=keywords,
            enabled=enabled,
            sample_size=len(annotated_random_sample),
            created_timestamp=time.time(),
            created_timestamp_readable=datetime.datetime.now().strftime("%d. %B %Y %I:%M%p"),
            annotated_random_sample=annotated_random_sample,
            confidence=confidence,
        )


KeywordGroups = Dict[int, KeywordGroup]


@dataclass
class Label:
    keyword_groups: KeywordGroups


@dataclass
class KeywordExpansion(LoadableJson):
    labels: Dict[str, Label]
    max_keyword_group_id: int
    current_label: Optional[str] = None
    name: str = ""
    description: str = ""

    def predict(
        self,
        dataset: Dataset,
        allowed_keyword_group_ids: Optional[KeywordGroups] = None,
    ) -> Dataset:
        """
        @param dataset Uses unannotated Dataset only. Annotations are ignored.
        @return New Dataset objects with same text data and predictions made.
        """
        predictions = [
            self._predict(text, allowed_keyword_group_ids)
            for text
            in dataset.get_texts()
        ]

        return Dataset.from_data(predictions)

    def _predict(
        self,
        text: str,
        allowed_keyword_group_ids: Optional[KeywordGroups] = None,
    ) -> DatasetRow:
        predicted_labels = []

        for label_name, label in self.labels.items():
            prediction_confidences = []

            for keyword_group_id, keyword_group in label.keyword_groups.items():
                if ((
                        allowed_keyword_group_ids is None
                        and
                        keyword_group.enabled
                    ) or (
                        allowed_keyword_group_ids is not None
                        and
                        keyword_group_id in allowed_keyword_group_ids
                )):
                    for keyword in keyword_group.keywords:

                        # Tokenization
                        text_split = re.split("[\s" + string.punctuation + "]+", text.lower())
                        if keyword.lower() in text_split:
                            # prediction_confidences.append(keyword_group.confidence)
                            # We want prediction to work even if confidence is 0 (when we are lazy to annotate, we want this to work)
                            prediction_confidences.append(1)

            confidence = max([0] + prediction_confidences)

            predicted_labels.append(DatasetLabel(
                label=label_name,
                confidence=confidence,
            ))

        return DatasetRow(
            labels=predicted_labels,
            text=text,
        )

    def create_label(self, label: str) -> None:
        if label not in self.labels:
            self.labels[label] = Label(
                keyword_groups=[],
            )
        else:
            raise LabelExistsError()

    def get_labels(self) -> List[str]:
        return list(self.labels.keys())

    def add_keywords(
        self,
        label: str,
        keywords: List[str],
        annotated_random_sample: AnnotatedRandomSample,
        confidence: float,
        enabled: bool,
    ) -> int:
        """
        @return keyword_group_id
        """
        self.max_keyword_group_id = self.max_keyword_group_id + 1

        self.labels[label].keyword_groups[self.max_keyword_group_id] = KeywordGroup.create(
            keywords=keywords,
            enabled=enabled,
            annotated_random_sample=annotated_random_sample,
            confidence=confidence,
        )

        return self.max_keyword_group_id

    def enable_keyword_group(self, keyword_group_id: int) -> bool:
        return self._set_keyword_group(True, keyword_group_id)

    def disable_keyword_group(self, keyword_group_id: int) -> bool:
        return self._set_keyword_group(False, keyword_group_id)

    def _set_keyword_group(
        self,
        is_enabled: bool,
        keyword_group_id: int,
    ) -> bool:
        for label in self.labels.values():
            if keyword_group_id in label.keyword_groups:
                if label.keyword_groups[keyword_group_id].enabled != is_enabled:
                    label.keyword_groups[keyword_group_id].enabled = is_enabled
                    return True
        return False

    def remove_keyword_group(self, keyword_group_id: int) -> bool:
        for label in self.labels.values():
            if keyword_group_id in label.keyword_groups:
                del label.keyword_groups[keyword_group_id]
                return True
        return False

    def get_annotation(self, keyword_group_id: int) -> Optional[Annotation]:
        for label in self.labels.values():
            if keyword_group_id in label.keyword_groups:
                return Annotation(
                    label.keyword_groups[keyword_group_id].annotated_random_sample,
                    label.keyword_groups[keyword_group_id].keywords
                )

        return None

    def get_annotated_random_sample(self, keyword_group_id: int) -> Optional[AnnotatedRandomSample]:
        for label in self.labels.values():
            if keyword_group_id in label.keyword_groups:
                return label.keyword_groups[keyword_group_id].annotated_random_sample

        return None

    def get_keywords_for_label(self, label: str, enabled_keywords_only=True) -> List[str]:
        result = []

        for keyword_group in self.labels[label].keyword_groups.values():
            if not enabled_keywords_only or keyword_group.enabled:
                result += keyword_group.keywords

        return result

    def get_keywords_for_keyword_group(self, keyword_group_id: int) -> List[str]:
        for label in self.labels.values():
            if keyword_group_id in label.keyword_groups:
                return [
                    keyword
                    for keyword
                    in label.keyword_groups[keyword_group_id].keywords
                ]
        return []

    def print_keywords(self, label: str, training_dataset: Dataset, gold_data: Dataset, do_evaluate=False) -> str:
        keywords = self.get_label_keywords(label)

        train_word_index = training_dataset.get_word_index()

        prediction_repeat_counts = train_word_index.get_keyword_prediction_repeat_counts(keywords)

        gold_word_index = gold_data.get_word_index()
        gold_labels = gold_data.get_labels()

        result = ""

        for keyword_group_id, keyword_group in self.labels[label].keyword_groups.items():
            keywords = keyword_group.keywords
            evaluator = MultiLabelEvaluator(
                gold_labels=gold_labels,
                predicted_labels=gold_word_index.get_keyword_predicted_labels(keywords, label)
            )
            pr_rec = evaluator.get_precision_recall_f1().filter_by_tags(label)
            precision = pr_rec.result_matrix[1][2]
            recall = pr_rec.result_matrix[1][3]

            occurences = train_word_index.get_occurences(keywords)
            minimum_required_annotation_count = Annotation.minimum_required_annotation_count(train_word_index.get_occurence_ratio(keywords))

            if keyword_group.sample_size < minimum_required_annotation_count:
                confidence = "NaN "
            else:
                confidence = f"{100*keyword_group.confidence:>3.0f}%"

            keyword_group_stats = (
                f"keyword_group_id: "
                f"{keyword_group_id:<3}"
                f" enabled: "
                f"""{"True" if keyword_group.enabled else "False":<7}"""
                f"occurences:"
                f"{occurences:<22}"
                f"overlaps:"
                f"""{train_word_index.get_overlap_percentage(keywords, prediction_repeat_counts):>5.1f}%"""
                f"  confidence:"
                f"{confidence}"
                f"  annotated:"
                f"{keyword_group.sample_size:>3}/{minimum_required_annotation_count}"
            )
            if do_evaluate:
                keyword_group_stats += (
                    f" precision:"
                    f"{precision:<6}"
                    f" recall:"
                    f"{recall:<6}"
                )

            result += keyword_group_stats + "\n"

            result += (
                "---------------------------------------------------------------------\n"
                "       -keyword-               -occurences-         -overlaps- \n"
            )
            for keyword in keyword_group.keywords:
                occurences = train_word_index.get_occurences([keyword])
                result += (
                    "       "
                    f"{keyword:<18}"
                    f"{occurences:<27}"
                    f"""{train_word_index.get_overlap_percentage([keyword], prediction_repeat_counts):.1f}%"""
                    # "average overlapping keyword matches:"
                    # f"""{train_word_index.get_average_overlap_count(keyword, prediction_repeat_counts):.2f}"""
                    f"\n"
                )
            result += "\n"

        return result

    def get_label_keywords(self, label) -> List[str]:
        return list(itertools.chain(*[
            [keyword for keyword in keyword_group.keywords]
            for keyword_group
            in self.labels[label].keyword_groups.values()
        ]))

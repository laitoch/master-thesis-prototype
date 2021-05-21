#!/usr/bin/env python3

import argparse
import itertools
import json
from collections import Counter
from enum import Enum
from typing import List, Optional

from keyword_labeler.dataset import Dataset

"""
Multi-label multi-class classification evaluation.
"""


class CountOutputType(str, Enum):
    Count = "count"
    Ratio = "ratio"
    Percentage = "percentage"
    PercentageAndCount = "percentage_and_count"


def count_output_convert(count: float, total: int, count_output_type: CountOutputType) -> str:
    if count_output_type == CountOutputType.Ratio:
        try:
            return f"{count / total:.3f}"
        except ZeroDivisionError:
            return "  NaN "
    elif count_output_type == CountOutputType.Percentage:
        try:
            return f"{count / total * 100:.1f}%"
        except ZeroDivisionError:
            return "  NaN "
    elif count_output_type == CountOutputType.PercentageAndCount:
        try:
            return f"{count / total * 100:>{4}.1f}% ({count}/{total})"
        except ZeroDivisionError:
            return "  NaN "
    elif count_output_type == CountOutputType.Count:
        return str(count)
    else:
        raise TypeError(f"Invalid CountOutputType: {count_output_type}")


class ResultMatrix:
    # Fist row is the header
    result_matrix: List[List[str]]

    def __init__(self, result_matrix):
        self.result_matrix = result_matrix

        if not result_matrix:
            raise ValueError()

        result_matrix_width = len(result_matrix[0])
        for row in result_matrix:
            if len(row) != result_matrix_width:
                raise ValueError("Not all output table rows have the same width.")

    def filter_by_tags(self, acceptable_tags: List[str]) -> "ResultMatrix":
        tag_index = self._get_tag_index()

        filtered_result_matrix = [
            row
            for row
            in self.result_matrix[1:]
            if row[tag_index] in acceptable_tags
        ]

        return ResultMatrix([self.result_matrix[0]] + filtered_result_matrix)

    def to_csv(self):
        return "\n".join(",".join(cell for cell in row) for row in self.result_matrix)

    def to_string(self, cols: int = 28, print_tags: bool = True):
        if print_tags:
            tag_index = -1
        else:
            tag_index = self._get_tag_index()

        result = ""

        for index, name in enumerate(self.result_matrix[0]):
            if index != tag_index:
                result += f"{name:>{cols}}: " + " ".join(
                    f"{row[index]:<{len(self.result_matrix[i + 1][0]) + 1}}"
                    for i, row
                    in enumerate(self.result_matrix[1:])
                ) + "\n"

        return result

    def _get_tag_index(self):
        try:
            return self.result_matrix[0].index("Tag")
        except ValueError:
            raise ValueError(f"Row \"Tag\" not in ResultMatrix")


class MultiLabelEvaluator:
    gold_labels: List[List[str]]
    predicted_labels: List[List[str]]
    data_point_count: int
    labels: List[str]
    name: str
    description: str

    def __init__(self, gold_labels: List[List[str]], predicted_labels: List[List[str]], name: str = "", description: str = ""):
        self.gold_labels = gold_labels
        self.predicted_labels = predicted_labels

        def flatten(l):
            return [item for sublist in l for item in sublist]

        self.labels = list(set(flatten(gold_labels)).union(set(flatten(predicted_labels))))

        if len(gold_labels) != len(predicted_labels):
            raise ValueError("Different data point count for gold labels and predicted labels.")

        self.data_point_count = len(gold_labels)
        self.description = description
        self.name = name

    def get_data_point_count(self):
        return ResultMatrix([
            ["Data point #"],
            [f"{self.data_point_count}"],
        ])

    def get_exact_matches(self):
        exact_matches = 0

        for gold_labels, predicted_labels in zip(self.gold_labels, self.predicted_labels):
            gold_labels_set = set(gold_labels)
            predicted_labels_set = set(predicted_labels)

            if len(gold_labels_set) != len(predicted_labels_set):
                continue

            if gold_labels_set.union(predicted_labels_set) == gold_labels_set:
                exact_matches += 1

        return ResultMatrix([
            ["Exact match %"],
            [f"{100 * exact_matches / self.data_point_count:.1f}%"],
        ])

    def get_top_n_matches(self):
        top_1_matches = 0
        top_2_matches = 0

        for gold_labels, predicted_labels in zip(self.gold_labels, self.predicted_labels):
            gold_labels_set = set(gold_labels)

            if len(gold_labels_set) == 0:
                top_1_matches += 1
                top_2_matches += 1
                continue

            if len(predicted_labels) > 0 and predicted_labels[0] in gold_labels_set:
                top_1_matches += 1
                if (len(predicted_labels) > 1 and predicted_labels[1] in gold_labels_set) or len(gold_labels_set) < 2:
                    top_2_matches += 1

        return ResultMatrix([
            ["Top-1 match %", "Top-2 match %"],
            [
                f"{100 * top_1_matches / self.data_point_count:.1f}%",
                f"{100 * top_2_matches / self.data_point_count:.1f}%"
            ],
        ])

    def get_micro_average_f1(self):
        """Takes class imbalance into account, unlike macro-averaging."""
        per_label_confusion_matrix = self.get_per_label_confusion_matrix().result_matrix

        def _compute_accuracy(true_positives, true_negatives, total):
            try:
                accuracy = (true_positives + true_negatives) / total
                return f"{accuracy*100:.1f}%"
            except:
                return "  NaN "

        def _compute_precision(true_positives, false_positives):
            try:
                precision = true_positives / (true_positives + false_positives)
                return f"{precision*100:.1f}%"
            except:
                return "  NaN "

        def _compute_recall(true_positives, false_negatives):
            try:
                recall = true_positives / (true_positives + false_negatives)
                return f"{recall*100:.1f}%"
            except:
                return "  NaN "

        def _compute_f1(true_positives, false_positives, false_negatives):
            try:
                precision = true_positives / (true_positives + false_positives)
                recall = true_positives / (true_positives + false_negatives)
                f1 = 2 * precision * recall / (precision + recall)
                return f"{f1*100:.1f}%"
            except:
                return "  NaN "

        true_negatives = sum(float(x[1]) for x in per_label_confusion_matrix[1:])
        true_positives = sum(float(x[2]) for x in per_label_confusion_matrix[1:])
        false_negatives = sum(float(x[3]) for x in per_label_confusion_matrix[1:])
        false_positives = sum(float(x[4]) for x in per_label_confusion_matrix[1:])
        total = true_negatives + true_positives + false_negatives + false_positives

        accuracy = _compute_accuracy(true_positives, true_negatives, total)
        precision = _compute_precision(true_positives, false_positives)
        recall = _compute_recall(true_positives, false_negatives)
        f1 = _compute_f1(true_positives, false_positives, false_negatives)

        return ResultMatrix([
            ["Micro Accuracy", "Micro Precision", "Micro Recall", "Micro F1"],
            [accuracy, precision, recall, f1]
        ])

    def get_macro_average_f1(self):
        per_label_confusion_matrix = self.get_per_label_confusion_matrix().result_matrix

        def _compute_precision(true_positives, false_positives):
            try:
                precision = true_positives / (true_positives + false_positives)
                return precision
            except:
                return 1

        def _compute_recall(true_positives, false_negatives):
            try:
                recall = true_positives / (true_positives + false_negatives)
                return recall
            except:
                return 1

        def _compute_f1(true_positives, false_positives, false_negatives):
            try:
                precision = true_positives / (true_positives + false_positives)
                recall = true_positives / (true_positives + false_negatives)
                f1 = 2 * precision * recall / (precision + recall)
                return f1
            except:
                return 1

        return ResultMatrix([["Macro F1", "Macro Precision", "Macro Recall"]] + [
            [
                "%.3f" % (sum(
                    _compute_f1(float(true_positives), float(false_positives), float(false_negatives))
                    for _tag, true_negatives, true_positives, false_negatives, false_positives
                    in per_label_confusion_matrix[1:]
                ) / len(per_label_confusion_matrix[1:])),
                "%.3f" % (sum(
                    _compute_precision(float(true_positives), float(false_positives))
                    for _tag, true_negatives, true_positives, false_negatives, false_positives
                    in per_label_confusion_matrix[1:]
                ) / len(per_label_confusion_matrix[1:])),
                "%.3f" % (sum(
                    _compute_recall(float(true_positives), float(false_negatives))
                    for _tag, true_negatives, true_positives, false_negatives, false_positives
                    in per_label_confusion_matrix[1:]
                ) / len(per_label_confusion_matrix[1:])),
            ]
        ])

    def get_jaccard_index(self):
        def _sample_jaccard_index(gold_labels: List[str], predicted_labels: List[str]) -> float:
            gold_labels_set = set(gold_labels)
            predicted_labels_set = set(predicted_labels)

            return (
                len(gold_labels_set.intersection(predicted_labels_set)) /
                len(gold_labels_set.union(predicted_labels_set))
            )

        jaccard_index = sum(
            _sample_jaccard_index(gold, predicted)
            for gold, predicted
            in zip(self.gold_labels, self.predicted_labels)
        ) / self.data_point_count

        return ResultMatrix([
            ["Jaccard index"],
            [f"{jaccard_index:.3f}"],
        ])

    def get_per_label_confusion_matrix(self, count_output_type: CountOutputType = CountOutputType.Count):
        per_label_confusion_matrix = {
            label: {
                "true_negatives": 0,
                "true_positives": 0,
                "false_negatives": 0,
                "false_positives": 0,
            }
            for label
            in self.labels
        }

        for gold_label, predicted_label in zip(self.gold_labels, self.predicted_labels):
            for label in self.labels:
                if label in predicted_label and label in gold_label:
                    per_label_confusion_matrix[label]["true_positives"] += 1
                elif label in predicted_label and label not in gold_label:
                    per_label_confusion_matrix[label]["false_positives"] += 1
                elif label not in predicted_label and label in gold_label:
                    per_label_confusion_matrix[label]["false_negatives"] += 1
                elif label not in predicted_label and label not in gold_label:
                    per_label_confusion_matrix[label]["true_negatives"] += 1
                else:
                    raise ValueError("")

        if count_output_type == CountOutputType.Ratio:
            result_matrix_header = ["Tag", "True Positive Rate", "False Negative Rate", "False Positive Rate", "True Negative Rate"]
        elif count_output_type == CountOutputType.Percentage:
            result_matrix_header = ["Tag", "True Positive %", "False Negative %", "False Positive %", "True Negative %"]
        else:
            result_matrix_header = ["Tag", "True Positive", "False Negative", "False Positive", "True Negative"]

        result_matrix = ResultMatrix(
            [
                result_matrix_header,
                *[
                    [
                        tag,
                        count_output_convert(confusion["true_positives"], self.data_point_count, count_output_type),
                        count_output_convert(confusion["false_negatives"], self.data_point_count, count_output_type),
                        count_output_convert(confusion["false_positives"], self.data_point_count, count_output_type),
                        count_output_convert(confusion["true_negatives"], self.data_point_count, count_output_type),
                    ]
                    for tag, confusion
                    in per_label_confusion_matrix.items()
                ],
            ]
        )

        return result_matrix

    def get_label_counts(self, count_output_type: CountOutputType = CountOutputType.PercentageAndCount):
        gold_counter = Counter(itertools.chain(*self.gold_labels))
        predicted_counter = Counter(itertools.chain(*self.predicted_labels))

        result_matrix = ResultMatrix(
            [
                ["Tag", "Gold", "Predicted"],
                *[
                    [
                        label,
                        count_output_convert(gold_counter[label], self.data_point_count, count_output_type),
                        count_output_convert(predicted_counter[label], self.data_point_count, count_output_type),
                    ]
                    for label
                    in self.labels
                ],
            ]
        )

        return result_matrix

    def get_precision_recall_f1(self):
        per_label_confusion_matrix = self.get_per_label_confusion_matrix().result_matrix

        def _compute_accuracy(true_positives, true_negatives, total):
            try:
                accuracy = (true_positives + true_negatives) / total
                return f"{accuracy*100:.1f}%"
            except:
                return "  NaN "

        def _compute_precision(true_positives, false_positives):
            try:
                precision = true_positives / (true_positives + false_positives)
                return f"{precision*100:.1f}%"
            except:
                return "  NaN "

        def _compute_recall(true_positives, false_negatives):
            try:
                recall = true_positives / (true_positives + false_negatives)
                return f"{recall*100:.1f}%"
            except:
                return "  NaN "

        def _compute_f1(true_positives, false_positives, false_negatives):
            try:
                precision = true_positives / (true_positives + false_positives)
                recall = true_positives / (true_positives + false_negatives)
                f1 = 2 * precision * recall / (precision + recall)
                return f"{f1*100:.1f}%"
            except:
                return "  NaN "

        return ResultMatrix([["Tag", "Accuracy", "Precision", "Recall", "F1"]] + [
            [
                tag,
                _compute_accuracy(
                    float(true_positives),
                    float(true_negatives),
                    float(true_positives) + float(true_negatives) + float(false_positives) + float(false_negatives),
                ),
                _compute_precision(float(true_positives), float(false_positives)),
                _compute_recall(float(true_positives), float(false_negatives)),
                _compute_f1(float(true_positives), float(false_positives), float(false_negatives)),
            ]
            for (
                tag,
                true_positives,
                false_negatives,
                false_positives,
                true_negatives,
            )
            in per_label_confusion_matrix[1:]
        ])

    def get_label_cardinality(self):

        def label_cardinality(labels_list):
            return sum(len(x) for x in labels_list) / len(labels_list)

        return ResultMatrix([
            ["Gold Label Cardinality", "Predicted Label Cardinality"],
            [
                f"{label_cardinality(self.gold_labels):.2f}",
                f"{label_cardinality(self.predicted_labels):.2f}"
            ],
        ])

    def get_oneliner(self) -> ResultMatrix:
        return ResultMatrix([
                    ["Name"] +
                    self.get_data_point_count().result_matrix[0] +
                    self.get_exact_matches().result_matrix[0] +
                    self.get_jaccard_index().result_matrix[0] +
                    self.get_top_n_matches().result_matrix[0] +
                    self.get_micro_average_f1().result_matrix[0] +
                    self.get_label_cardinality().result_matrix[0] +
                    ["Description"],
                    [self.name] +
                    self.get_data_point_count().result_matrix[1] +
                    self.get_exact_matches().result_matrix[1] +
                    self.get_jaccard_index().result_matrix[1] +
                    self.get_top_n_matches().result_matrix[1] +
                    self.get_micro_average_f1().result_matrix[1] +
                    self.get_label_cardinality().result_matrix[1] +
                    [self.description],
                ])

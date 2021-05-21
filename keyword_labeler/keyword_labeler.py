#!/usr/bin/env python3

import argparse
import cmd
import dataclasses
import json
import threading
from pathlib import Path
from typing import List, Optional
import logging
from datetime import datetime

from keyword_labeler.util.highlight import blue_highlight, green_highlight, red_highlight
from keyword_labeler.util.util import query_user_yes_no
from keyword_labeler.dataset import Dataset
from keyword_labeler.evaluator import CountOutputType, MultiLabelEvaluator
from keyword_labeler.keyword_expansion import KeywordExpansion, LabelExistsError, KeywordGroup
from keyword_labeler.keyword_similarity_search import KeywordSimilarity
from keyword_labeler.annotation import Annotation, _query_user_should_keep_keyword


# Main file of the Keyword Labeler CLI.


class KeywordLabelerShell(cmd.Cmd):
    prompt = "(no-label) "

    current_path: Path

    top_n: int = 40
    keyword_expansion: KeywordExpansion

    training_set: Dataset
    development_set: Dataset

    # Exit upon CTRL-D (=EOF)
    def do_EOF(self, _):
        return True

    def do_exit(self, _):
        return True

    def cmdloop_ignoring_keyboard_interrupt(self):
        doQuit = False
        while not doQuit:
            try:
                self.cmdloop()
                doQuit = True
            except KeyboardInterrupt:
                self.intro = None
                print()

    def __init__(self, project_name: str):
        super().__init__()

        self.current_path = Path(f"keyword-labeler-projects/{project_name}/")

        self.logging_path = self.current_path / "annotation_and_similarity.log"

        self.init_keyword_similarity_thread = threading.Thread(target=self._init_keyword_similarity, daemon=True)
        self.init_keyword_similarity_thread.start()

        self.keyword_expansion = KeywordExpansion(labels={}, max_keyword_group_id=0)

        self.training_set = Dataset.from_fasttext_file(Path("data/yelp.txt"))
        self.development_set = Dataset.from_fasttext_file(Path("data/new_test_set.txt"))
        # self.training_set = Dataset.from_fasttext_file(self.current_path / "training_set.txt")
        # self.development_set = Dataset.from_fasttext_file(self.current_path / "annotation.dev")

        self.do_load()
        self.do_set_label(self.keyword_expansion.current_label)

    def _init_keyword_similarity(self) -> None:
        self.keyword_similarity = KeywordSimilarity(Path("data/svd2vec_yelp.bin"), self.logging_path)
        # self.keyword_similarity = KeywordSimilarity(self.current_path / "svd2vec.bin", self.logging_path)

    def do_set_name(self, name: str) -> None:
        self.keyword_expansion.name = name

    def do_set_description(self, description: str) -> None:
        self.keyword_expansion.description = description

    def do_create_label(self, label: str) -> None:
        try:
            self.keyword_expansion.create_label(label)
            print(f"Created label {label}.")
        except LabelExistsError:
            print(f"Label {label} already exists.")

        self._autosave()

    def do_set_label(self, label: str) -> None:
        """Sets the current 'label'. Commands often require a label to be set."""
        if label == "":
            self.keyword_expansion.current_label = None
            self.prompt = "(no-label) "
            return

        existing_labels = self.keyword_expansion.get_labels()

        if label in existing_labels:
            self.keyword_expansion.current_label = label
            self.prompt = f"({label}) "
        else:
            print(f"Label {label} doesn't exist yet.")

    def complete_set_label(self, text, line, begidx, endidx):
        existing_labels = self.keyword_expansion.get_labels()

        if not text:
            completions = existing_labels[:]
        else:
            completions = [
                label
                for label in existing_labels
                if label.startswith(text)
            ]

        return completions

    def do_set_similarity_top_n(self, top_n: str) -> None:
        try:
            self.top_n = int(top_n)
        except ValueError:
            return

    def _do_similar_keywords(self, keywords: str) -> None:
        self.init_keyword_similarity_thread.join()

        keyword_search_results = self.keyword_similarity.search(keywords.split(), self.top_n)
        keyword_search_results.print(
            self.training_set.get_word_index(),
            self.keyword_expansion.get_keywords_for_label(
                self.keyword_expansion.current_label,
                enabled_keywords_only=False,
            ),
        )

        return keyword_search_results

    def do_similar_keywords(self, keyword: str) -> None:
        """
        Given a keyword, print a list of similar words. Use this for keyword
        expansion: If a keyword is a good indicator for a classification
        category, it is likely that words similar to the keyword will be good
        indicators as well.
        """
        self._do_similar_keywords(keyword)

    def do_remove_keyword_group(self, keyword_group_id):
        try:
            keyword_group_id_int = int(keyword_group_id)
        except ValueError:
            return

        if self.keyword_expansion.remove_keyword_group(keyword_group_id_int):
            print(f"Removed keyword group id {keyword_group_id}")

        self._autosave()

    def do_split_keyword_group(self, _) -> None:
        """
        Chosen keywords will be removed from a keyword group to a new
        keyword group. 10 new data samples should be labeled to compute
        confidences. The previously 10 labeled samples will be reused.
        """
        try:
            if self.keyword_expansion.current_label is None:
                print("Label must be set")
                return

            print("Enter keyword group id (this group will be split):")
            try:
                keyword_group_id = int(input())
            except ValueError:
                print("Invalid keyword group id.")
                return

            if keyword_group_id not in self.keyword_expansion.labels[self.keyword_expansion.current_label].keyword_groups:
                print(f"Keyword group id {keyword_group_id} does not exist.")
                return

            keyword_group = self.keyword_expansion.labels[self.keyword_expansion.current_label].keyword_groups[keyword_group_id]

            print("Enter keywords to remove from the group. A new group will be created containing the keywords:")
            new_keywords = input().split()

            for keyword in new_keywords:
                if keyword not in keyword_group.keywords:
                    print("Keyword {keyword} is not in the keyword group.")
                    return

            old_keywords = [
                keyword
                for keyword
                in keyword_group.keywords
                if keyword not in new_keywords
            ]

            needed_old_annotation_sample_count = Annotation.minimum_required_annotation_count(
                self.training_set.get_word_index().get_occurence_ratio(old_keywords)
            )
            needed_new_annotation_sample_count = Annotation.minimum_required_annotation_count(
                self.training_set.get_word_index().get_occurence_ratio(new_keywords)
            )

            new_group_annotation = Annotation.annotate(
                self.training_set,
                self.keyword_expansion.current_label,
                new_keywords,
                self.keyword_expansion.get_keywords_for_label(self.keyword_expansion.current_label, enabled_keywords_only=False),
                sample_count=needed_new_annotation_sample_count,
                logging_path=self.logging_path,
            )

            old_group_annotation = Annotation.annotate(
                self.training_set,
                self.keyword_expansion.current_label,
                old_keywords,
                self.keyword_expansion.get_keywords_for_label(self.keyword_expansion.current_label, enabled_keywords_only=False),
                sample_count=needed_old_annotation_sample_count,
                logging_path=self.logging_path,
            )

            keyword_group.annotated_random_sample = old_group_annotation.get_annotated_random_sample()
            keyword_group.keywords = old_group_annotation.get_keywords()
            keyword_group.confidence = old_group_annotation.get_overall_confidence()
            keyword_group.sample_size = old_group_annotation.get_sample_count()

            self.keyword_expansion.add_keywords(
                label=self.keyword_expansion.current_label,
                keywords=new_group_annotation.get_keywords(),
                enabled=True,
                annotated_random_sample=new_group_annotation.get_annotated_random_sample(),
                confidence=new_group_annotation.get_overall_confidence(),
            )
        except KeyboardInterrupt:
            pass

        self._autosave()

    def do_add_keywords(self, keywords: str) -> None:
        """
        Adds a keyword. The keyword is added to the current label. Texts
        containing the keyword will be classified as the current label.
        """
        try:
            if keywords == "":
                print("No keywords to add.")
                return

            if self.keyword_expansion.current_label is None:
                print("Label must be set.")
                return

            with self.logging_path.open('a') as fp:
                timestamp = datetime.now().strftime("%d. %B %Y %I:%M%p")
                fp.write(timestamp + " ^^^ " + keywords + "\n")

            existing_keywords = set(self.keyword_expansion.get_label_keywords(self.keyword_expansion.current_label))
            updated_keywords = [
                keyword
                for keyword
                in keywords.split()
                if keyword not in existing_keywords
            ]

            removed_keywords = [
                keyword
                for keyword
                in keywords.split()
                if keyword in existing_keywords
            ]

            print()

            if removed_keywords:
                print(red_highlight(f"The following keywords were removed from your group: ") + blue_highlight(f"""{", ".join(removed_keywords)}""") + ".")
                print("The keywords were removed because they are already present in existing keyword groups.")
                print("Tip: Use commands `split_keyword_group` and `remove_keyword_group` do modify existing keyword groups.")
                print()

            if not updated_keywords:
                print(red_highlight(f"Error: No keywords left."))
                return

            total_keyword_occurence = self.training_set.get_word_index().get_occurences(updated_keywords)
            occurence_ratio = self.training_set.get_word_index().get_occurence_ratio(updated_keywords)
            minimum_required_annotation_count = Annotation.minimum_required_annotation_count(occurence_ratio)

            print(blue_highlight("Creating keyword group containing keywords:"))
            print()
            print("    -keyword-                      -occurences-")
            for keyword in updated_keywords:
                occurences = self.training_set.get_word_index().get_occurences([keyword])
                print(f"     {keyword:<25} {occurences}")

            print()
            print(f"    Total keyword occurences: {total_keyword_occurence}")
            print()
            print(blue_highlight(f"Please annotate {minimum_required_annotation_count} samples:"))
            print(f"    If less then {minimum_required_annotation_count} samples are annotated, we consider this keyword group's confidence zero.")
            print()
            print(blue_highlight(f"Tips:"))
            print(f"    Skip annotation and create keyword group by pressing `s/skip`.")
            print(f"    Resume annotation of a created keyword group using the `further_annotate` command.")
            print(f"    Abort creating the keyword group by pressing `CTRL-C`.")
            print()
            print()
            print(green_highlight(f"Consider using pen-and-paper when annotating to take note of new potential keywords."))
            print()
            print()

            annotation = Annotation.annotate(
                self.training_set,
                self.keyword_expansion.current_label,
                keywords=updated_keywords,
                all_keywords_to_emphasize=self.keyword_expansion.get_keywords_for_label(self.keyword_expansion.current_label, enabled_keywords_only=False),
                sample_count=minimum_required_annotation_count,
                logging_path=self.logging_path,
            )
            if annotation is None:
                print(f"Error: Not enough data points for provided keywords: {updated_keywords}.")
                return

            confidence = annotation.get_overall_confidence()

            annotated_random_sample = annotation.get_annotated_random_sample()

            keyword_group_id = self.keyword_expansion.add_keywords(
                label=self.keyword_expansion.current_label,
                keywords=updated_keywords,
                annotated_random_sample=annotated_random_sample,
                confidence=confidence,
                enabled=True,
            )

            print()
            print(f"Created keyword group {keyword_group_id}.")
            print()
        except KeyboardInterrupt:
            pass

        self._autosave()

    def do_bulk_add_keywords(self, _) -> None:
        try:
            if self.keyword_expansion.current_label is None:
                print("Label must be set")
                return

            print("Enter keyword(s) for similarity search:")
            keywords = input()
            keyword_search_results = self._do_similar_keywords(keywords)

            def _query_user_for_int():
                while True:
                    try:
                        result = int(input())
                        if 0 <= result <= self.top_n:
                            return result
                    except ValueError:
                        pass
                    print(f"Please respond with a value: 1 <=  integer <= {self.top_n}")

            print("Enter Threshold; Words 1 to Threshold (inc.) will be possible to select as keywords:")
            threshold = _query_user_for_int()

            existing_keywords = set(self.keyword_expansion.get_label_keywords(self.keyword_expansion.current_label))
            keywords = [
                keyword
                for keyword
                in keywords.split() + keyword_search_results.select(threshold)
                if keyword not in existing_keywords
            ]

            print(f"Do you wish to confirm keywords one-by-one before adding them? (Y/n)")
            if query_user_yes_no(default=True):
                print(blue_highlight(
                    f"Will the following keywords categorize texts as '{self.keyword_expansion.current_label}' correctly?"
                    f" (y=yes=keep / n=no=discard / m=more)"
                ))
                updated_keywords = [
                    keyword
                    for index, keyword
                    in enumerate(keywords, start=1)
                    if _query_user_should_keep_keyword(
                        self.training_set,
                        keyword,
                        self.keyword_expansion.get_keywords_for_label(self.keyword_expansion.current_label, enabled_keywords_only=False),
                        f"{index}. / {len(keywords)}: ",
                        self.logging_path,
                    )
                ]
            else:
                updated_keywords = keywords

            self.do_add_keywords(" ".join(updated_keywords))
        except KeyboardInterrupt:
            pass

        self._autosave()

    def do_further_annotate(self, keyword_group_id: Optional[int] = None):
        try:
            keyword_group_id = int(keyword_group_id)
        except ValueError:
            print("Please enter a valid keyword group id.")
            return

        annotation = self.keyword_expansion.get_annotation(keyword_group_id)

        if annotation is None:
            print("Please enter an existing keyword group id.")
            return

        required_sample_count = annotation.required_sample_count(self.training_set.get_word_index())
        if annotation.get_sample_count() < required_sample_count:
            sample_count = required_sample_count - annotation.get_sample_count()
        else:
            sample_count = required_sample_count

        additional_annotation = Annotation.annotate(
            self.training_set,
            self.keyword_expansion.current_label,
            annotation.get_keywords(),
            self.keyword_expansion.get_keywords_for_label(self.keyword_expansion.current_label, enabled_keywords_only=False),
            sample_count=sample_count,
            logging_path=self.logging_path,
        )

        new_annotation = Annotation(
            annotation.get_annotated_random_sample() + additional_annotation.get_annotated_random_sample(),
            annotation.get_keywords()
        )

        print(annotation.get_sample_count())
        print(additional_annotation.get_sample_count())
        print(new_annotation.get_sample_count())

        keyword_group = self.keyword_expansion.labels[self.keyword_expansion.current_label].keyword_groups[keyword_group_id]

        keyword_group.annotated_random_sample = new_annotation.get_annotated_random_sample()
        keyword_group.keywords = new_annotation.get_keywords()
        keyword_group.confidence = new_annotation.get_overall_confidence()
        keyword_group.sample_size = new_annotation.get_sample_count()

    def do_enable_keyword_group_ids(self, keyword_group_ids: str) -> None:
        """
        Enables keyword groups in case they were disabled.
        """
        for keyword_group_id in keyword_group_ids.split(" "):
            if self.keyword_expansion.enable_keyword_group(int(keyword_group_id)):
                print(f"Enabled keyword group {keyword_group_id}")

        self._autosave()

    def do_disable_keyword_group_ids(self, keyword_group_ids: str) -> None:
        """
        Disables keyword groups in case they were enabled.
        """
        for keyword_group_id in keyword_group_ids.split(" "):
            try:
                if self.keyword_expansion.disable_keyword_group(int(keyword_group_id)):
                    print(f"Disabled keyword group {keyword_group_id}")
            except ValueError:
                pass

        self._autosave()

    def do_view_annotated_sample(self, keyword_group_id: int):
        """
        Print the annotation done when a keyword group was added.
        Reqieres keyword_group_id as argument.
        """
        try:
            keyword_group_id = int(keyword_group_id)
        except ValueError:
            return

        annotation = self.keyword_expansion.get_annotation(keyword_group_id)

        if annotation is not None:
            text_group_keywords_to_emphasize = self.keyword_expansion.get_keywords_for_keyword_group(keyword_group_id)
            all_keywords_to_emphasize = self.keyword_expansion.get_keywords_for_label(self.keyword_expansion.current_label, enabled_keywords_only=False)
            print(annotation.to_string(text_group_keywords_to_emphasize, all_keywords_to_emphasize))

    def do_print_keywords(self, _):
        """
        Print all keywords for the current label. If no label is set, prints keywords for all labels.
        """
        print(self._print_keywords(do_evaluate=False))

    def do_print_keywords_and_evaluate(self, _):
        """
        Print all keywords for the current label. If no label is set, prints keywords for all labels.
        """
        print(self._print_keywords(do_evaluate=True))
        self.do_evaluate(None)

    def _print_keywords(self, do_evaluate):
        def _print(label):
            gold_occurences_ratio = sum(
                1 if label in gold_labels else 0
                for gold_labels
                in self.development_set.get_labels()
            ) / len(self.development_set.get_labels())

            keywords = self.keyword_expansion.get_keywords_for_label(label)
            total_occurences = self.training_set.get_word_index().get_occurences(keywords)

            total_confidence_numerator = sum(
                keyword_group.confidence * self.training_set.get_word_index().get_occurence_count(keyword_group.keywords)
                for keyword_group
                in self.keyword_expansion.labels[label].keyword_groups.values()
                if keyword_group.sample_size >= Annotation.minimum_required_annotation_count(
                    self.training_set.get_word_index().get_occurence_ratio(keyword_group.keywords)
                ) and keyword_group.enabled
            )
            total_confidence_denominator = sum(
                self.training_set.get_word_index().get_occurence_count(keyword_group.keywords)
                for keyword_group
                in self.keyword_expansion.labels[label].keyword_groups.values()
                if keyword_group.enabled
            )
            if total_confidence_denominator == 0:
                total_confidence = float("nan")
            else:
                total_confidence = total_confidence_numerator / total_confidence_denominator

            total_occurence_ratio = self.training_set.get_word_index().get_occurence_ratio(keywords)
            estimated_recall = total_occurence_ratio * total_confidence / gold_occurences_ratio

            return (
                f"-------------------------------------------------------------------------------------------------------------------------\n"
                f"-- {label:^115} --\n"
                f"-------------------------------------------------------------------------------------------------------------------------\n"
            ) + self.keyword_expansion.print_keywords(label, self.training_set, self.development_set, do_evaluate=do_evaluate) + (
                f"-------------------------------------------------------------------------------------------------------------------------\n"
                f"-- Estimated Precision: {100*total_confidence:.1f}% (= weighted average of confidences)\n"
                f"-- Total Occurences:     {total_occurences}\n"
                f"-- Dev Set Occurence:   {100*gold_occurences_ratio:>5.1f}% (constant)\n"
                f"-- Estimated Recall:    {100*estimated_recall:>5.1f}% (Total Confidence * Total Occurences / Dev Set Occurences)\n"
                f"--\n"
                f"-- The goal is to achieve as high Estimated Recall as possible (best is 100%),\n"
                f"-- while maintaining high Estimated Precision (preferably above 65%).\n"
                f"-------------------------------------------------------------------------------------------------------------------------\n"
                f"\n"
            )

        result = ""

        if self.keyword_expansion.current_label is None:
            for label in self.keyword_expansion.get_labels():
                result += _print(label) + "\n"
        else:
            result += _print(self.keyword_expansion.current_label)

        return result

    def _evaluate(self, allowed_keyword_group_ids=None):
        if self.keyword_expansion.current_label is None:
            labels = self.keyword_expansion.get_labels()
            show_overall_results = True
        else:
            labels = [self.keyword_expansion.current_label]
            show_overall_results = False

        # Decided to never show overall results in user study
        show_overall_results = False

        predictions = self.keyword_expansion.predict(
            self.development_set,
            allowed_keyword_group_ids,
        )
        evaluation = evaluate(
            gold_data=self.development_set,
            predictions=predictions,
            labels=labels,
            show_overall_results=show_overall_results,
            name=self.keyword_expansion.name,
            description=self.keyword_expansion.description,
        )

        return evaluation

    def do_evaluate(self, _):
        """
        If a currentlabel is set, show metrics for the current label.
        Otherwise evaluate all labels: Show combined metrics as well as individual metrics for each label.
        """
        print(self._evaluate())

    def do_evaluate_keyword_groups(self, keyword_group_ids):
        """
        Evaluate only a subset of keyword groups.
        """
        try:
            allowed_keyword_group_ids = [
                int(keyword_group_id)
                for keyword_group_id
                in keyword_group_ids.split()
            ]
        except ValueError:
            print("Keyword group id must be an integer.")
            return

        print(self._evaluate(allowed_keyword_group_ids))

    def _get_save_file_path(self, save_file: Optional[str]) -> str:
        if save_file is None or save_file == "":
            save_file = "keyword_expansion"

        return self.current_path / f"{save_file}.json"

    def do_save(self, save_file: Optional[str] = None):
        """
        Save changes. When save file is not specified, the default save file is used.
        """
        json_data = self._get_keyword_expansion_data_json()
        save_file_path = self._get_save_file_path(save_file)
        save_file_path.write_text(json_data)

    def _autosave(self):
        self.do_save("autosave")
        self.do_save("autosave-" + datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))

    def do_load(self, save_file: Optional[str] = None):
        """
        Discard changes and load a save file.
        """
        save_file_path = self._get_save_file_path(save_file)

        if save_file_path.is_file():
            self.keyword_expansion = KeywordExpansion.from_json_path(save_file_path)
            print(f"Loaded save file: {save_file_path}")
        else:
            self.do_save(save_file)
            print(f"Created new save file: {save_file_path}")

    def do_load_autosave(self, _):
        self.do_load("autosave")

    def do_save_results(self, save_file: str):
        """
        Save predictions and evaluation results.
        """
        predictions = self.keyword_expansion.predict(self.development_set).get_annotated_fasttext_document()
        (self.current_path / f"{save_file}.dev.pred").write_text("".join(predictions))

        printed_keywords = self._print_keywords(do_evaluate=True)
        evaluation = self._evaluate()
        (self.current_path / f"{save_file}.dev.eval").write_text(printed_keywords + "\n\n" + evaluation)

    def _get_keyword_expansion_data_json(self) -> str:
        return json.dumps(
            dataclasses.asdict(self.keyword_expansion),
            sort_keys=True,
            indent=4,
            separators=(',', ': '),
            cls=json.JSONEncoder,
        )


def evaluate(
    gold_data: Dataset,
    predictions: Dataset,
    labels: List[str],
    show_overall_results: bool,
    threshold: float = 0.0,
    name: str = "",
    description: str = "",
):
    evaluation = ""

    evaluator = MultiLabelEvaluator(
        gold_labels=gold_data.get_labels(),
        predicted_labels=predictions.filter_labels_by_confidence_threshold(threshold).get_labels(),
        name=name,
        description=description,
    )

    if show_overall_results:
        evaluation += evaluator.get_oneliner().to_string(cols=27) + "\n\n"

    evaluation += evaluator.get_per_label_confusion_matrix(
        CountOutputType.PercentageAndCount
    ).filter_by_tags(labels).to_string(cols=14) + "\n"

    evaluation += evaluator.get_precision_recall_f1(
        ).filter_by_tags(labels).to_string(cols=14, print_tags=False) + "\n"

    evaluation += evaluator.get_label_counts(
        ).filter_by_tags(labels).to_string(cols=14, print_tags=False)

    return evaluation


def run_keyword_labeler(arguments_string: List[str]) -> None:
    parser = argparse.ArgumentParser(description="Run keyword labeler.")

    parser.add_argument("name", type=str, help="Project name. If projects exists, loads it.")

    args = parser.parse_args(arguments_string)

    KeywordLabelerShell(args.name).cmdloop_ignoring_keyboard_interrupt()

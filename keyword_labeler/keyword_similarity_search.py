from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple, Set
import datetime

from svd2vec import svd2vec

from keyword_labeler.dataset import WordIndex
from keyword_labeler.util.highlight import yellow_highlight, blue_highlight


# Performs similarity search via svd2vec.
# Called by the `similarity_search` and `bulk_add_keywords` commands of the CLI.


class KeywordSimilarity:
    def __init__(self, model_path: Path, logging_path: Path):
        self.svd_model = svd2vec.load(str(model_path))
        self.logging_path = logging_path

    def search(self, keywords: List[str], top_n: int) -> Optional[List[Tuple[str, float]]]:
        try:
            search_results = KeywordSearchResults(self.svd_model.most_similar(positive=keywords, topn=top_n), keywords)

            with self.logging_path.open("a") as f:
                timestamp = datetime.datetime.now().strftime("%d. %B %Y %I:%M%p")
                f.write(timestamp + " " + search_results.to_string() + "\n")

            return search_results
        except ValueError:
            return KeywordSearchResults(None, [])


@dataclass
class KeywordSearchResults:
    # keyword_search_results = [(keyword_1, confidence_1), (keyword_2, confidence_2), ...]
    keyword_search_results: Optional[List[Tuple[str, float]]]
    keywords: List[str]

    def print(self, word_index: WordIndex, keywords_found_so_far: Set[str]):
        if self.keyword_search_results is not None:
            print(blue_highlight("Yellow keywords are already in some keyword group."))
            print()
            print(f"index  similar_word           confidence         occurences")
            for keyword in self.keywords:
                keyword_line = f"  0   " + f"{keyword:<25}" + " NaN " + f"     {word_index.get_occurences([keyword])}"
                if keyword in keywords_found_so_far:
                    print(yellow_highlight(keyword_line))
                else:
                    print(keyword_line)
            print(f"--------------------------------------------------------------")
            for index, (similar_word, confidence) in enumerate(self.keyword_search_results):
                keyword_line = f"  {index + 1:<4}" + f"{similar_word:<25}" + f"{confidence:.3f}" + f"     {word_index.get_occurences([similar_word])}"
                if similar_word in keywords_found_so_far:
                    print(yellow_highlight(keyword_line))
                else:
                    print(keyword_line)
        else:
            print("Cannot find similar words to keyword, it is rare in the dataset.")

    def to_string(self):
        result = "##".join(self.keywords) + "#####"
        result += "##".join(similar_word for similar_word, _ in self.keyword_search_results)
        return result

    def select(self, threshold) -> List[str]:
        return [
            keyword
            for keyword, _
            in self.keyword_search_results[:threshold]
        ]

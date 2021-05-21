# Keyword Labeler Prototype

Prototype implementation of a keyword labeler used to perform the user study in my master thesis:

```
Text Classification with Limited Training Data
Petr Laitoch, RNDr. Jiří Hana, Ph.D.
Charles University
2021
```

User instructions are present in the appendix of the thesis.

## Setup & Run

We assume Python 3. Tested on Python 3.7.4 on both Mac and Linux.

```
# Setup
python -m venv keyword-labeler-venv
./keyword-labeler-venv/bin/pip install svd2vec pydantic nltk

# Run
./keyword-labeler-venv/bin/python run_keyword_labeler.py example-project
```

## Some Example Commands

When the prompt loads, try to run some of the following example commands to get started:

```
print_keywords
add_keywords waiter
print_keywords_and_evaluate
bulk_all_keywords <Enter> helpful <CTRL_C>
bulk_all_keywords <Enter> helpful personable joked considerate
```

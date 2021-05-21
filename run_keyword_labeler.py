#!/usr/bin/env python3

import sys

from keyword_labeler.keyword_labeler import run_keyword_labeler

if __name__ == "__main__":
    run_keyword_labeler(sys.argv[1:])

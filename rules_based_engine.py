import xml.etree.ElementTree as ET
import spacy
from spacy.lang.zh.examples import sentences

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np

import torch
from torch.nn.utils.rnn import pad_sequence # not from scratch
from torch.utils.data import TensorDataset, DataLoader

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import random
import os

from tqdm import tqdm


import matplotlib.pyplot as plt
from scipy.stats import mode

#-----------------------------------------------------------------------------

# RULE 1: Subject Verb Agreement
def check_subject_verb_agreement(tagged_tokens):
    """
    Checks for subject-verb agreement errors in a sentence.
    This function assumes that the input is a list of (word, POS) tuples.

    Rules:
    - Singular subjects (e.g., he, she, it, singular nouns) must pair with third-person singular verbs (VBZ).
    - Plural subjects (e.g., they, plural nouns) must pair with plural verb forms (VBP).
    - First-person singular (I) and second-person singular/plural (you) have specific verb forms.

    Parameters:
        tagged_tokens (list of tuples): List of (word, POS) tuples.

    Returns:
        int: Number of subject-verb agreement errors detected.
    """
    errors = 0
    singular_pronouns = {"he", "she", "it"}  # Singular pronouns
    plural_pronouns = {"we", "they"}         # Plural pronouns
    second_person_pronouns = {"you"}         # Second person (singular/plural)
    first_person_singular = {"i"}            # First person singular
    verb_dict = {"VB", "VBP", "VBZ", "VBD", "VBG", "VBN"}

    for i, (word, pos) in enumerate(tagged_tokens):
        if pos == "PRP" and word.lower() in singular_pronouns:
            # Find the next verb skipping prepositions/conjunctions
            for j in range(i + 1, len(tagged_tokens)):
                if tagged_tokens[j][1] in verb_dict:
                    if tagged_tokens[j][1] != "VBZ":
                        errors += 1
                    break

        elif pos == "PRP" and word.lower() in plural_pronouns:
            for j in range(i + 1, len(tagged_tokens)):
                if tagged_tokens[j][1] in verb_dict:
                    if tagged_tokens[j][1] != "VBP":
                        errors += 1
                    break

        elif pos == "PRP" and word.lower() in second_person_pronouns:
            for j in range(i + 1, len(tagged_tokens)):
                if tagged_tokens[j][1] in verb_dict:
                    if tagged_tokens[j][1] not in {"VB", "VBP"}:
                        errors += 1
                    break

        elif pos == "PRP" and word.lower() in first_person_singular:
            for j in range(i + 1, len(tagged_tokens)):
                if tagged_tokens[j][1] in verb_dict:
                    if tagged_tokens[j][1] != "VBP":
                        errors += 1
                    break

        elif pos == "NN":
            for j in range(i + 1, len(tagged_tokens)):
                if tagged_tokens[j][1] in verb_dict:
                    if tagged_tokens[j][1] != "VBZ":
                        errors += 1
                    break

        elif pos == "NNS":
            for j in range(i + 1, len(tagged_tokens)):
                if tagged_tokens[j][1] in verb_dict:
                    if tagged_tokens[j][1] != "VBP":
                        errors += 1
                    break

    return errors

#-----------------------------------------------------------------------------

# helper function to enforce RULE 2
# not all-encompassing
word_to_num = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "hundred": 100,
    "hundreds": 100,
    "thousand": 1000,
    "thousands": 1000,
    "million": 10e6,
    "millions": 10e6
}

def word_to_number(word):
    """
    Converts a word or digit string into an integer.

    Parameters:
        word (str): The input word or string.

    Returns:
        int: The corresponding integer value, or None if not valid.
    """
    if word.isdigit():  # Check if it's a numeric string
        return int(word)
    elif word.lower() in word_to_num:  # Check if it's a number word
        return word_to_num[word.lower()]
    return None

#-----------------------------------------------------------------------------

# RULE 2: Singular/Plural Noun Confusion
def check_singular_plural_confusion(tagged_tokens):
    """
    Checks for singular/plural noun confusion errors based on POS tags and context.
    Uses Penn Treebank POS tags to detect errors in singular/plural noun usage.

    Parameters:
        tagged_tokens (list of tuples): List of (word, POS) tuples.

    Returns:
        int: Number of singular/plural noun confusion errors detected.
    """
    errors = 0
    for i, (word, pos) in enumerate(tagged_tokens):
        number = word_to_number(word)
        if pos in {"DT", "CD"} and word.lower() in {"a", "an", "one", "this", "that"} or (number == 1):
            for j in range(i + 1, len(tagged_tokens)):
                if tagged_tokens[j][1] in {"NN", "NNS"}:
                    if tagged_tokens[j][1] == "NNS":
                        errors += 1
                    break

        elif pos in {"DT", "CD", "JJ", "PDT"} and word.lower() in {"many", "several", "these", "those", "all", "both"} or (number and number > 1):
            for j in range(i + 1, len(tagged_tokens)):
                if tagged_tokens[j][1] in {"NN", "NNS"}:
                    if tagged_tokens[j][1] == "NN":
                        errors += 1
                    break

    return errors

#-----------------------------------------------------------------------------

# RULE 3: Verb Tense Confusion
def check_verb_tense_confusion(tagged_tokens):
    """
    Checks for verb tense confusion errors based on temporal expressions and verb tenses.
    Uses Penn Treebank POS tags to detect mismatches between temporal context and verb forms.

    Parameters:
        tagged_tokens (list of tuples): List of (word, POS) tuples.

    Returns:
        int: Number of verb tense confusion errors detected.
    """
    errors = 0
    past_markers = {"yesterday", "last", "ago", "earlier"}
    present_markers = {"today", "now", "currently"}
    future_markers = {"tomorrow", "next", "later", "soon", "will"}
    verb_dict = {"VB", "VBD", "VBZ", "VBP", "MD", "VBG", "VBN"}

    for i, (word, pos) in enumerate(tagged_tokens):
        if word.lower() in past_markers:
            for j in range(i + 1, len(tagged_tokens)):
                if tagged_tokens[j][1] in verb_dict:
                    if tagged_tokens[j][1] not in {"VBD", "VBN"}:
                        errors += 1
                    break

        elif word.lower() in present_markers:
            for j in range(i + 1, len(tagged_tokens)):
                if tagged_tokens[j][1] in verb_dict:
                    if tagged_tokens[j][1] not in {"VBZ", "VBP", "VBG"}:
                        errors += 1
                    break

        elif word.lower() in future_markers:
            for j in range(i + 1, len(tagged_tokens)):
                if tagged_tokens[j][1] in verb_dict:
                    if tagged_tokens[j][1] != "MD":
                        errors += 1
                    break

    return errors

#-----------------------------------------------------------------------------

# RULE 4: Omitting/Inserting Articles
def check_articles(tagged_tokens):
    """
    Checks for errors related to omitting or inserting articles in sentences.
    Uses Penn Treebank POS tags to detect errors in article usage.

    Parameters:
        tagged_tokens (list of tuples): List of (word, POS) tuples.

    Returns:
        int: Number of article-related errors detected.
    """
    errors = 0
    uncountable_nouns = {"homework", "air", "furniture", "information", "advice",
                         "rice", "fear", "safety", "water", "beauty", "knowledge", "love",
                         "research", "advice", "work", "bread", "traffic", "travel", "weather", "news"}

    for i, (word, pos) in enumerate(tagged_tokens):
        if pos == "NN":
            if i > 0 and tagged_tokens[i - 1][1] == "DT":
                if word.lower() in uncountable_nouns and tagged_tokens[i - 1][0].lower() in {"a", "an"}:
                    errors += 1
                continue
            elif word.lower() not in uncountable_nouns:
                errors += 1

        if pos in {"NNS", "NN"} and i > 0 and tagged_tokens[i - 1][1] == "DT":
            if (word.lower() in uncountable_nouns and tagged_tokens[i - 1][0].lower() in {"a", "an"}) or pos == "NNS":
                errors += 1

    return errors

#-----------------------------------------------------------------------------

rule_functions = [
    check_subject_verb_agreement,
    check_singular_plural_confusion,
    check_verb_tense_confusion,
    check_articles,
]

#-----------------------------------------------------------------------------

def check_all_rules(tagged_tokens):
    total_errors = 0
    for rule in rule_functions:
        errors = rule(tagged_tokens)
        print(f"Rule {rule.__name__} flagged {errors} error(s).")
        total_errors += errors
    return total_errors

#-----------------------------------------------------------------------------

nlp_en = spacy.load("en_core_web_sm")

def get_ptb_tags(sentence):
  doc = nlp_en(sentence)
  return [(token.text, token.tag_) for token in doc]

#-----------------------------------------------------------------------------

def check_sentences(sentences):
    """
    Checks multiple sentences for grammar errors using all rules.

    Parameters:
        sentences (list of str): List of sentences to be checked.

    Returns:
        dict: A dictionary where each sentence maps to its total error score.
    """

    results = {}
    for sentence in sentences:
        tagged_tokens = get_ptb_tags(sentence)
        print(f"Sentence: {sentence}")
        print(f"Tagged Tokens: {tagged_tokens}")
        
        total_errors = check_all_rules(tagged_tokens)
        results[sentence] = total_errors
        print(f"Total Errors Detected: {total_errors}\n")
    return results

#-----------------------------------------------------------------------------


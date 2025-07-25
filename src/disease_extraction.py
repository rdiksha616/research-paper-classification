#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Disease extraction using rule-based matching with a dictionary of known diseases.
"""

import re

# Simple dictionary of common diseases (expand as needed)
DISEASE_TERMS = [
    "lung cancer", "breast cancer", "colon cancer", "liver cancer", "cervical cancer",
    "prostate cancer", "skin cancer", "thyroid cancer", "leukemia", "melanoma",
    "pancreatic cancer", "brain tumor", "gastric cancer", "ovarian cancer",
    "kidney cancer", "stomach cancer", "esophageal cancer"
]


def extract_diseases_rulebased(text):
    """
    Extract diseases by searching known terms (case-insensitive).
    """
    found = []
    text_lower = text.lower()
    for disease in DISEASE_TERMS:
        if disease in text_lower:
            found.append(disease.title())

    # Optionally, match patterns like "<something> cancer"
    pattern = r"\b([A-Za-z]+ cancer)\b"
    matches = re.findall(pattern, text_lower)
    for m in matches:
        found.append(m.title())

    return sorted(list(set(found)))


if __name__ == "__main__":
    sample = "This study discusses lung cancer and pancreatic cancer treatment."
    print("Detected diseases:", extract_diseases_rulebased(sample))

"""Input preprocessing classes."""

import re
import string
from typing import Sequence, List
from nltk import download, word_tokenize
from nltk.corpus import stopwords

download('stopwords')


def clean_data(texts: Sequence[str]) -> List[str]:
    """Clean each string in a list by removing non-alphabetic characters and stopwords and making it lowercase."""
    stops = stopwords.words('english')
    non_alphabetic = re.compile(r'[^a-zA-Z ]')  # regex that matches non-alphabetic characters
    processed_texts = []
    for text in texts:
        sentence = non_alphabetic.sub(' ', text)  # Remove non-alphabetic characters
        words = word_tokenize(sentence.lower())  # Split lower-cased text into words
        processed_text = ""
        for w in words:
            if not w.isdigit() and w not in stops and w not in string.punctuation:
                processed_text += " " + w  # append word to text
        processed_texts.append(processed_text.lstrip())  # Remove first space before appending
    return processed_texts

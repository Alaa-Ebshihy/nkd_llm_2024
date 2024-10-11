"""
contain functions for text preporcessing
"""
from langdetect import detect
from langdetect import DetectorFactory
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
from numpy import dot
from numpy.linalg import norm


def detect_text_language(text):
    # set seed
    DetectorFactory.seed = 0

    lang = "en"
    try:
        if len(text) > 50:
            lang = detect(" ".join(text[:50]))
        elif len(text) > 0:
            lang = detect(" ".join(text[:len(text)]))
    except Exception as e:
        all_words = set(text)
        try:
            lang = detect(" ".join(all_words))
        except Exception as e:
            lang = ""
            pass
    return lang


def spacy_tokenizer(sentence, parser, stopwords, punctuations):
    mytokens = parser(sentence)
    mytokens = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens]
    mytokens = [word for word in mytokens if word not in stopwords and word not in punctuations]
    mytokens = " ".join([i for i in mytokens])
    return mytokens


# TODO: possibility to add sci bert also
def vectorize_text(text, model='en_core_sci_lg'):
    if model == 'tfidf':
        vectorizer = TfidfVectorizer(max_features=2 ** 12)
        return vectorizer.fit_transform(text)
    nlp = spacy.load(model)
    doc = nlp(text)
    return doc.vector


def cosine_similarity(a, b):
    cos_sim = dot(a, b) / (norm(a) * norm(b))
    return cos_sim

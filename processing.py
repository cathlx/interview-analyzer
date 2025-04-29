import joblib
import numpy as np
import pandas as pd
import spacy
from senticnet import senticnet
from spacy.matcher import DependencyMatcher


nlp = spacy.load('en_core_web_sm')


def process_sentence(sent: str):
    pattern = [
    {
        "RIGHT_ID": "interviewee",       
        "RIGHT_ATTRS": {"LOWER": {"IN": ["i", "me", "my", "mine", "myself"]}}
    },
    {
        "LEFT_ID": "interviewee",
        "REL_OP": "<<",
        "RIGHT_ID": "interviewee parent",
        "RIGHT_ATTRS": {},
    },
    {
        "LEFT_ID": "interviewee parent",
        "REL_OP": ">>",
        "RIGHT_ID": "description",
        "RIGHT_ATTRS": {"POS": {"IN": ["ADJ", "ADV", "INTJ", "NOUN", "VERB"]}},
    }
    ]

    matcher = DependencyMatcher(nlp.vocab)
    matcher.add("interviewee", [pattern])

    doc = nlp(sent)
    matches = matcher(doc)

    positivity = 0.0
    negativity = 0.0

    for match in matches:
        for i in range(1, len(match[1])):
            if doc[match[1][i]].lemma_ in senticnet.senticnet:
                _, _, _, _, _, _, polarity_label, polarity_value, _, _, _, _, _ = senticnet.senticnet[doc[match[1][i]].lemma_]
                if polarity_label == 'positive':
                    positivity += polarity_value
                else:
                    negativity += polarity_value

    inner = 0
    outer = 0

    for sent in doc.sents:
        for word in sent:
            if word.dep_ == 'nsubj' or word.dep_ == 'poss' or word.dep_ == 'attr':
                if (word.pos_ == 'NOUN' or word.pos_ == 'PROPN') and word.dep_ != 'nsubjpass':
                    outer += 1
                elif word.pos_ == 'PRON':
                    if word.morph.get('Person') == ['1']:
                        inner += 1
                    else:
                        outer += 1
            elif word.dep_ == 'dobj' or word.dep_ == 'iobj':
                if word.pos_ == 'NOUN' or word.pos_ == 'PROPN':
                    inner += 1
                elif word.pos_ == 'PRON':
                    if 'Person' in word.morph and word.morph.get('Person')[0] == '1':
                        outer += 1
                    else:
                        inner += 1

    return ' '.join([token.lemma_ for token in doc if not token.is_punct and not token.is_digit and (token.text.lower() == 'i' or not token.is_stop)]), positivity, negativity, inner, outer, inner / max(outer, 1), positivity * (inner - outer)
    
def extract_features(filename: str) -> np.ndarray:
    sents = []
    for answer in [line[3:] for line in open(filename).readlines() if line[0] == 'A']:
        sents.extend(list(nlp(answer).sents))

    df = pd.Series(sents, dtype='string').to_frame(name='phrase')
    df[['lemmas', 'pos', 'neg', 'ext', 'int', 'in/out', 'ass_score']] = df['phrase'].apply(lambda x: pd.Series(process_sentence(x)))

    tf_idf_vectorizer = joblib.load('model/vectorizer.pkl')
    tf_idf = tf_idf_vectorizer.transform(df.lemmas)

    tf_idf = pd.DataFrame(tf_idf.toarray(), columns=tf_idf_vectorizer.get_feature_names_out())
    features = pd.concat([tf_idf, df[['pos', 'neg', 'ext', 'int', 'in/out', 'ass_score']].reset_index()], axis=1)
    features.drop('index', axis=1, inplace=True)
    
    return features.values

def analyze(filename: str) -> int:
    features = extract_features(filename)

    model = joblib.load('model/svm_model.pkl')
    prediction = model.predict(features)

    return sum(prediction)


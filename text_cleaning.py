import spacy
from spacy.lang.en import English

nlp = English()
spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS

def text_data_cleaning(sentence):
    doc = nlp(sentence)

    tokens = [] # list of tokens
    for token in doc:
        if token.lemma_ != "-PRON-":
            temp = token.lemma_.lower().strip()
        else:
            temp = token.lower_
        tokens.append(temp)

    cleaned_tokens = []
    for token in tokens:
        if token not in spacy_stopwords and token.isalpha():
            cleaned_tokens.append(token)
    return cleaned_tokens

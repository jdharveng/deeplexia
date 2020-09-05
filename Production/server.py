from flask import Flask, jsonify, request
import pandas as pd
import joblib
import numpy as np
import pickle
from scipy import spatial
import spacy
import nltk
from nltk.corpus import stopwords


app = Flask(__name__)

nlp = spacy.load('en_core_web_lg')
emoji_symb2emb_dic = joblib.load('avg_glove_embedding.pkl')
glove_lookup = joblib.load('glove_lookup')

@app.route('/emojis', methods=['POST'])
def emojis_get():
    text = request.json['text']
    emoji = translate_text(text)
    return jsonify({'text' : text, 'emoji': emoji}), 201

def some_preprocessing(description):
    # Place to lower case
    prep_descr = description.lower()
    # Tokenize and remove non alphanumeric tokens
    tokens = nltk.word_tokenize(prep_descr)
    token_words = [w for w in tokens if w.isalpha()]
    # Removing stopwords
    stops = set(stopwords.words("english")) 
    without_stopwords = [w for w in token_words if not w in stops]
    return without_stopwords

def avg_glove_vector(descr_list):
    # counting number of vectors found in the lookup table
    n_vectors = 0
    # Getting back the size of the Glove Vectors
    glove_dim = len(glove_lookup['a'])
    # Average Vector
    avg_vector = np.zeros(glove_dim)
    
    # Going over each word in the input_list
    for word in descr_list:
        if word in glove_lookup.keys():
            n_vectors += 1
            avg_vector += glove_lookup[word]
        else:
            continue
    
    if n_vectors == 0:
        return ""
    else:
        return avg_vector / n_vectors

def find_closest_emoji_emb(sentence):
    # Preprocess the sentence: removing punctuations, stopswords, ...
    preprocessed_list = some_preprocessing(sentence)
    # Take the Avg of the GloVe Embedded Vectors
    embedded_sentence = avg_glove_vector(preprocessed_list)
    closest_emojis = sorted(emoji_symb2emb_dic.keys(), key= lambda emoji_symbol:\
                            spatial.distance.euclidean(emoji_symb2emb_dic[emoji_symbol], embedded_sentence))
    return closest_emojis

def translate_text(file):
    spacy_nlp_file = nlp(file)
    for num, sentence in enumerate(spacy_nlp_file.sents):
        print(f'{num}: {sentence}')
        closest_emojis = find_closest_emoji_emb(str(sentence))
        print(closest_emojis[:3])
        print("")
        
if __name__ == "__main__":
	app.run(debug=True)
        

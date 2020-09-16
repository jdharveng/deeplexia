import pandas as pd
import joblib
import numpy as np
import pickle
from scipy import spatial
from scipy import spatial
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from numpy.core._exceptions import UFuncTypeError

nlp = spacy.load('en_core_web_lg')
porter = PorterStemmer()
emoji_symb2emb_dic = joblib.load('avg_glove_embedding.pkl')
glove_lookup = joblib.load('glove_lookup')

def some_preprocessing(description):
   # Place to lower case
   prep_descr = description.lower()
   # Tokenize and remove non alphanumeric tokens
   tokens = nltk.word_tokenize(prep_descr)
   token_words = [w for w in tokens if w.isalpha()]
   # Removing stopwords
   stops = set(stopwords.words("english"))
   without_stopwords = [w for w in token_words if not w in stops]
   # Stemming the Words
   stemmed_words = [porter.stem(w) for w in without_stopwords]
   return stemmed_words

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
   try:
      closest_emojis = sorted(emoji_symb2emb_dic.keys(), key=lambda emoji_symbol: \
         spatial.distance.euclidean(emoji_symb2emb_dic[emoji_symbol], embedded_sentence))
      return closest_emojis

   except UFuncTypeError:
      return ""



def translate_text(file):
   spacy_nlp_file = nlp(file)
   emoji_translation = ""
   for sentence in spacy_nlp_file.sents:
      closest_emojis = find_closest_emoji_emb(str(sentence))
      if len(closest_emojis) > 0:
         emoji = closest_emojis[0]
      else:
         emoji = ""
      emoji_translation += (emoji + '\n')

   return emoji_translation


def translate_by_keywords(file):
   spacy_nlp_file = nlp(file)
   emoji_text = ""
   for sentence in spacy_nlp_file.sents:
      emoji_sentence = ""
      for token in sentence:
         token_pos = token.pos_
         if token_pos == 'PROPN' or token_pos == 'NOUN':
            closest_emojis = find_closest_emoji_emb(str(token))
            if len(closest_emojis) > 0:
               emoji_sentence = emoji_sentence + "   " + closest_emojis[0]

      emoji_text = emoji_text + emoji_sentence + '\n'

   return emoji_text

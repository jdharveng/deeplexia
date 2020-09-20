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
list_names = joblib.load('list_names.pkl')
emoji_symb2emb_dic = joblib.load('weighted_emoji_symb2emb_dic.pkl')
name_emoji_symb2emb_dic = joblib.load('name_emoji_symb2emb_dic.pkl')
glove_lookup = joblib.load('glove_lookup')

def some_preprocessing(description):
   '''
   Returns the preprocessed description.

   Parameters:
       description (str): The string description of the emoji, that will be
                          preprocessed.

       remove_stopw (bool): Default = True, if False => Stopwords aren't removed

   Returns:
       some_preprocessing(description): A list of string of lower case tokens
       punctuations and stopwords)
   '''

   # Place to lower case
   prep_descr = description.lower()
   # Tokenize and remove non alphanumeric tokens
   tokens = nltk.word_tokenize(prep_descr)
   token_words = [w for w in tokens if w.isalpha()]
   # Removing stopwords
   stops = set(stopwords.words("english"))
   without_stopwords = [w for w in token_words if not w in stops]
   # Removing first names
   without_first_names = [w for w in without_stopwords if not w in list_names]
   # Stemming the Words
   stemmed_words = [porter.stem(w) for w in without_first_names]
   return stemmed_words

def avg_glove_vector(descr_list):
   '''
     Returns from a preprocessed list of the emoji description the average embedding

     Parameters:
         descr_list (list): List containing string tokens of preprocessed emoji description

         emoji_2_embedding_lookup (dict): dictionnary with as keys the words of the
                         corpus and the values the vector of that word

     Returns:
         An embedding vector with the same dimensions as the Word Embedding used
   '''

   # counting number of vectors found in the lookup table
   n_vectors = 0
   # Getting back the size of the Glove Vectors
   embedding_dim = len(glove_lookup['a'])
   # Average Vector
   avg_vector = np.zeros(embedding_dim)

   # Creating an nlp object in order to apply Spacy methods
   nlp_object = nlp(" ".join(descr_list))

   for token in nlp_object:
      if str(token) in glove_lookup.keys():
         if token.pos_ == "NOUN":
            n_vectors += 4
            avg_vector += 4 * glove_lookup[str(token)]
         elif token.pos_ == "VERB":
            n_vectors += 3
            avg_vector += 3 * glove_lookup[str(token)]
         elif token.pos == "PROPN":
            n_vectors += 2
            avg_vector += 2 * glove_lookup[str(token)]
         else:
            n_vectors += 1
            avg_vector += 1 * glove_lookup[str(token)]
      else:
         continue

   if n_vectors == 0:
      return ""
   else:
      return avg_vector / n_vectors

def find_closest_emoji_emb(sentence, emoji_symb2emb_dic=emoji_symb2emb_dic):
   '''
     Returns a sorted list (Descending Order) with the closest Emoji to the sentence

     Parameters:
         sentence (str) : Sentence for which we want to find the closest emoji

         emoji_2_embedding_lookup (dict): dictionnary with as keys the words of the
                         corpus and the values the vector of that word

         emoji_symb2emb_dic (dict): keys emoji_symbols values averaged description Embedding

         distance_type (str): default is "euclidiean", but also "cosine" possible

     Returns:
         A sorted list (Descending Order) with the closest Emoji to the sentence
   '''

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
   '''
     Returns the input spacy_file split in sentences with the closest emoji

     Parameters:
         spacy_nlp_file) (spacy nlp object) : Text as nlp object that we want to translate

         emoji_2_embedding_lookup (dict): dictionnary with as keys the words of the
                         corpus and the values the vector of that word

         emoji_symb2emb_dic (dict): keys emoji_symbols values averaged description Embedding

         distance_type (str): default is "euclidiean", but also "cosine" possible

     Returns:
         The input spacy_file split in sentences with the closest emoji
   '''

   spacy_nlp_file = nlp(file)
   emoji_translation = "\n"
   for sentence in spacy_nlp_file.sents:
      emoji_translation += str(sentence)
      closest_emojis = find_closest_emoji_emb(str(sentence))
      if len(closest_emojis) > 0:
         emoji = closest_emojis[0]
      else:
         emoji = ""

      emoji_translation =  emoji_translation + emoji + '\n\n'
   return emoji_translation


def translate_by_keywords(file):
   '''
     Returns the input spacy_file split in sentences and in nouns with the
             closest emoji for each noun

     Parameters:
         spacy_nlp_file) (spacy nlp object) : Text as nlp object that we want to translate

         emoji_2_embedding_lookup (dict): dictionnary with as keys the words of the
                         corpus and the values the vector of that word

         emoji_symb2emb_dic (dict): keys emoji_symbols values averaged description Embedding

         distance_type (str): default is "euclidian", but also "cosine" possible

     Returns:
         The input spacy_file split in sentences and in nouns with the
         closest emoji for each noun.
   '''

   spacy_nlp_file = nlp(file)
   emoji_text = "\n"
   for sentence in spacy_nlp_file.sents:
      emoji_sentence = str(sentence)
      for token in sentence:
         token_pos = token.pos_
         if token_pos == 'PROPN' or token_pos == 'NOUN':
            closest_emojis = find_closest_emoji_emb(str(token),name_emoji_symb2emb_dic)
            if len(closest_emojis) > 0:
               emoji_sentence = emoji_sentence + "   " + closest_emojis[0]

      emoji_text = emoji_text  + emoji_sentence + '\n\n '

   return emoji_text

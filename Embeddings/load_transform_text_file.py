import pandas as pd
import numpy as np
import math
import pickle
import spacy
import warnings

nlp = spacy.load('en_core_web_lg')
from helper_functions import *

print("Please enter the path of the book to transform ")
path_text = input()
text_2_transform = open(path_text).read()
file_name = nlp(text_2_transform)

with open('Saved_Embeddings/glove_lookup', 'rb') as f:
    glove_lookup = pickle.load(f)

with open('Saved_Variables/emoji_symb2emb_dic.pkl', 'rb') as f:
    emoji_symb2emb_dic = pickle.load(f)

print("Enter the desired length of the subsentences")
length_subsentence = input()
print("Enter the desired length of the list with the closest predicted emojis")
length_predictions = input()

warnings.filterwarnings('ignore')
df_choose_emoji = df_4_dataset(file_name, glove_lookup, emoji_symb2emb_dic, int(length_subsentence), int(length_predictions), "euclidean")

# Test with pickle save list
with open('Saved_Variables/sub_sentence_list.pkl', 'wb') as f:
    pickle.dump(list(df_choose_emoji.sub_sentence), f)

with open('Saved_Variables/closest_emojis_list.pkl', 'wb') as f:
    pickle.dump(list(df_choose_emoji.closest_emojis), f)


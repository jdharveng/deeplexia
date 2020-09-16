import numpy as np
import pandas as pd
import math
from scipy import spatial
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
import spacy
from sklearn.manifold import TSNE
import plotly.express as px

############################################################
############## Functions for I.Simple_text_2_Emjoji#########
############################################################
nlp = spacy.load('en_core_web_lg')

def word2emoji(word1, emoji_df):
    '''
    Computing average similarity between a word and all the names of the emojis
    returning the emoji with highest similarity
    '''
    
    # Keeping track of the computed similarities
    similarity_list = []
    list_names = list(emoji_df.name)
    
    for description in list_names:
        lenght_desc = len(description) # in order to take the average similarity
        # Converting to a nlp object 
        doc = nlp(description)
        
        similarity = 0
        for word2 in doc:
            similarity += word1.similarity(word2)
        
        similarity_list.append(similarity/lenght_desc)
       
    # converting list to np array to get easily index of the max similarity
    np_similarity = np.asarray(similarity_list)
    # pdb.set_trace()
     
    # index of max similarity
    i_max = np.argmax(np_similarity)
  
    #return emoji_df.emoji.iloc[i_max]            
    return emoji_df.emoji.iloc[i_max] 








############################################################
############## Functions for II.emojiDatasetCluster#########
############################################################


def some_preprocessing(description, remove_stopw=True):
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

    if remove_stopw:
        # Removing stopwords
        stops = set(stopwords.words("english"))
        # In order to not taking into account the padding 
        stops.add("<SPACE>")
        stops.add("SPACE")
        stops.add("<space>")
        stops.add("space")
        token_words = [w for w in token_words if not w in stops]
    return token_words



def avg_glove_vector(descr_list, emoji_2_embedding_lookup):
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
    embedding_dim = len(emoji_2_embedding_lookup['a'])
    # Average Vector
    avg_vector = np.zeros(embedding_dim)

    # Going over each word in the input_list
    for word in descr_list:
        if word in emoji_2_embedding_lookup.keys():
            n_vectors += 1
            avg_vector += emoji_2_embedding_lookup[word]
        else:
            continue

    if n_vectors == 0:
        return ""
    else:
        return avg_vector / n_vectors




def building_tsne_df(emoji_symbols, emoji_names, emb_emoji_vectors):
    '''
      Returns a DataFrame with a t-SNE 2D reduction of the emoji description embedding

      Parameters:
          emoji_symbols (list): List of emoji symbols

          emoji_names (list): List of correlated emoji names

          emb_emoji_vectors(list): list of embedded emoji description

      Returns:
          An embedding vector with the same dimensions as the Word Embedding used
    '''
    tsne = TSNE(n_components=2, random_state=0)
    Y = tsne.fit_transform(emb_emoji_vectors)
    tsne_2d_df = pd.DataFrame({'emoji_names': np.asarray(emoji_names), \
                               'emoji_symbols': np.asarray(emoji_symbols),\
                              'X': Y[:, 0], 'Y': Y[:, 1]})

    return tsne_2d_df




def tsne_plot(tsne_2d_df, graph_title):
    '''
      Returns a 2D plotly graph of the t-SNE dimension reduction of the emoji descripition embedding

      Parameters:
          tsne_2d_df (DataFrame): DataFrame obtained with building_tsne_df(emoji_symbols,
          emoji_names, emb_emoji_vectors)

          graph_title (str): Desired title of the graph

      Returns:
          A 2D plotly graph of the t-SNE dimension reduction of the emoji descripition embedding
    '''
    fig = px.scatter(tsne_2d_df, x='X', y='Y', text='emoji_names')
    fig.update_traces(textposition='top center')
    fig.update_layout(
        height=1200,
        width=1000,
        title_text=graph_title)
    fig.show()




def find_closest_emoji_emb(sentence,emoji_2_embedding_lookup, emoji_symb2emb_dic, distance_type="euclidean"):
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
    embedded_sentence = avg_glove_vector(preprocessed_list, emoji_2_embedding_lookup)
    if embedded_sentence == "":
      return []

    if distance_type == "euclidean":
        closest_emojis = sorted(emoji_symb2emb_dic.keys(), key= lambda emoji_symbol:\
                            spatial.distance.euclidean(emoji_symb2emb_dic[emoji_symbol], embedded_sentence))
    elif distance_type == "cosine":
        closest_emojis = sorted(emoji_symb2emb_dic.keys(), key= lambda emoji_symbol:\
                            spatial.distance.cosine(emoji_symb2emb_dic[emoji_symbol], embedded_sentence))
    return closest_emojis




def translate_text(spacy_nlp_file, emoji_2_embedding_lookup, emoji_symb2emb_dic, distance_type="euclidean"):
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
    for num, sentence in enumerate(spacy_nlp_file.sents):
        print(f'{num}: {sentence}')
        closest_emojis = find_closest_emoji_emb(str(sentence), emoji_2_embedding_lookup, emoji_symb2emb_dic, distance_type)
        print(closest_emojis[:3])
        print("")



def translate_by_keywords(spacy_nlp_file, emoji_2_embedding_lookup, emoji_symb2emb_dic, distance_type="euclidean"):
    '''
      Returns the input spacy_file split in sentences and in nouns with the
              closest emoji for each noun

      Parameters:
          spacy_nlp_file) (spacy nlp object) : Text as nlp object that we want to translate

          emoji_2_embedding_lookup (dict): dictionnary with as keys the words of the
                          corpus and the values the vector of that word

          emoji_symb2emb_dic (dict): keys emoji_symbols values averaged description Embedding

          distance_type (str): default is "euclidiean", but also "cosine" possible

      Returns:
          The input spacy_file split in sentences and in nouns with the
          closest emoji for each noun.
    '''

    for num, sentence in enumerate(spacy_nlp_file.sents):
        print("\033[1m" + f'{num}: {sentence}' + "\033[0m")

        for token in sentence:
            token_pos = token.pos_
            if token_pos == 'PROPN' or token_pos == 'NOUN':
                  closest_emojis = find_closest_emoji_emb(str(token), emoji_2_embedding_lookup, emoji_symb2emb_dic, distance_type)
                  print(token, " --- EMOJI --->  ", closest_emojis[:3])
        print("")



def create_chunks(list_tokens, n_words):
    '''Help function for divide_sent2chunks

      Parameters:
          list_tokens (list) : list of str tokens of current sentence

          n_words (int) : size of the sub sentence chunks
    '''
    for i in range(0, len(list_tokens), n_words):
        yield list_tokens[i:i + n_words]


def divide_sent2chunks(file, n_words=5):
    '''Split of sentence tokens in chunks of n_words

      Parameters:
          file (Spacy nlp object) : of chosen input text

          n_words (int) : size of the sub sentence chunks

      Returns:
          Split of sentence tokens in chunks of n_words
    '''
    list_sent_chunks = []

    for num, sentence in enumerate(file.sents):
        help_list = []

        for i, token in enumerate(sentence):
            # Take only the words (not the spaces and punctuactions)
            if token.is_alpha:
                help_list.append(str(token))

        # Building sublists of lenght n_words
        l_list = len(help_list)
        if l_list % n_words != 0:
            rounded_div = math.ceil(l_list / n_words)
            full_length = rounded_div * n_words
            diff = full_length - l_list

            for i in range(diff):
                help_list.append("<SPACE>")

        list_sent_chunks.append(list(create_chunks(help_list, n_words)))

        #if num > 10:
        #    break

    return list_sent_chunks


def chunks2string(list_of_chunks):
    ''' Returns a list with number of original sentence and a list chunked sentences

      Parameters:
          list_of_chunks (list) : output of divide_sent2chunks

      Returns:
          A list with number of original sentence and a list chunked sentences
    '''
    list_num_sent = []
    list_of_str_chunks = []


    for num, sent in enumerate(list_of_chunks):
        for chunk in sent:
            sub_string = " ".join(chunk)
            list_of_str_chunks.append(sub_string)
            list_num_sent.append(num)

    return list_num_sent, list_of_str_chunks


def df_4_dataset(file, emoji_2_embedding_lookup,emoji_symb2emb_dic,n_words=5,\
                 n_emoji=5, distance_type="euclidean"):
    ''' Returns a DataFrame with nb of original sentence, subsentences and closest
          predicted emoji list for the subsentences

      Parameters:
          file (space nlp object): file we want to analyze

          emoji_2_embedding_lookup (dict): dictionnary with as keys the words of the
                corpus and the values the vector of that word

          emoji_symb2emb_dic (dict): keys emoji_symbols values averaged description Embedding

          n_words (int) : size of the sub sentence chunks

          n_emoji (int) : size of the predicted emoji list

          distance_type (str): default is "euclidiean", but also "cosine" possible


      Returns:
          A DataFrame with nb of original sentence, subsentences and closest
          predicted emoji list for the subsentences
    '''

    # Divide file into chunks of n_words length
    list_sent_chunks = divide_sent2chunks(file, n_words)
    # Convert the chunk lists to string sentences
    list_num_sent, list_of_str_chunks = chunks2string(list_sent_chunks)

    # Create a DataFrame from both lists:
    df = pd.DataFrame(zip(list_num_sent, list_of_str_chunks),columns=['num_sentence', 'sub_sentence'])


    # Add a column with list of closest emoji
    df["closest_emojis"] = df.sub_sentence.apply(lambda x: find_closest_emoji_emb(x, \
                                                           emoji_2_embedding_lookup, emoji_symb2emb_dic,\
                                                                      distance_type)[:n_emoji] )
    return df


def print_samples_cluster(df, label, n_lines=5):
    '''
      Prints samples for each of the clusters 

      Parameters:
          df (DataFrame): DataFrame containing emoji_names, emoji_symbols and K_means labels
          label (str): the chosen label (for example in our case: labels_K_8 or labels_K_9)
          n_linles (int): the amount of lines printed per cluster (default = 5)

      Returns:
          Prints samples for each of the clusters 
    '''
       
    # Some Dataframe formatting for the printing
    pd.set_option('expand_frame_repr', False)
    pd.set_option('display.max_columns', 999)
    
    n = df[label].nunique()
    for i in range(n):
        print("\033[1m" + f'Samples from cluster nb {i} ' + "\033[0m")
        print(df[df[label]==i][['emoji_names', 'emoji_symbols','labels_K_8']].sample(n_lines))
        print("")




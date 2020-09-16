from tensorflow.keras.preprocessing import sequence, text
from prettytable import PrettyTable
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re

dict_label_encoding = pd.read_csv('df_emoji_le.csv', index_col=0)
dict_label_encoding.cat_col = dict_label_encoding.cat_col.apply(lambda i : re.sub("\[|\]|'", "", i))

def text_to_seq(X_train_raw, X_test_raw, max_length=7, vocab=5000):
    ''' 
    Function that changes texts to a sequence of indices with length max_length
        Arguments:
            X_train_raw: list, series of training sentences
            X_test_raw: list, series of test sentences
            max_length: int, maximum sentence length
    
        Returns:
            X_train_padded: padded list of indices
            X_test_padded: padded list of indices
            tokenizer: the tokenzier to be used for other test sentences

    '''
    train_seq= np.array([i.split() for i in X_train_raw])
    test_seq = np.array([i.split() for i in X_test_raw])
    
    tokenizer = text.Tokenizer(num_words=vocab)
    tokenizer.fit_on_texts(train_seq)
    
    X_train = tokenizer.texts_to_sequences(train_seq)
    X_test = tokenizer.texts_to_sequences(test_seq)
    
    X_train_padded = sequence.pad_sequences(X_train, maxlen=max_length)
    X_test_padded = sequence.pad_sequences(X_test, maxlen=max_length)
    
    return X_train_padded, X_test_padded, tokenizer

def plot_history(history):
    plt.plot(history.history["loss"], label="loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.legend()
    plt.show()
    plt.close()
    
    plt.plot(history.history["accuracy"], label="accuracy")
    plt.plot(history.history["val_accuracy"], label="accuracy")
    plt.legend()
    plt.show()
    plt.close()
        
def preprocess_sentence(sentence, tokenizer, max_len=7):
    '''
    Function to preprocess a custom sentence to go into the model
        Arguments:
            sentence : sentences that you want to perform a prediction
        
        Returns:
            padded_sequence: a sequence of indices using the tokenizer for input into the model
    
    '''
    split_sentences = sentence.split()
    tokenized_s_array = np.array(tokenizer.texts_to_sequences(split_sentences)).transpose()
    # the sentence must be longer than 2 words for this to work 
    padded_sequence = sequence.pad_sequences(sequence.pad_sequences(tokenized_s_array).transpose(), maxlen=max_len)
    
    return padded_sequence

def prediction(pre_sentence, model):
    ''' 
    Function to return the top class out of the categories
        Arguments:
            pre_sentence = the preprocessed sentence
            model = the trained model
        Returns:
            the top predicted category
    
    '''
    top_class = model.predict_classes(pre_sentence)[0]
    
    return dict_label_encoding.iloc[top_class, 0]


def prediction_table(pre_sentence, model, n_classes=3):
    ''' 
    Function to output n_classes of categories predicted with the probability
        Arguments:
            pre_sentence: preprocessed sentence
            model: model used for prediction 
            n_classes: the number of top categories to display
        
        Returns:
            prediction table: a table with the category and probability attributed to it 
    '''
    
    prediction = model.predict(pre_sentence)
    order_of_classes = np.argsort(-prediction)[0][:n_classes]
    probability_of_classes = -np.sort(-prediction)[0][:n_classes]

    t=PrettyTable(['Category', 'Probability'])

    for i, p in zip(order_of_classes, probability_of_classes):
        t.add_row([dict_label_encoding.iloc[i, 0], '{:.2f}%'.format(p*100)])
    
    print(t)
    


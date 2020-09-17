import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import tensorflow as tf
import tensorflow_datasets as tfds
from sklearn.metrics import confusion_matrix
from textblob import TextBlob
from textblob.translate import NotTranslated
import math

def textblob_dataugm(df, name_text_col='features', name_label_col='target'):
    '''
    Function to extract a list with all the descriptions and one with the related emojis
    '''
    descr_list = list(df[name_text_col])
    label_list = list(df[name_label_col])
    
    corr_desc = []
    updated_label = [] # In case a NotTransalted Error would be raisen
    
    for i, descr in enumerate(descr_list):
        print(f'Augmenting description job: {math.ceil(((i+1)/len(label_list))*100)}%')
        clear_output(wait=True)
        blob_descr = TextBlob(descr)
        try:
            fr_descr = blob_descr.correct().translate(from_lang='en', to ='fr')
        except NotTranslated:
            print('Error with ', blob_descr)

        try:
            en_descr = fr_descr.translate(from_lang='fr', to='en')
            updated_label.append(label_list[i]) # Only keeping emoji if no NotTranslatedError
            corr_desc.append(str(en_descr).replace("Embolism","emoji").replace("embolism","emoji"))
        except NotTranslated:
            pass
    
    assert len(corr_desc) == len(updated_label)

    # adding old and new lists together
    new_desc_list = descr_list + corr_desc
    new_label_list = label_list + updated_label

    help_dict = {"features": new_desc_list, "target": new_label_list}
    new_df = pd.DataFrame(help_dict)

    return new_df


def df_to_dataset(dataframe, shuffle=True, batching = True, batch_size=32):
    '''
    Function to create a tf.data.Dataset object from a pandas dataframe
    
    '''
    dataframe = dataframe.copy()
    labels = dataframe.pop('target')
    ds = tf.data.Dataset.from_tensor_slices((dataframe.squeeze(), labels.squeeze()))
    print(dataframe.squeeze().shape,labels.shape) 

    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))

    if batching:
        ds = ds.batch(batch_size)
    return ds

def sentence_to_dataset(sentence):
    '''
    Function to create a tf.data.Dataset object from a list of sentences 
    '''
    list_sentences = [sentence]
    df = pd.DataFrame(list_sentences, columns=['sentence'])
    ds = tf.data.Dataset.from_tensor_slices(df)

    return ds

def predict4_sentence(sentence, model):
    '''
    Function to predict the category of a sentence
    '''
    ds = sentence_to_dataset(sentence)
    pred = model.predict(ds)
    cat_num = np.argmax(pred)
    pred_cluster = num_2_name_lookup[cat_num]

    return pred_cluster

def predict4_tfdataset(test_features, test_labels, model, num_2_name_lookup):
    '''
    Function for making predictions on a given dataset
    Arguments:
        test_features: tensor
        test_labels: tensor
        num_2_name_lookup: dictionary
    '''
    predictions = model.predict(test_features)
    # converting the feateres and test_labels dataset to a numpy array
    features_2_numpy = tfds.as_numpy(test_features, graph=None)
    labels_2_numpy = tfds.as_numpy(test_labels, graph=None)

    # Looping over the predictions and real labels
    for i,pred in enumerate(predictions):
        # finding back the category
        pred_cluster = num_2_name_lookup[np.argmax(pred)]
        real_cluster = num_2_name_lookup[labels_2_numpy[i]]
        print(f'The sentence is : {features_2_numpy[i]}')
        print(f'The REAL CATEGORY is : {real_cluster} versus the PREDICTED CATEGORY : {pred_cluster}')
        print("")

def multiclass_confusion_mat(test_features, test_labels, model):
    '''
    Function to print the class confusion matrix 
    Arguments:
        test_features: tensor
        test_labels: tensor
    '''
    # making the predictions on the given dataset
    pred_proba = model.predict(test_features)
    predictions = np.argmax(pred_proba, axis=1) # prediction = arg of max proba

    # converting the test_labels dataset to a numpy array
    labels_2_numpy = tfds.as_numpy(test_labels, graph=None)

    plt.figure(figsize = [10,10])
    cf_matrix = confusion_matrix(labels_2_numpy, predictions)
    sb.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, fmt='.2%', cmap='Blues')
    plt.xlabel("Predicted Labels")
    plt.ylabel("Real Labels")        


def plot_history(history):
    '''
    Function to plot training and validation loss and accuracy for each epoch
    '''
    plt.plot(history.history["loss"], label="loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.legend()
    plt.show()
    plt.close()

    plt.plot(history.history["accuracy"], label="accuracy")
    plt.plot(history.history["val_accuracy"], label="val_accuracy")
    plt.legend()
    plt.show()
    plt.close()


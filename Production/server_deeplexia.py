# source : https://pythonbasics.org/what-is-flask-python/

from flask import Flask, render_template, redirect, url_for, request
import pandas as pd
import joblib
import numpy as np
import pickle
from scipy import spatial
import spacy
import nltk
from nltk.corpus import stopwords
from help_functions import *

app = Flask(__name__)

nlp = spacy.load('en_core_web_lg')
emoji_symb2emb_dic = joblib.load('avg_glove_embedding.pkl')
glove_lookup = joblib.load('glove_lookup')

@app.route('/', methods = ['POST', 'GET'])
def index():
   if request.method == 'POST':
      if request.form.get('sentences') == 'Sentences':
         text = request.form['text']
         emojis = translate_text(text)
         return render_template("index.html", result = emojis, origin_text = text)
      elif request.form.get('keywords') == 'Keywords':
         text = request.form['text']
         emoji_text = translate_by_keywords(text)
         return render_template("index.html", result= emoji_text, origin_text = text)

      elif request.form.get('reset') == 'Reset':
         return render_template("index.html", result='')

   else:
      return render_template("index.html")



if __name__ == '__main__':
   app.run(debug = True)


# Deeplexia

https://www.deeplexia.com

## Motivation

> **Deeplexia** is an application developed as a reading, learning and speed reading tool. With ever increasing amounts of reading materials produced in the modern age, speed reading could not be a more valuable skill. It is also a potential tool to help dyslexic people, who experience difficulty reading and have short term memory deficiencies. It is estimated that 65% of people are more visual learners, and since dyslexia is estimated to affect 10% of the world population, there is a clear need for a tool that makes text visual.

## Process

> 1. We scraped https://emojipedia.org to obtain descriptions of emojis and use this to create emoji embeddings. Find the code for this in [Emojipedia](Emojipedia).

> 2. [I.Text2Emoji.ipynb](Embeddings/I.Text2Emoji.ipynb) is the first experiment using spacy 'en_core_web_lg' to directly compute the similarity between emoji description and nouns of children's books.

> 3. [II.EmojiCluster.ipynb](Embeddings/II.EmojiCluster.ipynb) contains code to get an emoji embedding using the average of Glove word vectors for each of the words in the emoji description. We used this averaging method to get sentence embeddings and compute cosine similarities. T-SNE is used to plot the emoji embeddings as points in a reduced dimension space. Cluster analysis is performed using K-means and the emoji categories in preparation for input into a classifier. [III.OtherEmbeddings.ipynb](Embeddings/III.OtherEmbeddings.ipynb) contains experiments with Spacmoji and Zalando Flair Transformers.

> 4. [TensorFlowHub](Embeddings/TensorFlowHub) and [LSTMs](Embeddings/LSTMs) contain examples of training a classifier with the emoji description to predict the emoji category with data-augmentation. This is then tested on sentences from children's books. [choose_best_emoji.py](Embeddings/choose_best_emoji.py) is a script that can be used to perform the manual data labelling required to get more data to produce better results with classifiers.

> 5. [Production](Production) contains the files needed to run a heroku or pythonanywhere web app.

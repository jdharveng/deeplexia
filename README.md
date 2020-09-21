# Deeplexia


https://giphy.com/gifs/il1yesdofGlZ6/html5

## Motivation

Deeplexia is an application developed as a reading, learning and speed reading tool. With the ever increasing amounts of reading materials produced in the modern age, speed reading could not be a more valuable skill. It is also a potential tool to help dyslexic people, who experience difficulty reading and have issues with short term memory. It is estimated that 65% of people are more visual learners, and since dyslexia is estimated to affect 10% of the world population the need for a tool to visualise text is clear. 

## Process

<iframe src="https://giphy.com/embed/NFA61GS9qKZ68" width="480" height="270" frameBorder="0" class="giphy-embed" allowFullScreen></iframe><p><a href="https://giphy.com/gifs/reading-dot-strategies-NFA61GS9qKZ68">via GIPHY</a></p>

1. We scraped https://emojipedia.org to obtain descriptions of emojis and use this to create Emoji Embeddings. Find the code for this in the [Emojipedia folder](Emojipedia).

2. [I.Text2Emoji.ipynb](Embeddings/I.Text2Emoji.ipynb) is the first experiment using spacy 'en_core_web_lg' to directly compute the similarity between Emoji description and nouns of children's book


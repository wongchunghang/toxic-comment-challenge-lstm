# Objective
The open source project aims to tackle a live problem about classification on negative online behavior (i.e. toxic comments) such as rude, disrespectful or otherwise likely to make someone leave a discussion.

The challenge is originated from Kaggle, an online machine learning competition platform.

The model is based on Implementation of Kaggler Larry Freeman(https://www.kaggle.com/larryfreeman/toxic-comments-code-for-alexander-s-9872-model)

# Architecture
First layer: concatenated fasttext and glove twitter embeddings. Fasttext vector is used by itself if there is no glove vector but not the other way around. Words without word vectors are replaced with a word vector for a word "something". Also, I added additional value that was set to 1 if a word was written in all capital letters and 0 otherwise.

Second layer: SpatialDropout1D(0.5)

Third layer: Bidirectional CuDNNLSTM with a kernel size 40. I found out that LSTM as a first layer works better than GRU.

Fourth layer: Bidirectional CuDNNGRU with a kernel size 40.

Fifth layer: A concatenation of the last state, maximum pool, average pool and two features: "Unique words rate" and "Rate of all-caps words"

Sixth layer: output dense layer.

# Hyperparameters and preprocessing:
Batch size: 512. I found that bigger batch size makes results more stable.
Epochs: 15.
Sequence length: 900.
Optimizer: Adam with clipped gradient.
Preprocessing: Unidecode library (https://pypi.python.org/pypi/Unidecode) to convert text to ASCII first and after that filtering everything except letters and some punctuation.
# Text normalization
I did a lot of work on fixing misspellings and I think it improved the score. I was only fixing misspellings that didn't have a fasttext vector. Things that I did:

Created a list of words that appear more often in toxic comments than in regular comments and words that appear more often in non-toxic comments. For every misspelled word I looked up if it has a word in the list with a small Levenshtein distance to it.
Fixed some misspellings with TextBlob dictionary.
Fixed misspellings by finding word vector neighborhoods. Fasttext tool can create vectors for out-of-dictionary words which is really nice. I trained my own fasttext vectors on Wikipedia comments corpus and used them to do this. I also used those vectors as embeddings but results were not as good as with regular fasttext vectors.

# Quick Start
Open the ipython notebook toxic_comment_9872_model.ipynb in colab (colab.research.google.com) and run in GPU runtime.

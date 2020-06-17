#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
tf.debugging.set_log_device_placement(True)

# 텐서 생성
a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
c = tf.matmul(a, b)

print(c)


# In[10]:


import tensorflow as tf
import csv
import matplotlib.pyplot as plt
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from keras.preprocessing import sequence
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from tensorflow.keras.datasets import imdb
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics
get_ipython().run_line_magic('matplotlib', 'inline')

tf.debugging.set_log_device_placement(True)

# 중립적인 문장 제거
source = ["cnn", "forbes", "foxnews", "nyt", "reddit"]

neu_verb = ["change", "suggest", "suggests", "suggested", "introduce", "introduces", 
                "introduced", "think", "thought", "try", "tries", "claims", "claim", "claimed", 
                "say", "says", "said", "tells", "told", "tell", "declare", "declares", "declared", 
                "deliver", "give", "gives", "delivers", "delivered", "gave", "signs", "sign", "signed",
                "announces", "announce", "announced"]


sid = SentimentIntensityAnalyzer()
en_stopwords = stopwords.words('english')


def sentencePreprocess(sentences):
    # tagged_sentences = [word_tokenize(sent) for sents in test_sents for sent in sent_tokenize(sents)]
    sent_tagged_pairs = [(sent, pos_tag(word_tokenize(sent)), news, url, date) for (sent, news, url, date) in sentences]
    return sent_tagged_pairs

def neuVerb_filter(tagged_sentence):
    for (word, tag) in tagged_sentence:
        if (tag in ["VB", "VBD", "VBP", "VBZ"]):
            if (word in ["is", "was", "are", "were", "am", "be"]):
                return True
            elif (word in neu_verb):
                return False
            elif (sid.polarity_scores(word)['compound'] == 0):
                return False
            else: return True


def polarScore_filter(sentence):
    sent_polarity_scores = sid.polarity_scores(sentence)
    if (abs(sent_polarity_scores['compound']) < 0.3):
        return 0
    elif (sent_polarity_scores['compound'] >= 0.3):
        return sent_polarity_scores['compound']
    else:
        return sent_polarity_scores['compound']


def nosto_filter(word):
    if word in en_stopwords:
        return False
    return True


for s in sourcs:
    f = open('{}_data.csv'.format(s),'r')
    rdr = csv.reader(f)

    sent_tagged_pairs = sentencePreprocess(rdr)
    f.close()

    filtered_pairs = [(sent, tagged_sent, news, url, date) for (sent, tagged_sent, news, url, date) in sent_tagged_pairs
                                                    if ((neuVerb_filter(tagged_sent)) or (polarScore_filter(sent)))]

    print(len(filtered_pairs))

    f = open('{}_data_filtered.csv'.format(s), 'w', newline='')
    wr = csv.writer(f)
    for (sent, tagged_sent, news, url, date) in filtered_pairs:
        wr.writerow([sent, news, url, date])

    f.close()


# ML Training

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

print("Shape of x_train :", x_train.shape)
print("Shape of x_test :", x_test.shape)

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

print("Shape of y_train :", y_train.shape)
print("Shape of y_test :", y_test.shape)

x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]


model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))


model.compile(optimizer=optimizers.RMSprop(lr=0.001),
             loss=losses.binary_crossentropy,
             metrics=[metrics.binary_accuracy])

history = model.fit(partial_x_train,
                  partial_y_train,
                  epochs=15,
                  batch_size=512,
                  validation_data=(x_val, y_val))

history_dict = history.history
history_dict.keys()

loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label="Training loss")
plt.plot(epochs, val_loss, 'b', label="Validation loss")
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.show()

plt.clf()
acc = history_dict['binary_accuracy']
val_acc = history_dict['val_binary_accuracy']

plt.plot(epochs, acc, 'bo', label="Training acc")
plt.plot(epochs, val_acc, 'b', label="Validation acc")
plt.title("Training and validation acc")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.show()

word2index = imdb.get_word_index()

def predict_pos_neg(review):
    token = word_tokenize(review)
    seq = []
    for t in token:
        try:
            seq.append(word2index[t])
        except:
            continue
    test=sequence.pad_sequences([seq],maxlen=10000)
    score = float(model.predict(test))
    if(score > 0.5):
#         print("[{}] is predicted as a positive sentence with {:.2f}% probability.".format(review, score * 100))
        return 1
    else:
#         print("[{}] is predicted as a negative sentence with {:.2f}% probability.".format(review, (1 - score) * 100))
        return 0

# Samples accuracy(with annotation sample data)
sources = ["cnn", "forbes", "foxnews", "nyt", "reddit"]
sentences = []
sample_sentences = []

for s in sources:
    with open('{}_data_filtered.csv'.format(s)) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for line in readCSV:
            if not line: break
            l = line[0]
            sentences.append((l, -1))
            
with open('annotated.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for line in readCSV:
        if not line: break
        l = line[0]
        r = int(line[4])
        sample_sentences.append((l, r))

acc = 0
for (s, res) in sample_sentences:
    temp = 0
    if res == 1:
#         print("Actually, It is a positive sentence.")
        temp = 1
    else:
#         print("Actually, It is a negative sentence.")
        temp = 0
    if predict_pos_neg(s.lower()) == temp:
        acc+=1
        
get_ipython().system('clear')
# 여론 결과 도출
total = len(sentences)
pos = 0
neg = 0
for (s, res) in sentences:
    res = predict_pos_neg(s.lower())
    if res == 1:
        pos+=1
    else:
        neg+=1

print("-"*9+"Samples"+"-"*9)
print("Correct Answer : {}".format(acc))
print("Wrong Answer : {}".format(len(sample_sentences)-acc))
print("RESULT ACCURACY : {:.1f}% ".format(acc/len(sample_sentences)*100))
print("-"*25)

print("-"*9+"Opinion"+"-"*9)
print("Positive : {:.2f}".format(pos/total*100))
print("Negative : {:.2f}".format(neg/total*100))
print("-"*25)


# In[ ]:





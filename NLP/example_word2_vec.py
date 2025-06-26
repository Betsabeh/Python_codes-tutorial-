!pip install nltk
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize
import gensim
from gensim.models import Word2Vec
import numpy as np
from sklearn.ensemble import RandomForestClassifier


#  simple text
s=[]
s.append('Dino the family dog helps to keep people safe on the roads.')
s.append('Angel goes on an adventure with his friend. What dangerous sea animals will he meet? ')
s.append("There's a strange new animal at the zoo")
s.append("Ali has found a magic carpet in his uncle's shop.")


# tokenization 
Y_Train=[]
data = []
for doc in s:
  d=[]
  for sent in nltk.tokenize.sent_tokenize(doc):
    #print(sent)
    temp = []
    # tokenize the sentence into words
    for word in nltk.tokenize.word_tokenize(sent):
        #print(word)
        temp.append(word.lower()) 
         
  data.append(temp)
  Y_Train.append(1)  

Y_Train=np.array(Y_Train)
Y_Train[3]=0
Y_Train=np.reshape(Y_Train,(4,1))

print('===================Our corpus===================')
print(data)    
#word2vec
# Create CBOW model
model = gensim.models.Word2Vec(data, min_count = 1,size=20, window = 5)
# Vocab
print('====================Most similar word to Book===================')
print(model.wv.most_similar('animal'))

#Generate aggregated sentence vectors based on the word vectors for each word in the sentence
words = set (model.wv.index2word)
print('======================All words=======================')
print(words)

X_train_vect =[]
for ls in data:
    temp=[]
    for i in ls: 
      if i in words:
        temp.append(model.wv[i])

    X_train_vect.append(temp)

X_train_vect=np.array(X_train_vect)

print("shape train=",np.shape(X_train_vect))
print("shape train feature vec=",np.shape(X_train_vect[0]))


# Why is the length of the sentence different than the length of the sentence vector?
for i in range(4):
    print("sentece: ", data[i], "word2vec features=", X_train_vect[i])


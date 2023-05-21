from sklearn.utils.fixes import loguniform
import numpy as np
import pandas as pd
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
import matplotlib.pyplot as plt
import re

nltk.download('punkt')
nltk.download('stopwords')
#-------------------------------------------
#1-read dataset
Data=pd.read_csv('movie.csv')
#print(Data.info)
Label=Data['sentiment']
#print(Label)
Label=(Label=='positive')
Label=np.int16(Label)
Text=Data['review']
#print(Label)
#print(type(Text))
#----------------------------------------------
#2-remove tags
def preprocessing(txt):
  #print(txt)
  #remove html tag
  txt=re.sub('<[^>]*>','',txt)
  emotion=re.findall('(?::|;|=)(?:=)?(?:\)|\(D|P)',txt)
  txt=re.sub('[\W]+',' ',txt)
  #print(txt)
  # lower case
  txt=txt.lower()
  emotion=' '.join(emotion)
  emotion=emotion.replace('-','')
  txt=txt+emotion
  return txt

New_T=Text
for i in range(len(Text)):
  t=preprocessing(Text[i]) 
  New_T[i]=t

print(New_T)
#print(type(New_T))
#print(np.shape(New_T))
#----------------------------------------------
#2-Tokenization and steming and stop word removal
def token_porter(Text):
  porter=nltk.stem.PorterStemmer()
  stop_word=stopwords.words('english')
  #print(stop_word)
  New_T=Text
  for i in range(len(Text)):
    #print(sent)
     temp=nltk.word_tokenize(Text[i],language='english')
     tokens=''
     for word in temp:
       if word not in stop_word:
           t=porter.stem(word)
           tokens=tokens+' '+t
     New_T[i]=tokens
          
  return New_T

New_T=token_porter(Text)
print(New_T)
#--------------------------------------------------
# 3-calculate TF-IDF
count=CountVectorizer()
Text=np.array(New_T)
tfidf=TfidfTransformer(use_idf=True,smooth_idf=True,norm="l2")
tfidf_value=tfidf.fit_transform(count.fit_transform(New_T)) # recieved array of string
tfidf_value=tfidf_value.toarray()
print("TFIDFvalues:")
print(tfidf_value)
#-------------------------------------------------
#4-Train Classifier
#classifier hyperparamter tuning
d=dict()
d['penalty']=['none','l1','l2']
d['solver']=['newton-cg', 'sag', 'lbfgs']
mdl=LogisticRegression()
Search=GridSearchCV(mdl,d,scoring='accuracy',n_jobs=-1)
result=Search.fit(tfidf_value,Label)
print('--------------------------------------------')
print("best score:",result.best_score_)
print("best_param:",result.best_params_)



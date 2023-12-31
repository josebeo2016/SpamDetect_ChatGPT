import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import email
import string
from bs4 import BeautifulSoup
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import os
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_recall_curve 
from sklearn.metrics import roc_auc_score 
from sklearn.metrics import roc_curve
import os
import warnings
warnings.filterwarnings('ignore')
np.random.seed(49)
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
class email_to_clean_text(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None): 
        return self
    def transform(self, X):
        text_list = []
        for mail in X:
            b = email.message_from_string(mail)
            body = ""

            if b.is_multipart():
                for part in b.walk():
                    ctype = part.get_content_type()
                    cdispo = str(part.get('Content-Disposition'))

                    # skip any text/plain (txt) attachments
                    if ctype == 'text/plain' and 'attachment' not in cdispo:
                        body = part.get_payload(decode=True)  # get body of email
                        break
            # not multipart - i.e. plain text, no attachments, keeping fingers crossed
            else:
                body = b.get_payload(decode=True) # get body of email
            #####################################################
            soup = BeautifulSoup(body, "html.parser") #get text from body (HTML/text)
            text = soup.get_text().lower()
            #####################################################
            text = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', text, flags=re.MULTILINE) #remove links
            ####################################################
            text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text, flags=re.MULTILINE) #remove email addresses
            ####################################################
            text = re.sub(r'[^a-zA-Z0-9,.;\s]', '', text) # remove punctuation
            ####################################################
            text = re.sub(r'\s+', ' ', text) # remove extra whitespace
            ####################################################
            text = text.strip() #remove whitespaces
            
            
            text_list.append(text)
        return text_list
def get_data(path):
    data = []
    files = os.listdir(path)
    for file in files:
        f = open(path+file, encoding = "ISO-8859-1")
        words_list = f.read()
        data.append(words_list)
        f.close()
    return data

easy_ham_path = 'data/spamassassin/easy_ham/easy_ham/'
hard_ham_path = 'data/spamassassin/hard_ham/hard_ham/'
spam_path = 'data/spamassassin/spam_2/spam_2/'

easy_ham = get_data(easy_ham_path)
hard_ham = get_data(hard_ham_path)
ham = easy_ham + hard_ham
spam = get_data(spam_path)

np.random.shuffle(ham)
np.random.shuffle(spam)

email_to_text = email_to_clean_text()

text_ham = email_to_text.transform(ham)
text_spam = email_to_text.transform(spam)

df = pd.DataFrame(columns=['text', 'label'])
df['text'] = text_ham + text_spam
df['label'] = ['ham']*len(text_ham) + ['spam']*len(text_spam)

df.to_csv('data/spamassassin_1.csv', index=False)
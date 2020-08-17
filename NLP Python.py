import numpy as np
import pandas as pd
import sqlite3
import re
from tqdm import tqdm
import seaborn as sns
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
pip install wordcloud
# from sklearn.preprocessing import StandardScaler

from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, roc_curve, recall_score
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import ngrams

# from imblearn.over_sampling import SMOTE
# from imblearn.under_sampling import RandomUnderSampler


df = pd.read_csv("Train.csv")
df.head()
df = df.iloc[:10000, :]
df.drop_duplicates('Title', inplace=True)
df.shape

# get number of tags for each title
df['Tag_count'] = df['Tags'].apply(lambda x: len(str(x).split()))

df.dropna()
df.shape

df.isnull().sum()

#plot distribution of tag count
fig = plt.figure(figsize=[10,7])
sns.countplot(df['Tag_count'])
plt.title('Distribution of tag count')
plt.ylabel('Frequency')
plt.xlabel('Tag count')
plt.show()



def clean_text(sentence):
    # remove non alphabetic sequences
    pattern = re.compile(r'[^a-z]+')
    sentence = sentence.lower()
    sentence = pattern.sub(' ', sentence).strip()
    
    # Tokenize
    word_list = word_tokenize(sentence)
    # stop words
    stopwords_list = set(stopwords.words('english'))
    # remove stop words
    word_list = [word for word in word_list if word not in stopwords_list]
    # stemming
    ps  = PorterStemmer()
    word_list = [ps.stem(word) for word in word_list]
    # list to sentence
    sentence = ' '.join(word_list)
    
    return sentence

# create tqdm for pandas
tqdm.pandas()
# clean text data
df['Title'] = df['Title'].progress_apply(lambda x: clean_text(str(x)))

# calculate number of questions covered by top n tags
def questions_covered(one_hot_tag, ntags):
    # number of questions
    nq = one_hot_tag.shape[0]
    # get number of questions covered by each tag
    tag_sum = one_hot_tag.sum(axis=0).tolist()[0]
    # sort tags based on number of questions covered by them
    tag_sum_sorted = sorted(range(len(tag_sum)),
                            key=lambda x: tag_sum[x],
                            reverse=True)
    # get one hot encoded matrix for top n tags
    one_hot_topn_tag = one_hot_tag[:, tag_sum_sorted[:ntags]]
    # get number of tags per question
    tags_per_question = one_hot_topn_tag.sum(axis=1)
    # get number of question with no tags
    q_with_0_tags = np.count_nonzero(tags_per_question == 0)
    
    return np.round((nq - q_with_0_tags)/nq*100, 2)

# get number of questions covered and tag id list
def questions_covered_list(one_hot_tag, window):
    # number of tags
    ntags = one_hot_tag.shape[1]
    # question id list
    qid_list = np.arange(100, ntags, window)
    # questions covered list
    ques_covered_list = []
    for idx in range(100, ntags, window):
        ques_covered_list.append(questions_covered(one_hot_tag, idx))
        
    return qid_list, ques_covered_list

# get multinomial tag matrix (top n tags)
def topn_tags(one_hot_tag, ntags):
    # get number of questions covered by each tag
    tag_sum = one_hot_tag.sum(axis=0).tolist()[0]
    # sort tags based on number of questions covered by them
    tag_sum_sorted = sorted(range(len(tag_sum)),
                            key=lambda x: tag_sum[x],
                            reverse=True)
    # get one hot encoded matrix for top n tags
    one_hot_topn_tag = one_hot_tag[:, tag_sum_sorted[:ntags]]
    return one_hot_topn_tag
# using bag of words to represent tags for each title
tag_vectorizer = CountVectorizer(tokenizer= lambda x: str(x).split(), binary=True)
y_multinomial = tag_vectorizer.fit_transform(df['Tags'])
x, y = questions_covered_list(y_multinomial, 100)
fig = plt.figure(figsize=[10,7])
plt.title('Questions covered Vs Numbre of Tags')
plt.ylabel('Percentage of Questions covered')
plt.xlabel('Number of Tags')
plt.plot(x,y, c=sns.xkcd_rgb['greenish cyan'])
plt.show()


y_multinomial = topn_tags(y_multinomial, 100)
# get index of questions covered
# and remove rest of the data
non_zero_idx = y_multinomial.sum(axis=1) != 0
non_zero_idx = non_zero_idx.A1
y_multinomial = y_multinomial[non_zero_idx,:]
train = df.iloc[non_zero_idx, :]
y_multinomial.shape, train.shape
# create tqdm for pandas
tqdm.pandas()
# clean text data
train['Title'] = train['Title'].progress_apply(lambda x: clean_text(str(x)))

# split data in 80-20 ratio
Xtrain, Xtest, Ym_train, Ym_test = train_test_split(train['Title'], y_multinomial, test_size=0.2, random_state=45)

# vectorize text data
tfid_vec = TfidfVectorizer(tokenizer=lambda x: str(x).split())
Xtrain = tfid_vec.fit_transform(Xtrain)
Xtest = tfid_vec.transform(Xtest)
Xtrain.shape, Xtest.shape

# create model instance
logreg_model1 = OneVsRestClassifier(SGDClassifier(loss='log',
                                                  alpha=0.001,
                                                  penalty='l1'),
                                   n_jobs=-1)
# train model
logreg_model1.fit(Xtrain, Ym_train)
# predict tags
Ym_test_pred = logreg_model1.predict(Xtest)

# print model performance metrics
print("Accuracy :",metrics.accuracy_score(Ym_test,Ym_test_pred))
print("f1 score macro :",metrics.f1_score(Ym_test,Ym_test_pred, average = 'macro'))
print("f1 scoore micro :",metrics.f1_score(Ym_test,Ym_test_pred, average = 'micro'))
print("Hamming loss :",metrics.hamming_loss(Ym_test,Ym_test_pred))
print("Precision recall report :\n",metrics.classification_report(Ym_test,Ym_test_pred))

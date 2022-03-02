import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
import pickle
import pandas as pd
import numpy as np
import datetime as dt
import spacy
import plotly_express as px


import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)


# Load Models
lda_model =  gensim.models.LdaModel.load('data/LDA/lda.model')
id2word = corpora.Dictionary.load('data/LDA/lda.model.id2word')

with open("data/cluster_model.pkl", "rb") as f:
    cluster_model = pickle.load(f)


### Calculate User Clusters Section

users = pd.read_csv('data/users.csv')
interests = pd.read_csv('data/interest.csv')

# Create buckets for the age of users

users['date'] =  pd.to_datetime(users['dob'], format='%d/%m/%Y', errors='coerce')

users['age'] = dt.datetime.now() - users['date']
users['age'] = (users['age']).dt.days
users['age'] = users['age']/365

users['age_cat'] = np.where(users['age']<20,1,
                   np.where((users['age']>=20) & (users['age']<25),2,
                   np.where((users['age']>=25) & (users['age']<30),3,
                   np.where((users['age']>=30) & (users['age']<35),4,
                   np.where((users['age']>=35) & (users['age']<40),5,
                   np.where((users['age']>=40) & (users['age']<45),6,
                   np.where((users['age']>=45) & (users['age']<50),7,
                   np.where((users['age']>=50) & (users['age']<55),8,
                   np.where((users['age']>=55) & (users['age']<60),9,
                   np.where((users['age']>=60) & (users['age']<65),10,11))))))))))



user_age = users[['uid', 'age_cat']]

user = pd.merge(users,interests, left_on='uid', right_on='uid', how='left')

# Load categories saved
cats = pd.read_csv('data/interest_cats.csv')

user = pd.merge(user,cats, left_on='interest', right_on='categories')
rows=len(users['uid'])
cols=len(cats['id']) # add 2 extra columns for age buckets and uid

matrix = pd.DataFrame(np.zeros((rows,cols)))
data = pd.concat([user_age.reset_index(drop=True), matrix], axis=1)

for i in range(1,len(user['uid'])):
    #get row number
    uid = user['uid'][i]
    rnum = data.index[data['uid']==uid]
    col = user['id'][i]+1
    data.iloc[rnum.values[0],col] = 1


data.drop(columns='uid',axis=1,inplace=True)

# Calculate cluster values
clusters = cluster_model.predict(data)
print(clusters)

### Calculate Post Topics Section

# Import data
words = pd.read_csv('data/stop_words.csv')
stop_words = words['0'].tolist()

users = pd.read_csv('data/users.csv')
interests = pd.read_csv('data/interest.csv')
posts = pd.read_csv('data/posts.csv')

data = posts['text'] + posts['hashtags']

#  Preprocess data
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

data_words = list(sent_to_words(data))


# Build the bigram and trigram models
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

# Functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

# Remove Stop Words
data_words_nostops = remove_stopwords(data_words)

# Form Bigrams
data_words_bigrams = make_bigrams(data_words_nostops)

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

# Create Corpus
texts = data_lemmatized

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]
vector = lda_model[corpus]


def getKey(item):
    return item[1]

topic_probability = []
for t in vector:
      print(sorted(t[0],key=getKey,reverse=True))
      topic_probability.append(sorted(t[0],key=getKey,reverse=True))

results_test = pd.DataFrame(topic_probability,columns=['col 1','col 2',
                                                       'col 3','col 4','col 5'])


results_test['topic'] = 0
results_test['topic_val'] = float(0)

for i in range(0,len(results_test['col 1'])):

        val = results_test.iloc[i,0]
        results_test['topic'][i] = val[0]
        results_test['topic_val'][i] = val[1]

# Visualise the cluster grouping to post topics.

enriched_posts = pd.concat([posts.reset_index(drop=True), results_test[['topic','topic_val']]], axis=1)
enriched_posts = pd.concat([enriched_posts.reset_index(drop=True), pd.DataFrame(clusters)], axis=1)
enriched_posts.columns.values[8] = 'cluster'

# Rank the topics by cluster

groups = enriched_posts.groupby(['cluster', 'topic'])['uid'].count().unstack(fill_value=0).stack()
groups = groups.reset_index()
groups.columns.values[2] = 'counts'
groups['ratio'] = groups.groupby(['cluster'], group_keys=False).apply(lambda g: g.counts/(g.counts).sum())
groups['ratio'] = np.where(groups['ratio'] == 0, 0.0001,groups['ratio'])
groups['rank'] = groups.groupby(['cluster'], group_keys=False)['ratio'].rank('dense',ascending=False,)

groups.to_csv('data/cluster_mapping.csv',index=False)

print(groups)


# Visualise the cluster and topic groups

fig = px.histogram(groups, x="cluster", y="counts",
             color='topic', barmode='group',
             height=400)
fig.show()
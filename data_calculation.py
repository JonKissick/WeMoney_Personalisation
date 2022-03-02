import pandas as pd
import pickle
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
import numpy as np
import datetime as dt

from LDA import remove_stopwords, lemmatization, make_bigrams, sent_to_words

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

# LOAD CLUSTERING MODEL
with open("data/cluster_model.pkl", "rb") as f:
    cluster_model = pickle.load(f)

# LOAD LDA MODEL
lda_model = gensim.models.LdaModel.load('data/LDA/lda.model')
id2word = corpora.Dictionary.load('data/LDA/lda.model.id2word')


def get_interests():
    """
    Load the raw interest csv file.
    :return: The full interest.csv file in pandas dataframe
    """

    interest = pd.read_csv('data/interest.csv')

    return(interest)

def get_posts():
    """
    Load the raw posts csv file.
    :return: The full posts.csv file in pandas dataframe
    """

    posts = pd.read_csv('data/posts.csv')

    return(posts)

def get_users():
    """
    Load the raw users csv file.
    :return: The full users.csv file in pandas dataframe
    """

    users = pd.read_csv('data/users.csv')

    return(users)

def filter_posts(uid,date):
    """
    Returns posts that have been filtered to be before a given date and aren't owned by the user
    :param uid (str): user-id to filter by
    :param date (str): date value to filter by
    :return: pandas dataframe filtered of any posts greater than date and not owned by user
    """
    posts = get_posts()

    posts = posts[posts['uid'] != uid]
    posts = posts[posts['post_time'] < date]

    return posts


def get_user_data(uid):
    """
    Returns the selected user account information
    :param uid (str): user-id
    :return: single-row pandas dataframe of user account information
    """
    users = get_users()

    user = users[users['uid'] == uid].reset_index(drop=True)
    return user


def get_user_interest(uid):
    """
    Returns the selected user interest information
    :param uid (str): user-id
    :return: single-row pandas dataframe of user interest information
    """
    interests = get_interests()

    interest = interests[interests['uid'] == uid].reset_index(drop=True)
    return interest


def cluster_user(uid):
    """
    Returns categorised ID of the selected user from the clustering model
    :param uid (str): user-id
    :return: single integer value of ID category
    """
    # Load needed data for user
    users = get_user_data(uid)
    interests = get_user_interest(uid)

    # Create Age Buckets for clustering
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

    # Load the categories in order used to cluster
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

    # Use model to predict grouping of user
    clusters = cluster_model.predict(data)


    return clusters[0]


def get_post_topics(uid,date):
    """
    Returns the filtered list of posts that are enriched by topic information
    :param uid: user-id (str)
    :param date: datetime (str)
    :return: sorted dataframe of filtered posts with topic information
    """

    # Filter posts for datetime given

    posts = filter_posts(uid, date)

    data = posts['text'] + posts['hashtags']

    # Preprocess data
    data_words = list(sent_to_words(data))

    # Remove Stop Words
    data_words_nostops = remove_stopwords(data_words)

    # Form Bigrams
    data_words_bigrams = make_bigrams(data_words_nostops)

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
        topic_probability.append(sorted(t[0], key=getKey, reverse=True))

    results = pd.DataFrame(topic_probability, columns=['col 1', 'col 2',
                                                       'col 3', 'col 4', 'col 5'])

    results['topic'] = 0
    results['topic_val'] = float(0)

    for i in range(0, len(results['col 1'])):
        val = results.iloc[i, 0]
        results['topic'][i] = val[0]
        results['topic_val'][i] = val[1]

    topics = results[['topic', 'topic_val']]

    return topics


def enrich_posts(uid,date):
    """
    Returns the final list of ranked posts
    :param uid: user-id (str)
    :return: sorted dataframe of ranked posts relevent to a certain user given a date
    """

    # Convert date to datetime
    timestamp = pd.Timestamp(date)

    # Filter posts
    posts = filter_posts(uid,date)

    # Get relevent ranking system of post topic interests
    cluster_cat=cluster_user(uid)
    rankings = pd.read_csv('data/cluster_mapping.csv')
    ranked = rankings[rankings['cluster']==cluster_cat]

    topics = get_post_topics(uid,date)

    ranked_topics = pd.merge(topics,ranked, left_on='topic', right_on = 'topic')

    # Calculate statistics used to rank
    enriched_posts = pd.concat([posts.reset_index(drop=True), ranked_topics], axis=1)
    enriched_posts['parent_flag'] = np.where(enriched_posts['parent_id'] != '0',0,1)
    enriched_posts['datetime'] =  pd.to_datetime(enriched_posts['post_time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

    enriched_posts['age_of_post'] = (timestamp - enriched_posts['datetime'])
    enriched_posts['clean_parent_id'] = np.where(enriched_posts['parent_id']=='0',enriched_posts['post_id'],enriched_posts['parent_id'])

    age_stats = enriched_posts.groupby(['clean_parent_id']).agg({'age_of_post':['mean','min','max']})
    age_stats.columns = ['_'.join(col) for col in age_stats.columns]
    age_stats.reset_index(level=0, inplace=True)

    age_stats['age_rank'] = age_stats['age_of_post_min'].rank(ascending=True)

    rating_stats = enriched_posts.groupby(['clean_parent_id']).agg({'ratio': ['mean', 'sum','count']})
    rating_stats.columns = ['_'.join(col) for col in rating_stats.columns]
    rating_stats.reset_index(level=0, inplace=True)

    enriched_posts = pd.merge(enriched_posts,age_stats,left_on='clean_parent_id',right_on='clean_parent_id',how='left')
    enriched_posts = pd.merge(enriched_posts,rating_stats,left_on='clean_parent_id',right_on='clean_parent_id',how='left')

    # Rank posts according to methodology
    enriched_posts['seconds'] = (enriched_posts['age_of_post_min']).dt.total_seconds()
    enriched_posts['rating'] = (enriched_posts['ratio_mean'] * enriched_posts['ratio_sum'] * (1000000/enriched_posts['seconds'])**3)
    enriched_posts['post_rank'] = enriched_posts['rating'].rank(ascending=False)

    # Clean and sort values to send
    final_posts = enriched_posts[['uid', 'post_time', 'text', 'hashtags', 'post_id', 'parent_id', 'post_rank']]
    final_posts = final_posts.sort_values('post_rank').reset_index(drop=True)

    return final_posts


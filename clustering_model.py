from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import datetime as dt
import pickle
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore",category=FutureWarning)

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

# Create categories and index to be used in production so prod is same as training

cats = pd.DataFrame(user['interest'].unique())
cats.columns = ['categories']
cats = cats.sort_values('categories').reset_index(drop=True)
cats['id'] = cats.index +1

# Create fitting dataset

user = pd.merge(user,cats, left_on='interest', right_on='categories')
rows=len(users['uid'])
cols=len(cats['id'])

matrix = pd.DataFrame(np.zeros((rows,cols)))
data = pd.concat([user_age.reset_index(drop=True), matrix], axis=1)

for i in range(1,len(user['uid'])):
    #get row number
    uid = user['uid'][i]
    rnum = data.index[data['uid']==uid]
    col = user['id'][i]+1
    data.iloc[rnum.values[0],col] = 1


data.drop(columns='uid',axis=1,inplace=True)

# Determine a sensible number of clusters

kmeans_kwargs = {
"init": "random",
"n_init": 10,
"max_iter": 300,
"random_state": 42,
}

# A list holds the SSE values for each k

sse = []
for k in range(1, 11):
   kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
   kmeans.fit(data)
   sse.append(kmeans.inertia_)

plt.plot(range(1, 11), sse)
plt.xticks(range(1, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
#plt.show()  # Uncomment to see SSE/Clusters graph
# Elbow point = 4

# Create and fit model

kmeans = KMeans(n_clusters=4, init = 'random', n_init=10, max_iter=250, random_state=111)
clusters = kmeans.fit(data)
print(clusters.labels_)

# Save trained clustering model and categories

with open("data/cluster_model.pkl", "wb") as f:
    pickle.dump(clusters, f)

cats.to_csv('data/interest_cats.csv',index=False)
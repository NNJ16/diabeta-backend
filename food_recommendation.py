import pandas as pd
import pickle
import numpy as np

df = pd.read_csv("Exercise.csv")
df.head(10)

#remove duplicate exersises
df.drop_duplicates(subset=['Exercise'], inplace= True)

# convert to lowercase and remove spaces
def clean(sentence):
    temp = ""
    for word in sentence:
        temp = temp + (word.lower().replace(' ', ''))
    return temp

df['Lifestyle'] = [clean(x) for x in df['Lifestyle']]
df['Category'] = [clean(x) for x in df['Category']]
df['Activity-level'] = [clean(x) for x in df['Activity-level']]
df['PreDiabetic'] = df ['PreDiabetic'].replace({1:'prediabetic',0:'nonprediabetic' })
df['Diabetic'] = df ['Diabetic'].replace({1:'diabetic',0:'nondiabetic' })

# combining all the columns data
columns = ['Lifestyle', 'Category', 'Activity-level', 'PreDiabetic', 'Diabetic']

df["life_cat"] = df[['Lifestyle', 'Category']].apply("".join, axis=1)
df["clean_input"] = df[['life_cat', 'Activity-level', 'PreDiabetic', 'Diabetic']].apply(" ".join, axis=1)
df["clean_input"]

df = df[['Exercise', 'clean_input']]
df.reset_index(inplace=True)

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
tfidf = TfidfVectorizer()
features = tfidf.fit_transform(df['clean_input'])

from sklearn.metrics.pairwise import cosine_similarity
cosine_sim = cosine_similarity(features, features)

filename = 'exercise-recommendation-model.pkl'
pickle.dump(cosine_sim, open(filename, 'wb'))

# index = pd.Series(df['Exercise'])

# def recommend_exercise(exercise):
#     exercises = []
#     idx = index[index == exercise].index[0]
#     print(idx)
#     score = pd.Series(cosine_sim[idx]).sort_values(ascending=False)
#     top5 = list(score.iloc[1:11].index)
#     print(top5)
    
#     for i in top5:
#         exercises.append(df['Exercise'][i])
#     return exercises

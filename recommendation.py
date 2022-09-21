import json
import pandas as pd
import numpy as np
import sqlite3 as dbsql
import pickle
import streamlit as st
from sklearn.preprocessing import LabelEncoder

db = dbsql.connect("Dataset/Database_2.db")
sim = pd.read_sql_query("SELECT * FROM tagsimilarity", db) 

data = pd.read_csv("Dataset/Binarized_Data_Count_10months.csv")
app_data = pd.read_csv("Dataset/AppDescription.csv")
app_data = app_data[['App Name', 'Description']]

le_app = LabelEncoder()
app_data['App Name'] = le_app.fit_transform(app_data['App Name'])

le_bot = LabelEncoder()
data['Bot Number'] = le_bot.fit_transform(data['Bot Number']) 

final_df = pd.read_csv("Dataset/Similarity_Matrix.csv")

#function for creating feature map for 2 most popular apps for a certain user
def OR_ops(app1,app2):
    
        store_app1=final_df[final_df["Display name of the application"]==app1].iloc[:,4:].values
        print(final_df)
        store_app2=final_df[final_df["Display name of the application"]==app2].iloc[:,4:].values
    
        feature_map = np.logical_or(store_app1,store_app2).astype(int)
        print(np.logical_or(store_app1,store_app2))
        return feature_map

#function for creating feature map for the most popular app for a user
def ops(app1):
    store_app1=final_df[final_df["Display name of the application"]==app1].iloc[:,4:].values
    return store_app1

#function for content recommendation
def content_filtering(feature_map):
    # app_list_sim = sim.App.to_list()
    # selected_app_index = app_list_sim.index(selected_app)
    # sim_scores = np.array(json.loads(sim.Description[selected_app_index]))

    # similarities = pd.DataFrame(app_list_sim, columns=['AppNames'])
    # similarities['Cont_Scores'] = list(sim_scores)
    # similarities = similarities.sort_values('Cont_Scores', ascending=False).reset_index(drop=True)
    # similarities.drop(0,inplace=True)
    # return similarities
    
    #new content filtering 
    
    final_df_array_genre = final_df.iloc[:,4:].values
    similarity_score = []
    for i in final_df_array_genre:
        sim = np.dot(feature_map, i)/(np.linalg.norm(feature_map)*np.linalg.norm(i))
        similarity_score.append(sim)

    df = pd.DataFrame(final_df.iloc[:,2].values, columns=['AppNames'])
    df['Cont_Scores'] = similarity_score

    return df.sort_values('Cont_Scores', ascending=False)



#function for collaborative recommendation
def collaborative_filtering(selected_user):
    
    f = open('Dataset/pnnewmatrix.txt', 'r')
    P = np.array(json.loads(f.read()))
    f.close()

    f = open('Dataset/qnnewmatrix.txt', 'r')
    Q = np.array(json.loads(f.read()))
    f.close()
    
    # Obtained loss
    #print("Loss while testing: ", loss(test, P, Q))
    estimates = np.dot(P[selected_user, :], Q.T)
    # print("P mat: ", P)
    # print("Q mat: ", Q)
    

    rec = pd.DataFrame(le_app.classes_.tolist(), columns=['AppNames'])
    rec['Coll_Scores'] = estimates.tolist()
    return rec.sort_values('Coll_Scores', ascending=False)

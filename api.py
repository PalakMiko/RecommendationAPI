#Specify the model version
model_version = 'v3'

#Libraries
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import sqlite3 as dbsql
from recommendation import content_filtering, collaborative_filtering, le_bot, le_app, ops, OR_ops
import gspread

#Title
st.title("Recommendation API")

# user_record = pd.read_csv('user_selection_record.csv')

sa = gspread.service_account('recommendation-api-363118-46846431ca85.json')
sh = sa.open('user_selection_record')
wks = sh.worksheet("Sheet1")


#Function to read datasets
def read_data():

    #count data
    count_data = pd.read_csv("Dataset/interaction_count_no_to_aug.csv")
    count_data = count_data[['Bot Number', 'App Name', 'Count']]

    #app data
    app_data = pd.read_csv("Dataset/AppDescription.csv")
    app_data = app_data[['App Name', 'Description']]

    return count_data, app_data


#Fetch datasets
count_data, app_data = read_data()
print("Data fectched!")

#Function for preprocessing
def data_processing(count_data, app_data, selected_bot):

    #Merge datasets
    left_join = pd.merge(app_data, count_data, on='App Name', how='right')
    left_join = left_join.dropna()

    # This now becomes count data
    left_join.reset_index(drop=True, inplace=True)
    count_data = left_join

    le_app = LabelEncoder()
    app_data['App Name'] = le_app.fit_transform(app_data['App Name'])

    count_data['App Name'] = le_app.transform(count_data['App Name'])

    #Converting type of app name which are currently their ids
    count_data['App Name'] = count_data['App Name'].astype(str)


    # #plotting
    # #individual bot data
    bot_data = count_data[count_data['Bot Number']==selected_bot]

    # #converting to plot
    bot_data.Count = bot_data.Count/sum(bot_data.Count.to_list())
    bot_data.Count = bot_data.Count*100

    temp_bot = bot_data.copy()
    temp_bot['App Name'] = le_app.inverse_transform(temp_bot['App Name'].astype('int32'))
    fig = px.histogram(temp_bot, x="App Name", y="Count",
                   hover_data=bot_data.columns)
    # selected_app = bot_data.sort_values('Count', ascending=False)
    selected_app=[]
    #from more than 75 quantile
    th1 = np.array(bot_data.Count.quantile([0.75]))[0]
    data1 = bot_data[bot_data['Count']>=th1]
    selected_app.append(data1.sort_values('Count', ascending=False)['Count'].to_list()[0])

    th2 = np.array(bot_data.Count.quantile([0.50]))[0]
    data2 = bot_data[bot_data['Count']>=th2]
    data2 = data2[data2['Count']<th1]
    selected_app.append(data1.sort_values('Count', ascending=False)['Count'].to_list()[0])
    
    # selected_app = selected_app[['App Name', 'Count']][selected_app['Count'] >=selected_app['Count'].to_list()[0]/2]
    # if len(selected_app)>=2:
    #     selected_app = selected_app.sort_values('Count', ascending=False)
    #     selected_app = selected_app['App Name'].to_list()[:2]
      
    st.plotly_chart(fig)
    return selected_app
    

#Button functions
def on_click_function(model_version, label, selected_user, recommendations, 
                      content_weight, collab_weight):
    wks.append_row([model_version, selected_user, label, 
                    recommendations,content_weight, collab_weight])

def user_feedback(response):
    user_feedback = response

           
####################################################################################################3
selected_user = st.selectbox('Select Bot ID', ['None']+ list(set(count_data['Bot Number'])))
collab_weight = st.text_input("Enter Collab Weight: ","0.5")
content_weight = st.text_input("Enter Content Weight: ", "0.5")

if selected_user!="None" and collab_weight and content_weight:
    selected_app = data_processing(count_data=count_data, app_data=app_data, selected_bot=selected_user)
  
    if len(selected_app)>1:
        feature_map = OR_ops(list(le_app.classes_)[int(selected_app[0])], list(le_app.classes_)[int(selected_app[1])])
        content_reco = content_filtering(feature_map)
    else:
        feature_map = ops(list(le_app.classes_)[int(selected_app.to_list()[0])])
        content_reco =  content_filtering(feature_map)
    
    selected_user = list(le_bot.classes_).index(selected_user)
    collab_reco = collaborative_filtering(selected_user)
        
    collab_reco.Coll_Scores = collab_reco.Coll_Scores/sum(collab_reco.Coll_Scores.to_list())
    content_reco.Cont_Scores = content_reco.Cont_Scores/sum(content_reco.Cont_Scores.to_list())
    

    st.write("Content Based Recommendation", 
            content_reco.sort_values('Cont_Scores', ascending =False)['AppNames'][:5].to_list())

    st.write("Collaborative Filtering Based Recommendation", 
            collab_reco.sort_values('Coll_Scores', ascending =False)['AppNames'][:5].to_list())
    

        
    #below LOCs are for hybrid recommendation
    hybrid_df = content_reco.merge(collab_reco, on='AppNames', how='left')
    hybrid_df = hybrid_df.fillna(0)

    num_recommendation = 5
    hybrid_df['Rankings'] = 0
    
    hybrid_df['Rankings'] = hybrid_df['Cont_Scores'] * float(content_weight) + hybrid_df['Coll_Scores'] * float(collab_weight)
    hybrid_df = hybrid_df.sort_values('Rankings',ascending=False)
    
    st.markdown('') 
    st.write("Top '{}' Hybrid recommendations of Apps: ".format(num_recommendation))
    hybrid_recomm = hybrid_df['AppNames'].to_list()
    
    user_rating = 1
    user_comment = 'No comments!'
    

    
    col1, col2, col3, col4, col5 = st.columns([1,1,1,1,1])
    cols = [col1, col2, col3, col4, col5]
    for i in range(5):
        with cols[i]:
            label = hybrid_recomm[i]
            st.button(label, on_click = on_click_function, 
                      args=[model_version, label, list(le_bot.classes_)[selected_user], str(hybrid_recomm),
                            content_weight, collab_weight])
    
        
    st.markdown("_________________________________________________________________")
    st.write("Rest Hybrid recommendations of Apps: ")
    hybrid_recomm = hybrid_df['AppNames'].to_list()
        
    col1, col2, col3, col4, col5 = st.columns([1,1,1,1,1])
    cols = [col1, col2, col3, col4, col5]
    c = 0
    for i in range(5,len(hybrid_recomm)):
        with cols[c]:
            label = hybrid_recomm[i]
            st.button(label, on_click = on_click_function, 
                      args=[model_version, label, list(le_bot.classes_)[selected_user], str(hybrid_recomm)])
        c+=1
        if c==5:
            c=0
        
    # st.write(wks.row_count)
    st.text('')
    st.markdown("_________________________________________________________________")

    row = wks.row_count
    col1, col2 = st.columns([1,1])
    with col1:
        user_rating = st.text_input("Rate the recommendations (1-5):", "1")  
    with col2:
        user_comment = st.text_area("Write your comment", "No comments!")

    if st.button("Submit"):
        wks.update(f'G{row}', int(user_rating))
        wks.update(f'H{row}', user_comment)
        wks.update('H1', 'user_comment')
        wks.update('G1', 'user_rating')

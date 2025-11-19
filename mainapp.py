#imports 
import streamlit as st
import pandas as pd
import pickle 
from sklearn.tree import DecisionTreeClassifier
from collections import Counter

# Set page configuration
st.set_page_config(layout="wide", page_title="COMP 560 Final App")

#Sidebar
st.sidebar.title("Credits")
st.sidebar.markdown("Group Members: Al Pagar, Kaw Bu, Jennifer Lee, Navya Katraju, Saaketh Akula")

st.sidebar.markdown(""" 
    ### Example Download of Data
    Data inputted into the application must have the same column names as the csv file downloadable below.
""")

@st.cache_data
def convert_for_download(df):
    return df.to_csv().encode("utf-8")

#example data is from diamond rank games
exampledf = pd.read_csv("data/example.csv")
csv = convert_for_download(exampledf)

st.sidebar.download_button(
    label="Download Example CSV",
    data=csv,
    file_name="data.csv",
    mime="text/csv",
    icon=":material/download:",
)




#Header and Title
st.title("COMP 560 Final Application")
"""Welcome to our final application for COMP 560! This app demonstrates the skills and knowledge we've acquired throughout the course.
Our main feature is an application where a user can upload their league of legends stats and a machine learning model will attempt to predict your rank!
"""

"________________________________"
st.header("Uploading Game Stats")


my_upload = st.file_uploader("Upload Your League Game Stats Here!", type=['csv','xlsx'],accept_multiple_files=False,key="fileUploader")
statsdf = None
if my_upload is not None: 
    try:
        statsdf = pd.read_csv(my_upload)
        statsdf = statsdf.drop(columns=["Unnamed: 0"], axis=1)
    except:
        try:
            statsdf = pd.read_excel(my_upload)
            statsdf = statsdf.drop(columns=["Unnamed: 0"], axis=1)
        except:
            pass

if statsdf is not None: 
    st.dataframe(data=statsdf)

vanilla_decision_tree = pickle.load(open("models/vanilla_tree.sav", "rb"))
ranks = ["Unranked","Iron", "Bronze", "Silver", "Gold", "Platinum", "Emerald", "Diamond", "Master", "Grandmaster", "Challenger"]


if st.button("Predict your rank!", width="stretch") and statsdf is not None:
    predictions = list(vanilla_decision_tree.predict(statsdf))
    predictions = [ranks[x] for x in predictions]
    counted_predictions = Counter(predictions)
    
    #get max 
    keys = list(counted_predictions.keys())
    maxkey = keys[0]
    for key in keys:
        if counted_predictions[maxkey] < counted_predictions[key]:
            maxkey = key
    
    st.header("Rank Prediction!")
    gamepredictiondf = pd.DataFrame()
    gamepredictiondf["Game Number"] = range(0, len(predictions))
    gamepredictiondf["Rank Predictions"] = predictions

    st.dataframe(data=gamepredictiondf)
    st.markdown(f"### Overall Rank Prediction: {maxkey}")
    
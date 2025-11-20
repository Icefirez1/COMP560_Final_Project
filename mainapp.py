#imports 
import streamlit as st
import pandas as pd
import pickle 
from sklearn.tree import DecisionTreeClassifier
from collections import Counter
from league_api import get_match_prediction_summary

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
One feature is an application where a user can upload their league of legends stats and a machine learning model will attempt to predict your rank!
Another feature is that you can input a match ID and we'll predict the ranks of each player! 
"""

"________________________________"
st.header("Predicting Ranks Based on Uploaded Game Stats!")


my_upload = st.file_uploader("Upload Your League Game Stats Here", type=['csv','xlsx'],accept_multiple_files=False,key="fileUploader")
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


if st.button("Predict your rank", width="stretch") and statsdf is not None:
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
    st.metric("Average Predicted Rank", maxkey)

"________________________________"
st.header("Predict Player Ranks from Match ID")
st.markdown("Provide a Riot match ID (ex: `NA1_5416214402`) to pull the match data and run predictions for every player.")

match_id_input = st.text_input("Match ID", value="", placeholder="Region_GameId (ex: NA1_5416214402)", key="matchIdInput")
match_predict_btn = st.button("Fetch Match Predictions", use_container_width=True, key="matchPredictBtn")

if match_predict_btn:
    match_id_input = match_id_input.strip()
    if not match_id_input:
        st.warning("Please enter a valid match ID before requesting predictions.")
    else:
        with st.spinner("Fetching match data and generating predictions..."):
            summary = get_match_prediction_summary(match_id_input)
        if summary is None:
            st.error("Unable to retrieve match data or predictions. Double-check the match ID and ensure the API key is configured.")
        else:
            st.subheader("Blue Team (Loss)")
            st.dataframe(summary['blue_team'])
            st.subheader("Red Team (Win)")
            st.dataframe(summary['red_team'])

            st.subheader("Rank Distribution")
            if summary['rank_counts']:
                dist_df = pd.DataFrame(
                    [(rank, count) for rank, count in summary['rank_counts'].items()],
                    columns=["Rank", "Players"]
                )
                st.dataframe(dist_df)
            else:
                st.write("No rank predictions available.")

#imports 
import streamlit as st

# Set page configuration
st.set_page_config(layout="wide", page_title="COMP 560 Final App")

#Sidebar
st.sidebar.title("Credits")
st.sidebar.markdown("Group Members: Al Pagar, Kaw Bu, ")


#Header and Title
st.title("COMP 560 Final Application")
"""Welcome to our final application for COMP 560! This app demonstrates the skills and knowledge we've acquired throughout the course.
Our main feature is a dashboard for users to utilize 
"""

"________________________________"
st.header("Image Upload (If we do logo uploader)")
my_upload = st.file_uploader("Upload an image of a logo!", type=["png", "jpg", "jpeg"])
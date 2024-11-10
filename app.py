import streamlit as st

# Title and description
st.title("Hello Streamlit!")
st.write("This is a basic Streamlit app.")

# Input form
name = st.text_input("Enter your name:")
age = st.number_input("Enter your age:", min_value=0, max_value=120, step=1)

# Display greeting message when input is provided
if name and age:
    st.write(f"Hello, {name}! You are {age} years old.")
else:
    st.write("Please enter your name and age above.")

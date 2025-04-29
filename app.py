import streamlit as st

import streamlit.components.v1 as components

# Wide layout 
st.set_page_config(layout="wide")

# Set page title
st.title("Apple Stock Backtest Results")

# Read and display the HTML file
def display_html():
    with open('apple_backtest_results.html', 'r', encoding='utf-8') as f:
        html_content = f.read()
    components.html(html_content, height=1200, scrolling=True)


# Main app
try:
    display_html()
except FileNotFoundError:
    st.error("Error: apple_backtest_results.html file not found. Please make sure the file exists in the same directory.")
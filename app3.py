import os
import openai
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tenacity import retry, wait_exponential, stop_after_attempt, RetryError

# Set up proxy if required
# os.environ['HTTP_PROXY'] = 'http://your_proxy_address:port'
# os.environ['HTTPS_PROXY'] = 'http://your_proxy_address:port'

# Set up the OpenAI API key
openai.api_key = 'sk-proj-r8u8HC64Ev91WJHi5xIHT3BlbkFJH7cKgGj1ELmSseJh4MiS'

@retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(5), reraise=True)
def get_formula_suggestions(dataframe):
    prompt = f"Given the following data:\n{dataframe.head().to_string()}\nSuggest useful Excel formulas for analysis and processing:"
    
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.5,
    )
    return response.choices[0].text.strip()

# Streamlit app
st.title("Formulabot")
st.write("Upload an Excel file to get useful Excel formula suggestions and visualize the data:")

# File uploader
uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")

if uploaded_file:
    # Read the Excel file
    df = pd.read_excel(uploaded_file)
    st.write("### Data Preview:")
    st.write(df)

    if st.button("Get Formula Suggestions"):
        with st.spinner('Generating suggestions...'):
            try:
                suggestions = get_formula_suggestions(df)
                st.write("### Formula Suggestions:")
                st.write(suggestions)
            except RetryError as re:
                st.write(f"Retries failed: {re}")
            except Exception as e:
                st.write(f"An error occurred: {e}")
    
    if st.button("Visualize"):
        st.write("### Data Visualization")
        
        # Select only numeric columns for correlation heatmap
        numeric_df = df.select_dtypes(include=['float64', 'int64'])
        
        if numeric_df.empty:
            st.write("No numeric data available for visualization.")
        else:
            # Basic visualization: Correlation heatmap
            st.write("#### Correlation Heatmap")
            fig, ax = plt.subplots()
            sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
            st.pyplot(fig)

            # Optional: Additional plots based on data types
            for col in numeric_df.columns:
                st.write(f"#### Distribution of {col}")
                fig, ax = plt.subplots()
                sns.histplot(numeric_df[col], kde=True, ax=ax)
                st.pyplot(fig)
else:
    st.write("Please upload an Excel file.")

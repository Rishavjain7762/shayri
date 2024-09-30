import streamlit as st
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os

# Set up Groq API key
GROQ_API_KEY = "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"

# Initialize Groq LLM
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama3-70b-8192")

# Create a prompt template
prompt_template = PromptTemplate(
    input_variables=["words"],
    template="Create a hindi shayri using the following words: {words}. The shayri should be creative, heartwarming and coherent."
)

# Create an LLMChain
poem_chain = LLMChain(llm=llm, prompt=prompt_template)

# Streamlit app
st.title("🎭 Shayri Creator")
st.write("Enter comma-separated words, and I'll create a shayri for you!")

# User input
user_input = st.text_input("Enter words (comma-separated):", "love, nature, hope")

if st.button("Generate Shayri"):
    if user_input:
        with st.spinner("Creating your poem..."):
            # Generate the poem
            result = poem_chain.run(user_input)
            st.subheader("Your Generated Poem:")
            st.write(result)
    else:
        st.warning("Please enter some words to create a poem.")

st.markdown("---")
st.write("Created with ❤️ using Langchain, Groq, and Streamlit")
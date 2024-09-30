import streamlit as st
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from gtts import gTTS
import os

# Set up Groq API key
GROQ_API_KEY = "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"

# Initialize Groq LLM
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama3-70b-8192")

# Create a prompt template
prompt_template = PromptTemplate(
    input_variables=["words"],
    template="Create a hindi shayri using the following words: {words}. The shayri should be creative, heartwarming, and coherent."
)

# Create an LLMChain
poem_chain = LLMChain(llm=llm, prompt=prompt_template)

# Streamlit app
st.title("🎭 Shayri Creator")
st.write("Enter comma-separated words, and I'll create a shayri for you!")

# Gender selection
gender = st.radio("Please select your gender:", ('Male', 'Female'))

# Mood selection
mood = st.selectbox("Select your mood:", ['Happy', 'Sad', 'Funny'])

# Default statement for recording
st.write("Please record the following statement to analyze your voice:")
default_statement = "Main apni aawaaz ko record kar raha hoon / rahi hoon."
st.text(default_statement)

# User uploads a voice sample (for the default statement)
uploaded_file = st.file_uploader("Record your voice saying the above statement (Hindi and English)", type=["wav", "mp3"])

# Function to convert text to speech using gTTS
def text_to_speech(text, lang='hi', mood='Happy'):
    # Adjust voice settings based on mood
    if mood == 'Sad':
        tts = gTTS(text=text, lang=lang, slow=True)
    elif mood == 'Funny':
        tts = gTTS(text=text, lang=lang, slow=False)
    else:  # Default 'Happy' or neutral mood
        tts = gTTS(text=text, lang=lang, slow=False)
    
    tts.save("shayari_audio.mp3")
    return "shayari_audio.mp3"

# User input for shayari generation
user_input = st.text_input("Enter words (comma-separated):", "love, nature, hope")

if st.button("Generate Shayri"):
    if user_input:
        with st.spinner("Creating your poem..."):
            # Generate the poem
            result = poem_chain.run(user_input)
            st.subheader("Your Generated Poem:")
            st.write(result)

            # Convert the generated shayari to speech based on mood
            audio_file = text_to_speech(result, mood=mood)
            
            # Play the generated audio in the app
            st.audio(audio_file, format="audio/mp3")
            
            # Option to download the audio file
            with open(audio_file, "rb") as file:
                st.download_button(label="Download Shayari Audio", data=file, file_name="shayari.mp3", mime="audio/mp3")
    else:
        st.warning("Please enter some words to create a poem.")

st.markdown("---")
st.write("Created with ❤️ using Langchain, Groq, and Streamlit")

# Further processing of the uploaded voice sample
if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    # In a real-world scenario, you could analyze this voice file and create a custom TTS model with it.
    

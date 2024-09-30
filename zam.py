import streamlit as st
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from gtts import gTTS
import os
import re

# Set up Groq API key
GROQ_API_KEY = "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"

# Initialize Groq LLM
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama3-70b-8192")

# Create a prompt template
prompt_template = PromptTemplate(
    input_variables=["words", "mood"],
    template="Create a hindi shayri using the following words: {words}. The shayri should be creative, heartwarming, and coherent, reflecting a {mood} mood."
)

# Create an LLMChain
poem_chain = LLMChain(llm=llm, prompt=prompt_template)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
    }
    .shayri-output {
        background-color: white;
        padding: 20px;
        border-radius: 5px;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Streamlit app
st.title("🎭 Shayri Creator")
st.write("Enter comma-separated words, and I'll create a shayri for you!")

# Gender and mood selection
gender = st.radio("Please select your gender:", ('Male', 'Female'))
mood = st.selectbox("Select your mood:", ['Happy', 'Sad', 'Romantic', 'Funny'])

# Default statement for recording
st.write("Please record the following statement to analyze your voice:")
default_statement = "Main apni aawaaz ko record kar raha hoon / rahi hoon."
st.text(default_statement)

# User uploads a voice sample
uploaded_file = st.file_uploader("Record your voice saying the above statement (Hindi and English)", type=["wav", "mp3"])

# Function to convert text to speech using gTTS
def text_to_speech(text, lang='hi', mood='Happy'):
    try:
        # Adjust voice settings based on mood
        if mood == 'Sad':
            tts = gTTS(text=text, lang=lang, slow=True)
        elif mood == 'Funny':
            tts = gTTS(text=text, lang=lang, slow=False)
        else:  # Default 'Happy' or 'Romantic' mood
            tts = gTTS(text=text, lang=lang, slow=False)
        
        tts.save("shayari_audio.mp3")
        return "shayari_audio.mp3"
    except Exception as e:
        st.error(f"An error occurred during text-to-speech conversion: {str(e)}")
        return None

# User input for shayari generation
user_input = st.text_input("Enter words (comma-separated):", "pyaar, khushi, zindagi")

# Input validation
if not re.match(r'^[\w\s,]+$', user_input):
    st.warning("Please enter valid, comma-separated words.")

# Handling uploaded audio file
if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    st.success("Uploaded voice sample successfully!")
    st.info("Your uploaded voice will be used for TTS in future development (requires integration with a voice cloning service).")

if st.button("Generate Shayri"):
    if user_input and re.match(r'^[\w\s,]+$', user_input):
        try:
            with st.spinner("Creating your shayri..."):
                # Generate the shayri
                result = poem_chain.run(words=user_input, mood=mood)
                st.markdown("<div class='shayri-output'>", unsafe_allow_html=True)
                st.subheader("Your Generated Shayri:")
                st.write(result)
                st.markdown("</div>", unsafe_allow_html=True)

                # Convert the generated shayari to speech based on mood
                audio_file = text_to_speech(result, mood=mood)
                
                if audio_file:
                    # Play the generated audio in the app
                    st.audio(audio_file, format="audio/mp3")
                    
                    # Option to download the audio file
                    with open(audio_file, "rb") as file:
                        st.download_button(label="Download Shayari Audio", data=file, file_name="shayari.mp3", mime="audio/mp3")
                
                # Save to history (you could expand this to use a database)
                if 'shayri_history' not in st.session_state:
                    st.session_state.shayri_history = []
                st.session_state.shayri_history.append((user_input, mood, result))
        except Exception as e:
            st.error(f"An error occurred: {str(e)}. Please try again later.")
    else:
        st.warning("Please enter valid, comma-separated words to create a shayri.")

# Display shayri history
if 'shayri_history' in st.session_state and st.session_state.shayri_history:
    st.markdown("### Your Shayri History")
    for i, (words, mood, shayri) in enumerate(reversed(st.session_state.shayri_history[-5:])):
        with st.expander(f"Shayri {len(st.session_state.shayri_history) - i}"):
            st.write(f"Words: {words}")
            st.write(f"Mood: {mood}")
            st.write(shayri)

st.markdown("---")
st.write("Created with ❤️ using Langchain, Groq, and Streamlit")
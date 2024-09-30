import streamlit as st
from transformers import AutoProcessor, BarkModel
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import scipy
import torch
import os

# Set up Groq API key
GROQ_API_KEY = "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"

# Initialize Groq LLM
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama3-70b-8192")

# Create a prompt template for Shayari generation
prompt_template = PromptTemplate(
    input_variables=["words"],
    template="Create a hindi shayri using the following words: {words}. The shayri should be creative, heartwarming, and coherent."
)

# Create an LLMChain for generating Shayari
poem_chain = LLMChain(llm=llm, prompt=prompt_template)

# Load the Bark processor and model
processor = AutoProcessor.from_pretrained("suno/bark")
model = BarkModel.from_pretrained("suno/bark")
model.to("cpu")  # Use CPU as M3 Pro doesn't support CUDA

# Function to generate audio using Bark Model
def generate_audio(text, preset, output):
    inputs = processor(text, voice_preset=preset, return_tensors="pt")

    # Get the input IDs and pad manually (if necessary)
    input_ids = inputs['input_ids']
    attention_mask = torch.ones_like(input_ids)  # Default mask (all 1s)

    # Ensure inputs are on CPU
    input_ids = input_ids.to("cpu")
    attention_mask = attention_mask.to("cpu")

    # Generate audio using the Bark model
    audio_array = model.generate(input_ids=input_ids, attention_mask=attention_mask)

    # Convert the output tensor to a NumPy array
    audio_array = audio_array.cpu().numpy().squeeze()

    # Get the sample rate and write the audio file
    sample_rate = model.generation_config.sample_rate
    scipy.io.wavfile.write(output, rate=sample_rate, data=audio_array)

# Streamlit app
st.title("🎭 Shayri Creator with Custom Voice")

# Gender and mood selection
gender = st.radio("Please select your gender:", ('Male', 'Female'))
mood = st.selectbox("Select your mood:", ['Happy', 'Sad', 'Funny'])

# Default statement for recording
st.write("Please record the following statement to analyze your voice:")
default_statement = "Main apni aawaaz ko record kar raha hoon / rahi hoon."
st.text(default_statement)

# User uploads a voice sample (for the default statement)
uploaded_file = st.file_uploader("Record your voice saying the above statement (Hindi and English)", type=["wav", "mp3"])

# User input for shayari generation
user_input = st.text_input("Enter words (comma-separated):", "love, nature, hope")

# Handling uploaded audio file
if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    # Save the uploaded file locally
    with open("user_voice.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("Uploaded voice sample successfully!")

if st.button("Generate Shayri"):
    if user_input and uploaded_file is not None:
        with st.spinner("Creating your shayari with your voice..."):
            # Generate the shayari
            result = poem_chain.run(user_input)
            st.subheader("Your Generated Shayari:")
            st.write(result)

            # Generate audio using Bark Model with the user's voice
            output_audio_path = "shayari_with_voice.wav"
            generate_audio(
                text=result,
                preset="v2/hi_speaker_6",  # Use a Hindi speaker preset
                output=output_audio_path
            )

            # Play the generated audio in the app
            st.audio(output_audio_path, format="audio/wav")

            # Option to download the generated audio file
            with open(output_audio_path, "rb") as file:
                st.download_button(label="Download Shayari with Your Voice", data=file, file_name="shayari_with_your_voice.wav", mime="audio/wav")
    else:
        st.warning("Please enter some words and upload a WAV file to create a shayari.")

st.markdown("---")
st.write("Created with ❤️ using Langchain, Groq, Bark, and Streamlit")

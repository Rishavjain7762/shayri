from transformers import AutoProcessor, BarkModel
import scipy

# Load the processor and model
processor = AutoProcessor.from_pretrained("suno/bark")
model = BarkModel.from_pretrained("suno/bark")
model.to("cpu")  # Use CPU for M3 Pro

def generate_audio(text, preset, output):
    # Process the input text without padding
    inputs = processor(text, voice_preset=preset)
    
    # Move inputs to the CPU
    for k, v in inputs.items():
        inputs[k] = v.to("cpu")
    
    # Generate audio using the Bark model
    audio_array = model.generate(**inputs)
    
    # Convert the output tensor to a NumPy array
    audio_array = audio_array.cpu().numpy().squeeze()
    
    # Get the sample rate and write the audio file
    sample_rate = model.generation_config.sample_rate
    scipy.io.wavfile.write(output, rate=sample_rate, data=audio_array)

# Call the function to generate audio with the specified text and speaker preset
generate_audio(
    text="Hey there lola, You are just awesome",
    preset="v2/en_speaker_9",  # Choose a speaker preset
    output="music.wav"  # Output file name
)

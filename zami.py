from transformers import AutoProcessor, BarkModel
import scipy

# Load the processor and model
processor = AutoProcessor.from_pretrained("suno/bark")
model = BarkModel.from_pretrained("suno/bark")
model.to("cpu")  # Set model to use CPU (M3 Pro doesn't support CUDA)

def generate_audio(text, preset, output):
    # Process the input text
    inputs = processor(text, voice_preset=preset, return_tensors="pt")

    # Create attention mask manually if not already present
    if "attention_mask" not in inputs:
        # Attention mask is 1 for non-padding tokens, 0 for padding tokens
        inputs["attention_mask"] = (inputs["input_ids"] != processor.tokenizer.pad_token_id).long()
    
    # Ensure inputs are on CPU
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
    text=" Goli Beta Masti nai, Jethalal naam hai mera. Babita Ji mai hi pataunga",
    preset="v2/hi_speaker_5",  # Choose a speaker preset
    output="output5.wav"  # Output file name
)

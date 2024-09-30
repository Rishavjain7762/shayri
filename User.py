from transformers import AutoProcessor, BarkModel
import scipy
import torch

# Load the processor and model
processor = AutoProcessor.from_pretrained("suno/bark")
model = BarkModel.from_pretrained("suno/bark")
model.to("cpu")  # Set model to use CPU (since M3 Pro doesn't support CUDA)


def generate_audio(text, preset, output):
    # Process the input text without padding
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


# Call the function to generate audio with the specified text and speaker preset
generate_audio(
    text="♪ Goli Beta Masti nai, Jethalal naam hai mera. Babita Ji mai hi pataunga ♪",
    preset= "v2/hi_speaker_9",  # Choose a speaker preset
    output="outputio.wav"  # Output file name
)

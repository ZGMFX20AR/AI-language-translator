import torch
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wavfile
import whisper
import time
from datetime import timedelta
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Constants
DURATION = 30  # seconds
SAMPLE_RATE = 16000  # 16 kHz
FILENAME = "recording.wav"

# Initialize Whisper model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
whisper_model = whisper.load_model("base").to(device)

# Initialize M2M100 model and tokenizer for translation
tokenizer = AutoTokenizer.from_pretrained("facebook/m2m100_418M")
translation_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/m2m100_418M").to(device)

def countdown(seconds):
    print("Get ready to record...")
    for i in range(seconds, 0, -1):
        print(i, "...")
        time.sleep(1)
    print("Recording started!")

def display_duration(duration):
    start_time = time.time()
    while time.time() - start_time < duration:
        elapsed = int(time.time() - start_time)
        print(f"Recording... {elapsed}/{duration} seconds", end="\r")  # Overwrite the same line
        time.sleep(1)
    print("\nRecording finished!")  # Move to the next line after recording

def record_audio(filename, duration, sample_rate):
    countdown(3)  # Optional countdown
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    
    # Display live recording duration
    display_duration(duration)

    sd.wait()  # Wait until recording is complete
    wavfile.write(filename, sample_rate, audio)
    print(f"Recording saved as {filename}")

def translate_to_malay(text):
    """Translate English text to Malay using M2M100 model."""
    tokenizer.src_lang = "en"  # English as source language
    encoded_text = tokenizer(text, return_tensors="pt").to(device)

    # Generate translation to Malay
    generated_tokens = translation_model.generate(
        **encoded_text,
        forced_bos_token_id=tokenizer.get_lang_id("ms")  # Malay target
    )

    # Decode the translated text
    translation = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    return translation

def save_transcription_with_timestamps(transcription, output_file="transcription.txt"):
    with open(output_file, "w") as f:
        for segment in transcription["segments"]:
            start = str(timedelta(seconds=segment["start"]))
            end = str(timedelta(seconds=segment["end"]))
            text = segment["text"]
            f.write(f"[{start} --> {end}] {text}\n")

def main():
    # Record audio with live duration display
    record_audio(FILENAME, DURATION, SAMPLE_RATE)

    # Transcribe audio
    print("Transcribing...")
    result = whisper_model.transcribe(FILENAME)

    # Display transcription in console
    english_text = result["text"]
    print(f"Full Transcription: {english_text}")

    # Save transcription with timestamps
    print("Saving transcription with timestamps...")
    save_transcription_with_timestamps(result)
    print("Transcription saved as 'transcription.txt'.")

    # Translate the English text to Malay
    malay_translation = translate_to_malay(english_text)
    print("Translation (Malay):", malay_translation)

    # Save the translation to a text file
    with open("translation.txt", "w") as f:
        f.write(malay_translation)
    print("Translation saved as 'translation.txt'.")

if __name__ == "__main__":
    main()

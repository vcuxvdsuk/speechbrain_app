import streamlit as st
from audiorecorder import audiorecorder
from speechbrain.inference import EncoderDecoderASR
import ollama

st.title("Speechbrain App")

def generate_story_from(question):
    response = ollama.chat(model='llama3', messages=[
        {
            'role': 'system',
            'content': 'you are the best speaker and you can have a fluid conversation with any one'
        },
        {
            'role': 'user',
            'content': question
        }
    ])
    return response['message']['content']


def convert_speech_to_text():
    try:
        asr_model = EncoderDecoderASR.from_hparams(
            source="speechbrain/asr-conformer-transformerlm-librispeech",
            savedir="pretrained_models/asr-transformer-transformerlm-librispeech")
        text = asr_model.transcribe_file("audio.wav")
        return text
    except Exception as e:
        st.error(f"Error in transcription: {e}")
        return ""


st.write("Welcome to my Speechbrain App")

audio = audiorecorder("record")

if audio:  # Check if audio is recorded
    st.audio(audio.export().read(), autoplay=True)
    audio.export("audio.wav", format="wav")
    transcript = convert_speech_to_text()
    if transcript:  # Only generate the story if the transcription succeeded
        st.markdown(transcript)
        story = generate_story_from(transcript)
        st.markdown(story)
    else:
        st.write("No transcription available.")
else:
    st.write("No audio recorded.")

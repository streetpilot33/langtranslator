import streamlit as st
import speech_recognition as sr
import gtts
import pytesseract
import pdfplumber
import numpy as np
import os
import requests
from bs4 import BeautifulSoup
from io import BytesIO
from PIL import Image
from deep_translator import GoogleTranslator
import ollama  # ✅ Integrated Ollama AI
import docx
from gtts.lang import tts_langs
from textblob import TextBlob  # ✅ Sentiment Analysis & Grammar Correction
from transformers import pipeline  # ✅ Summarization & Emotion Detection

# ✅ Set Streamlit page config
st.set_page_config(page_title="AI Language Translator", layout="wide")

# ✅ Set Tesseract Path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
os.environ["TESSDATA_PREFIX"] = r"C:\Program Files\Tesseract-OCR\tessdata"

st.title("🌍 AI-Powered Multi-Function Language Translator")

# ✅ Initialize session state
if "user_input" not in st.session_state:
    st.session_state.user_input = ""
if "translated_text" not in st.session_state:
    st.session_state.translated_text = ""

# ✅ Fetch supported languages dynamically
try:
    google_supported_langs = GoogleTranslator().get_supported_languages()
    supported_languages = {lang.capitalize(): lang for lang in google_supported_langs}
except Exception as e:
    st.error(f"⚠ Error fetching supported languages: {e}")
    supported_languages = {}

# ✅ Get supported languages for gTTS
gtts_supported_langs = tts_langs()

# ✅ Sidebar: Input Mode Selection
mode = st.sidebar.selectbox("Select Input Mode:", ["Text", "Speech", "Image", "File", "URL", "Camera"])

# ✅ Language Selection
src_lang = st.sidebar.selectbox("Source Language:", list(supported_languages.keys()))
tgt_lang = st.sidebar.selectbox("Target Language:", list(supported_languages.keys()))

# ✅ Convert Language Name to Code
src_code = supported_languages.get(src_lang, "auto")
tgt_code = supported_languages.get(tgt_lang, "en").lower()

# ✅ Check if tgt_code is supported by gTTS, otherwise fallback to English
tts_fallback_code = tgt_code if tgt_code in gtts_supported_langs else "en"


def translate_text(text, use_ollama=False):
    """Translate text using Google Translator or Ollama AI"""
    if use_ollama:
        try:
            response = ollama.chat(
                model="mistral",
                messages=[{"role": "user", "content": f"Translate this from {src_lang} to {tgt_lang}: {text}"}]
            )
            return response["message"]["content"]
        except Exception as e:
            return f"❌ Error: {str(e)}"
    else:
        return GoogleTranslator(source=src_code, target=tgt_code).translate(text)


def text_to_speech(text):
    """Convert text to speech (Fallback if language is not supported)"""
    try:
        tts = gtts.gTTS(text, lang=tts_fallback_code)
        audio_file = BytesIO()
        tts.write_to_fp(audio_file)
        audio_file.seek(0)
        return audio_file
    except Exception as e:
        st.error(f"❌ TTS Error: {e}")
        return None


def speech_to_text():
    """Convert speech to text"""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎤 Speak now...")
        try:
            audio = recognizer.listen(source, timeout=5)
            text = recognizer.recognize_google(audio, language=src_code)
            st.success(f"🗣 Recognized Speech: {text}")
            return text
        except sr.UnknownValueError:
            st.error("❌ Could not understand the speech.")
        except sr.RequestError:
            st.error("❌ Speech Recognition API unavailable.")
    return ""


def extract_text_from_image(image):
    """Extract text from image using OCR"""
    image = np.array(Image.open(image))
    try:
        return pytesseract.image_to_string(image, lang=src_code)
    except pytesseract.TesseractError:
        return pytesseract.image_to_string(image, lang="eng")


def extract_text_from_pdf(pdf_file):
    """Extract text from PDF"""
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text


def extract_text_from_txt(txt_file):
    """Extract text from TXT file"""
    return txt_file.read().decode("utf-8")


def extract_text_from_docx(docx_file):
    """Extract text from DOCX file"""
    doc = docx.Document(docx_file)
    return "\n".join([para.text for para in doc.paragraphs])


def extract_text_from_url(url):
    """Extract text from a webpage"""
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        return soup.get_text()
    except:
        return "Error fetching webpage text."


def correct_grammar(text):
    """Correct grammar using TextBlob"""
    return str(TextBlob(text).correct())


def analyze_sentiment(text):
    """Analyze sentiment (Positive/Negative/Neutral)"""
    sentiment = TextBlob(text).sentiment.polarity
    if sentiment > 0:
        return "😊 Positive"
    elif sentiment < 0:
        return "😡 Negative"
    else:
        return "😐 Neutral"


def detect_emotion(text):
    """Detect emotion using Hugging Face's transformers pipeline"""
    emotion_pipeline = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
    emotions = emotion_pipeline(text)
    return emotions[0]["label"]


def summarize_text(text):
    """Summarize text using Hugging Face's transformers pipeline"""
    summarizer = pipeline("summarization")
    summary = summarizer(text, max_length=100, min_length=30, do_sample=False)
    return summary[0]['summary_text']


# ✅ Input Handling
if mode == "Text":
    st.session_state.user_input = st.text_area("Enter text to translate:")

elif mode == "Speech":
    if st.button("Start Recording"):
        st.session_state.user_input = speech_to_text()

elif mode == "Image":
    uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if uploaded_image:
        st.session_state.user_input = extract_text_from_image(uploaded_image)

elif mode == "File":
    uploaded_file = st.file_uploader("Upload a file", type=["pdf", "docx", "txt"])
    if uploaded_file:
        if uploaded_file.name.endswith(".pdf"):
            st.session_state.user_input = extract_text_from_pdf(uploaded_file)
        elif uploaded_file.name.endswith(".docx"):
            st.session_state.user_input = extract_text_from_docx(uploaded_file)
        elif uploaded_file.name.endswith(".txt"):
            st.session_state.user_input = extract_text_from_txt(uploaded_file)

elif mode == "URL":
    url = st.text_input("Enter webpage URL:")
    if st.button("Extract Text"):
        st.session_state.user_input = extract_text_from_url(url)

elif mode == "Camera":
    st.info("📸 Capture an image to extract text and translate it.")
    captured_image = st.camera_input("Take a photo")
    if captured_image:
        st.image(captured_image, caption="📷 Captured Image", use_column_width=True)
        st.session_state.user_input = extract_text_from_image(captured_image)

# ✅ Translation
if st.session_state.user_input:
    st.write("📝 *Extracted Text:*", st.session_state.user_input)
    use_ollama = st.sidebar.checkbox("Use AI-Powered Translation?", key="use_ollama")
    st.session_state.translated_text = translate_text(st.session_state.user_input, use_ollama)
    st.success(st.session_state.translated_text)

    st.write("🔍 *Sentiment Analysis:*", analyze_sentiment(st.session_state.user_input))
    st.write("🎭 *Emotion Detected:*", detect_emotion(st.session_state.user_input))
    st.write("📝 *Grammar Corrected Text:*", correct_grammar(st.session_state.user_input))
    st.write("📌 *Summarized Text:*", summarize_text(st.session_state.user_input))

    audio_output = text_to_speech(st.session_state.translated_text)
    if audio_output:
        st.audio(audio_output)



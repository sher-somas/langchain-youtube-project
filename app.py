import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import YoutubeLoader
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
import warnings

warnings.filterwarnings('ignore')

load_dotenv()
parser = StrOutputParser()
model = ChatOpenAI(model="gpt-4")

# Function to get transcription from YouTube video
def get_transcription(youtube_url):
    loader = YoutubeLoader.from_youtube_url(youtube_url)
    documents = loader.load()
    transcript = "\n".join([doc.page_content for doc in documents])
    return transcript

# Function to summarize the transcript using LLMChain
def summarize_with_langchain(transcript):
    
    summarize_template = """Summarize the following text:\n\n {transcript}"""

    summarization_prompt = ChatPromptTemplate.from_template(summarize_template)

    summarization_prompt.format(transcript=transcript)

    summarize_chain = summarization_prompt | model | parser

    summary = summarize_chain.invoke({"transcript": transcript})
    
    return summary

# Function to translate the transcript using LLMChain
def translate_with_langchain(transcript, target_language):
    
    translate_template = """Translate {transcript} to {target_language}"""

    translation_prompt = ChatPromptTemplate.from_template(translate_template)

    translation_prompt.format(transcript=transcript, target_language=target_language)

    translate_chain = translation_prompt | model | parser

    translation = translate_chain.invoke({"transcript": transcript, "target_language": target_language})
    

    return translation

# Mapping for language codes
language_codes = {
    'German': 'German',
    'Spanish': 'Spanish',
    'Hindi': 'Hindi'
}

# Streamlit app layout
st.title("YouTube Video Transcription, Translation & Summarization")
st.image("project_logo.jpeg")

# Input for YouTube URL, Language and Summarization in one go
with st.form("user_input"):
    youtube_url = st.text_input("Enter YouTube URL")
    language = st.selectbox("Select Language for Translation", options=['German', 'Spanish', 'Hindi'])
    summarize = st.checkbox("Summarize the transcript")
    submitted = st.form_submit_button("Submit")

if submitted and youtube_url:
    try:
        # Fetch transcription
        transcript = get_transcription(youtube_url)
        st.text_area("Transcription", value=transcript, height=300)

        # Summarization option
        if summarize:
            st.write("Summarizing transcript...")
            summary = summarize_with_langchain(transcript)
            st.text_area("Summary", value=summary, height=150)
            transcript = summary  # Replace transcript with summary for translation

        # Translate the transcript or summary
        st.write(f"Translating to {language}...")
        target_language = language_codes[language]
        translated_transcript = translate_with_langchain(transcript, target_language)
        st.text_area(f"Translated Transcript ({language})", value=translated_transcript, height=300)
    
    except Exception as e:
        st.error(f"Error occurred: {str(e)}")

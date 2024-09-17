import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import YoutubeLoader
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
import streamlit_scrollable_textbox as stx

from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(temperature=0.2,
                 model="gpt-4")

parser = StrOutputParser()

def get_transcription(youtube_url):
    loader = YoutubeLoader.from_youtube_url(youtube_url)
    documents = loader.load()
    transcript = "\n".join([doc.page_content for doc in documents])
    return transcript

translation_prompt = ChatPromptTemplate.from_template(
    "Translate {answer} to {language}"
)

chain = translation_prompt | llm | parser

# translation_chain = (
#     {"answer": chain, "language": itemgetter("language")} | translation_prompt | llm | parser
# )



page_config = {"page_title": "Transcribe and translate youtube videos",
                   "page_icon": "project_logo.jpeg",
                   'layout': "centered",
                   'initial_sidebar_state': 'auto'}

st.set_page_config(**page_config)
st.image('project_logo.jpeg')

menu = ["Home", "Architecture and details", "About"]
choice = st.sidebar.selectbox("Menu", menu)


llm_string = st.text_input("youtube url")

button_clicked = st.button("transcribe")


option = st.selectbox(
    "Translate to",
    ("German", "Spanish", "French"),
)

st.write("You selected:", option)

if button_clicked:

    transcript = None

    if llm_string is not None:
        loader = YoutubeLoader.from_youtube_url(llm_string, add_video_info=False)
        
        docs = loader.load()

        transcript = docs[0].page_content

        st.write("transcription")
        stx.scrollableTextbox(transcript, height=200)

        if transcript:
            translation_chain = ({"answer": chain, "language": itemgetter("language")} | translation_prompt | llm | parser)
            
            translate_button = st.button("Translate")

            if translate_button:
            
                translation_result = translation_chain.invoke(
            {
                "answer" : transcript,
                "language": option,
            }
        )
                print(option)

                print(translation_result)

                st.write("translation")
                stx.scrollableTextbox(translation_result, height=200)


    else:
        st.warning("Enter a valid youtube url")


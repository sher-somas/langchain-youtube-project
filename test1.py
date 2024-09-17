import os
from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from operator import itemgetter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from sklearn.metrics.pairwise import cosine_similarity
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import DocArrayInMemorySearch
from templates import get_test_template
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_pinecone import PineconeVectorStore
import warnings
warnings.filterwarnings('ignore')
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# This is the YouTube video we're going to use.
YOUTUBE_VIDEO = "https://www.youtube.com/watch?v=cdiD-9MMpb0"


model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")

# model.invoke("What MLB team won the World Series during the COVID-19 pandemic?")


parser = StrOutputParser()

chain = model | parser
# chain.invoke("What MLB team won the World Series during the COVID-19 pandemic?")


input_template = get_test_template()
prompt = ChatPromptTemplate.from_template(input_template)
prompt.format(context="Mary's sister is Susana", question="Who is Mary's sister?")

chain = prompt | model | parser
chain.invoke(
    {"context": "Mary's sister is Susana", "question": "Who is Mary's sister?"}
)

translation_prompt = ChatPromptTemplate.from_template(
    "Translate {answer} to {language}"
)



translation_chain = (
    {"answer": chain, "language": itemgetter("language")} | translation_prompt | model | parser
)

translation_result = translation_chain.invoke(
    {
        "context": "Mary's sister is Susana. She doesn't have any more siblings.",
        "question": "How many sisters does Mary have?",
        "language": "Kannada",
    }
)

print(translation_result)

with open("transcription.txt") as file:
    transcription = file.read()

print(transcription[:100])
# try:
#     chain.invoke({
#         "context": transcription,
#         "question": "Is reading papers a good idea?"
#     })
# except Exception as e:
#     print(e)



loader = TextLoader("transcription.txt")
text_documents = loader.load()
text_documents



text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
text_splitter.split_documents(text_documents)[:5]

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
documents = text_splitter.split_documents(text_documents)


embeddings = OpenAIEmbeddings()
embedded_query = embeddings.embed_query("Who is Mary's sister?")

# print(f"Embedding length: {len(embedded_query)}")
# print(embedded_query[:10])

sentence1 = embeddings.embed_query("Mary's sister is Susana")
sentence2 = embeddings.embed_query("Pedro's mother is a teacher")
sentence3 = embeddings.embed_query("antarctica is a cold continent")



query_sentence1_similarity = cosine_similarity([embedded_query], [sentence1])[0][0]
query_sentence2_similarity = cosine_similarity([embedded_query], [sentence2])[0][0]
query_sentence3_similarity = cosine_similarity([embedded_query], [sentence3])[0][0]

# print(query_sentence1_similarity, query_sentence2_similarity, query_sentence3_similarity)



vectorstore1 = DocArrayInMemorySearch.from_texts(
    [
        "Mary's sister is Susana",
        "John and Tommy are brothers",
        "Patricia likes white cars",
        "Pedro's mother is a teacher",
        "Lucia drives an Audi",
        "Mary has two siblings",
    ],
    embedding=embeddings,
)

retriever1 = vectorstore1.as_retriever()
retriever1.invoke("Who is Mary's sister?")



setup = RunnableParallel(context=retriever1, question=RunnablePassthrough())
setup.invoke("What color is Patricia's car?")

chain = setup | prompt | model | parser
chain.invoke("What color is Patricia's car?")

vectorstore2 = DocArrayInMemorySearch.from_documents(documents, embeddings)

setup2 = RunnableParallel(context=vectorstore2.as_retriever(), question=RunnablePassthrough())

chain = (
    {"context": vectorstore2.as_retriever(), "question": RunnablePassthrough()}
    | prompt
    | model
    | parser
)

# Can invoke this chain as well. Much simpler format
chain2 = setup2 | prompt | model | parser
# chain.invoke("What is synthetic intelligence?")
chain2_result = chain2.invoke("What is AGI?")
print(chain2_result)



load_dotenv()
index_name = "test-project"

pinecone = PineconeVectorStore.from_documents(
    documents, embeddings, index_name=index_name
)

pinecone.similarity_search("What is Hollywood going to start doing?")[:3]

chain = (
    {"context": pinecone.as_retriever(), "question": RunnablePassthrough()}
    | prompt
    | model
    | parser
)

print(chain.invoke("What is Hollywood going to start doing?"))
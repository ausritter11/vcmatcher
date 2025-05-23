import os
import streamlit as st
import io
import pandas as pd
from typing_extensions import TypedDict, List
from langchain_community.chat_models import ChatOpenAI
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import START, StateGraph

# Set up Streamlit
st.set_page_config(page_title="VC Matcher", page_icon="🤝", layout="wide")
st.title("🤖 VC Matchmaker for Red Beard Ventures & Denarii Labs")
st.markdown("This app preloads a VC relationship CSV and matches your startup to the best-fit VCs in our network.")

# Session state initialization
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "graph" not in st.session_state:
    st.session_state.graph = None

# Load OpenAI API key from Streamlit secrets
openai_api_key = st.secrets["OPENAI_API_KEY"]
os.environ["OPENAI_API_KEY"] = openai_api_key

# Preload CSV and convert to text
@st.cache_data(show_spinner="Loading VC relationship data...")
def load_csv_text(file_path):
    df = pd.read_csv(file_path)
    return df.to_string(index=False)

csv_text = load_csv_text("vc_relationships.csv")  # CSV file path in repo
vc_doc = Document(page_content=csv_text, metadata={"source": "vc_relationships.csv"})

# Initialize RAG system
if st.button("Initialize VC Matcher") or ("initialized" not in st.session_state):
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        vector_store = InMemoryVectorStore(embeddings)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        splits = text_splitter.split_documents([vc_doc])
        vector_store.add_documents(splits)
        st.session_state.vector_store = vector_store

        system_template = """You are a startup-VC matching assistant. Given startup details, recommend 3-5 VCs from the context that are a strong fit. 
Use VC's preferred sector, stage, and any relationship strength if helpful.

Context:
{context}

Startup Description:
{question}

Answer with a concise list of VC names with reasons."""

        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("human", "{question}")
        ])

        class State(TypedDict):
            question: str
            context: List[Document]
            answer: str

        def retrieve(state: State):
            docs = vector_store.similarity_search(state["question"], k=4)
            return {"context": docs}

        def generate(state: State):
            context_text = "\n\n".join(doc.page_content for doc in state["context"])
            llm = ChatOpenAI(model="gpt-4o-mini")
            messages = [
                {"type": "system", "content": system_template.format(context=context_text, question=state["question"])} ,
                {"type": "human", "content": state["question"]}
            ]
            response = llm.invoke(messages)
            return {"answer": response.content}

        graph = StateGraph(State).add_sequence([retrieve, generate])
        graph.add_edge(START, "retrieve")
        st.session_state.graph = graph.compile()
        st.session_state.initialized = True
        st.success("RAG system initialized successfully!")

    except Exception as e:
        st.error(f"Failed to initialize: {str(e)}")

# Input box for startup matching
if st.session_state.graph:
    user_input = st.text_area("Enter your startup info (sector, stage, focus, etc):")
    if st.button("Find VC Matches") and user_input:
        with st.spinner("Matching with VCs..."):
            result = st.session_state.graph.invoke({"question": user_input})
            st.subheader("Top VC Matches:")
            st.markdown(result["answer"])

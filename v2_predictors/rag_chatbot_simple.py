import os
import re
import argparse
import json 

from transformers import pipeline
from huggingface_hub import login
from langchain_huggingface import HuggingFaceEndpoint
from langchain_ollama import OllamaEmbeddings

from functools import partial

import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from langchain_core.prompts import PromptTemplate
from typing_extensions import List, TypedDict, Annotated
from typing import Literal

WEB_SOURCE = "https://lilianweng.github.io/posts/2023-06-23-agent/"

os.environ["LANGSMITH_TRACING"] = "true"
QUERY_SPLIT_START_LABEL = "beginning"
QUERY_SPLIT_MID_LABEL = "middle"
QUERY_SPLIT_END_LABEL = "end"
QUERY_SPLIT_UNDEFINED_LABEL = "undefined"

RAG_PROMPT = """
Instructions: \
Use the provided context to answer the question at the end. \
If you don't find the answer from the context, just say that you don't know. \
Give one answer and stop after you have provided first answer.

Context: {context}

Question: {question}
"""

QUERY_ANALYSIS_PROMPT = """
Generate a JSON object for the following question:
{question}

Return valid JSON contaning query and section in format:
{{
    "section": "<one of 'beginning', 'middle', 'end'>",
    "query": "<provided question>"
}}

json:"""

VECTOR_STORAGE = "faiss_index_RAG"
parser = argparse.ArgumentParser("stock_predict")
parser.add_argument('--langsmith_api_key', required=True, help='API key')
parser.add_argument('--huggingface_api_key', required=True, help='API key')
parser.add_argument('--use_existing_storage', dest='use_existing_storage', default=False, action=argparse.BooleanOptionalAction)
args = parser.parse_args()

class SearchQuery(TypedDict):
    query: Annotated[str, ..., "Search query to run."]
    section: Annotated[
        Literal[QUERY_SPLIT_START_LABEL, QUERY_SPLIT_MID_LABEL, QUERY_SPLIT_END_LABEL, QUERY_SPLIT_UNDEFINED_LABEL],
        ...,
        "Section to query.",
    ]

class LLMState(TypedDict):
    question: str
    query: SearchQuery
    context: List[Document]
    answer: str
    

def retrieveContextFromStore(state, vector_store):
    query = state["query"]
    retrieved_docs = vector_store.similarity_search(query["query"], k=8)
    if not query["section"] == QUERY_SPLIT_UNDEFINED_LABEL:
        filtered_docs = [doc for doc in retrieved_docs if doc.metadata.get("section") == query["section"]]
        if filtered_docs:
            retrieved_docs = filtered_docs
    return {"context": retrieved_docs}

def analyzeQuery(state, llm):
    query = {"query": state['question'], "section": QUERY_SPLIT_UNDEFINED_LABEL}
    try:
        query_analyze_prompt = PromptTemplate.from_template(QUERY_ANALYSIS_PROMPT)
        messages = query_analyze_prompt.invoke({"question": state["question"]})
        response = llm(messages.to_string(), max_length=512)
        if isinstance(response, list) and len(response) > 0:
            print(f"DEBUG: Check query analysis answer: {response}")
            response_text = response[0].get("generated_text", "")
            try:
                query = json.loads(response_text)
            except json.JSONDecodeError as e:
                print(f"Error parsing the response with error: {e}")
        else:
            print("Warning: Unexpected response format from LLM.")
    except Exception as e:
        print(f"Error invoking model for analyzing query: {e}")
    return {"query": query}
    
def generateResponse(state, llm):
    rag_prompt = PromptTemplate.from_template(RAG_PROMPT)
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = rag_prompt.invoke({"question": state["question"], "context": docs_content})
    try:
        response = llm(messages.to_string(), max_length=512, do_sample=False, top_p=None, temperature=None)
    except Exception as e:
        print(f"Error invoking model: {e}")
        return {"answer": "Sorry, I encountered an issue."}
    if isinstance(response, list) and len(response) > 0:
        return {"answer": response[0].get("generated_text", "")}
    
def loadDataSourceFromWeb(web_page):
    loader = WebBaseLoader(
        web_paths=(web_page,),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    return loader.load()
    

def main(): 
    if not os.environ.get("LANGSMITH_API_KEY"):
        os.environ["LANGSMITH_API_KEY"] = args.langsmith_api_key
        
    login(token=args.huggingface_api_key)
    
    model_name = "meta-llama/Llama-3.2-1B"
    llm = pipeline("text-generation",
        model=model_name, model_kwargs={"torch_dtype": "bfloat16"},
        device_map="cpu",
        temperature=0.1,
        return_full_text=False
    )

    embeddings = OllamaEmbeddings(model="llama3")
    all_splits = None
    if not args.use_existing_storage:
        print(f"DEBUG: Loading external data source for RAG context")
        external_data = loadDataSourceFromWeb(WEB_SOURCE)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True,)
        all_splits = text_splitter.split_documents(external_data)
        
        total_documents = len(all_splits)
        third = total_documents // 3

        for i, document in enumerate(all_splits):
            if i < third:
                document.metadata["section"] = QUERY_SPLIT_START_LABEL
            elif i < 2 * third:
                document.metadata["section"] = QUERY_SPLIT_MID_LABEL
            else:
                document.metadata["section"] = QUERY_SPLIT_END_LABEL

    vector_store = None
    if args.use_existing_storage:
        vector_store = FAISS.load_local(VECTOR_STORAGE, embeddings, allow_dangerous_deserialization=True)
    else:
        print(f"DEBUG: Storing context data and adding embeddings to it")
        vector_store = FAISS.from_documents(all_splits, embeddings)
        vector_store.save_local(VECTOR_STORAGE)
    
    print(f"DEBUG: running the LLM with augmented context")
    graph_builder = StateGraph(LLMState).add_sequence([
        ("analyzeQuery", partial(analyzeQuery, llm=llm)),
        ("retrieveContext", partial(retrieveContextFromStore, vector_store=vector_store)),
        ("generateResponse", partial(generateResponse, llm=llm))
    ])
    graph_builder.add_edge(START, "analyzeQuery")
    graph = graph_builder.compile()

    for step in graph.stream(
        {"question": "What does the beginning of the post say about Task Decomposition?"},
        stream_mode="updates",
    ):
        print(f"{step}\n\n----------------\n")
    
    

if __name__ == "__main__":
    main()

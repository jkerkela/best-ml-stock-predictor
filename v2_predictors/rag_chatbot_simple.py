import os
import argparse
import json
import pickle
import numpy as np

from transformers import pipeline
from huggingface_hub import login
from langchain_huggingface import HuggingFaceEndpoint
from langchain_ollama import OllamaEmbeddings
import bm25s

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

#TODO: the answer is too long and it contains extra information not directly related to question.
RAG_PROMPT = """
Instructions: \
Use the provided context to answer the question at the end. \
Treat the whole context as equal, do not prioritize any section, even if question contains section information. \
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
    "section": "<one of 'beginning', 'middle', 'end', undefined>",
    "query": "<question>"
}}

json:"""

VECTOR_STORAGE = "faiss_index_RAG"
BM25_STORAGE = "bm25_index.pkl"

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
    

def retrieveContextFromStore(state, vector_store, bm25_index_data):
    top_k = 6
    query = state["query"]
    retrieved_docs = vector_store.similarity_search(query["query"], k=top_k)
    retrieved_doc_ids = []
    if not query["section"] == QUERY_SPLIT_UNDEFINED_LABEL:
        filtered_docs = [doc for doc in retrieved_docs if doc.metadata.get("section") == query["section"]]
        if filtered_docs:
            retrieved_docs = filtered_docs
    
    for doc in retrieved_docs:
        doc_id = doc.metadata.get("id")
        if doc_id is not None:
            retrieved_doc_ids.add(doc_id)
    
    
    bm25_engine = bm25_index_data["bm_engine"]
    original_docs_data = bm25_index_data["documents"]
    query_tokens = bm25s.tokenize(query["query"], stopwords="en")
    docs_as_text, bm25_scores = bm25_engine.retrieve(query_tokens, k=top_k)
    bm25_scores = bm25_scores[0]
    number_of_first_chars_to_compare = 20

    bm25_results = []
    for idx, doc_text in enumerate(docs_as_text[0]):
        original_index = None
        orig_doc_full = None
        bm25_index_text = doc_text
        for orig_doc in original_docs_data:
            if orig_doc.page_content[:number_of_first_chars_to_compare] == bm25_index_text[:number_of_first_chars_to_compare]:
                original_index = orig_doc.metadata["id"]
                orig_doc_full = orig_doc
                break
        if not orig_doc_full:
            continue
        score = bm25_scores[idx]
        # Skip if document is already in the dense retrieval results
        if original_index in retrieved_doc_ids:
            continue
        bm25_results.append({
            "id": original_index,
            "rank": idx,
            "score": score,
            "doc": orig_doc_full,
        })
        
    #TODO: we can do rank fusion of bm25 and vectorstore embedding matches
    combined_docs = []
    if bm25_results:
        combined_docs += [item["doc"] for item in bm25_results]
    if retrieved_docs:
        combined_docs += retrieved_docs  
    return {"context": combined_docs}

def analyzeQuery(state, llm):
    query = {"query": state['question'], "section": QUERY_SPLIT_UNDEFINED_LABEL}
    try:
        query_analyze_prompt = PromptTemplate.from_template(QUERY_ANALYSIS_PROMPT)
        messages = query_analyze_prompt.invoke({"question": state["question"]})
        response = llm(messages.to_string(), max_new_tokens=256, temperature=0.5)
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
        response = llm(messages.to_string(), max_new_tokens=256, do_sample=False, top_p=None, temperature=None)
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
            document.metadata["id"] = i
            if i < third:
                document.metadata["section"] = QUERY_SPLIT_START_LABEL
            elif i < 2 * third:
                document.metadata["section"] = QUERY_SPLIT_MID_LABEL
            else:
                document.metadata["section"] = QUERY_SPLIT_END_LABEL
    
    vector_store = None
    bm25_index_data = None
    if args.use_existing_storage:
        vector_store = FAISS.load_local(VECTOR_STORAGE, embeddings, allow_dangerous_deserialization=True)
        with open(BM25_STORAGE, "rb") as f:
            bm25_index_data = pickle.load(f)
    else:
        print(f"DEBUG: Storing context data and adding embeddings to it")
        vector_store = FAISS.from_documents(all_splits, embeddings)
        vector_store.save_local(VECTOR_STORAGE)
        
        documents_as_text = [doc.page_content for doc in all_splits]
        tokenized_corpus = bm25s.tokenize(documents_as_text, stopwords="en")
        bm25_engine = bm25s.BM25(corpus=documents_as_text)
        bm25_engine.index(tokenized_corpus)
        bm25_index_data = {"bm_engine": bm25_engine, "documents": all_splits}
        with open(BM25_STORAGE, "wb") as f:
            pickle.dump(bm25_index_data, f)
    
    print(f"DEBUG: running the LLM with augmented context")
    graph_builder = StateGraph(LLMState).add_sequence([
        ("analyzeQuery", partial(analyzeQuery, llm=llm)),
        ("retrieveContext", partial(retrieveContextFromStore, vector_store=vector_store, bm25_index_data=bm25_index_data)),
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

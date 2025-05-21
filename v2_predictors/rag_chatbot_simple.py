import os
from enum import Enum
import argparse
import json
import pickle
import fitz
import magic
import numpy as np
from urllib.parse import urlparse

from transformers import pipeline
from huggingface_hub import login
from langchain_huggingface import HuggingFaceEndpoint
from langchain_ollama import OllamaEmbeddings
import bm25s
import faiss
from sentence_transformers import CrossEncoder

from functools import partial

import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_community.document_compressors import JinaRerank
from langchain.retrievers import ContextualCompressionRetriever
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

#TODO: extend to types that FITZ doesn't support, when needed
class FileTypeSupport(Enum):
    FITZ = 1
    NON_SUPPORTED = 2

#TODO: the answer is too long and it contains extra information not directly related to question.
RAG_PROMPT = """
Instructions: \
Use the provided context to answer the question at the end. \
Treat the whole context as equal, do not prioritize any section, even if question contains section information. \
If you don't find the answer from the context, just say that you don't know. \
Always immdiately stop after you have provided first answer, do not provide follow-up questions or answers.

Context: {context}

Question: {question}
"""

QUERY_ANALYSIS_PROMPT = """
Generate a JSON object for the following question:
{question}

Return valid JSON containing query and section, in format:
{{
    "section": "<one of 'beginning', 'middle', 'end', undefined>",
    "query": "<question>"
}}

json:"""

VECTOR_STORAGE = "faiss_index_RAG"
BM25_STORAGE = "bm25_index.pkl"

parser = argparse.ArgumentParser("stock_predict")
parser.add_argument('--langsmith_api_key', required=True)
parser.add_argument('--huggingface_api_key', required=True)
parser.add_argument('--use_existing_storage', dest='use_existing_storage', default=False, action=argparse.BooleanOptionalAction)
parser.add_argument('--append_to_storage', dest='append_to_existing_storage',
    help='data source is appended to existing storage if used, if not data source will be stored to in memory storage only and used from there', default=False, action=argparse.BooleanOptionalAction)
parser.add_argument('--query', required=True)
parser.add_argument('--sources', nargs='+', required=False,
    help='The sources to use to answer for query, can be different types of documents or web page, if used with --append_to_storage, the sources will be stored to persistent storage')
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

def isValidUrl(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False
        
def getFileTypeSupport(file_path):
    supported_fitz_types = [
        "application/pdf", "application/vnd.ms-xpsdocument",
        "application/epub+zip", "application/x-cbz", "application/x-cbr"
    ]

    mime = magic.Magic(mime=True)
    file_type = mime.from_file(file_path)

    if file_type in supported_fitz_types:
        return FileTypeSupport.FITZ
    else:
        return FileTypeSupport.NON_SUPPORTED
        
def retrieveContextFromStore(state, reranker_llm, vector_store, bm25_index_data):
    top_k = 10
    top_k_rerank = 5
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
            retrieved_doc_ids.append(doc_id)
    
    
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
        # Skip if document is already in the vector store retrieval results
        if original_index in retrieved_doc_ids:
            continue
        bm25_results.append({
            "id": original_index,
            "rank": idx,
            "score": score,
            "doc": orig_doc_full,
        })
        
    combined_docs = []
    if bm25_results:
        combined_docs += [item["doc"] for item in bm25_results]
    if retrieved_docs:
        combined_docs += retrieved_docs
    
    doc_texts = [doc.page_content for doc in combined_docs]
    doc_rank_infos = reranker_llm.rank(query["query"], doc_texts, top_k=top_k_rerank)
    reranked_docs = []
    for doc_rank_info in doc_rank_infos:
        reranked_docs.append(combined_docs[doc_rank_info["corpus_id"]])
    return {"context": reranked_docs}

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
   
def updateRankingStorage(source_splits, store_persistently):
    documents_as_text = [doc.page_content for doc in source_splits]
    tokenized_corpus = bm25s.tokenize(documents_as_text, stopwords="en")
    bm25_engine = bm25s.BM25(corpus=documents_as_text)
    bm25_engine.index(tokenized_corpus)
    bm25_index_data = {"bm_engine": bm25_engine, "documents": source_splits}
    if store_persistently:
        with open(BM25_STORAGE, "wb") as f:
            pickle.dump(bm25_index_data, f)
    return bm25_index_data

def main():
    print(f"DEBUG: starting to load depedency llm models")
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
    
    reranker_llm = CrossEncoder("jinaai/jina-reranker-v1-tiny-en", trust_remote_code=True)

    embeddings = OllamaEmbeddings(model="llama3")
    print(f"DEBUG: Loading external data source for RAG context")
    source_splits = []
    if args.sources:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500,
            chunk_overlap=100,
            add_start_index=False,
            separators=["\n\n"]
        )
        for source in args.sources:
            if isValidUrl(source):
                external_data_as_doc_type = loadDataSourceFromWeb(source)
                source_splits += text_splitter.split_documents(external_data_as_doc_type)
            else:
                support = getFileTypeSupport(source)
                if support == FileTypeSupport.FITZ:
                    doc = fitz.open(source)
                    external_data_as_doc_type = []
                    for page in doc:
                        text = page.get_text("text")
                        metadata = {"page": page.number}
                        external_data_as_doc_type.append(Document(page_content=text, metadata=metadata))
                    source_splits += text_splitter.split_documents(external_data_as_doc_type)
                else:
                    raise Exception("File type not supported")
        
        total_documents = len(source_splits)
        third = total_documents // 3

        for i, document in enumerate(source_splits):
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
        print(f"DEBUG: loading context data from existing storage")
        ranking_storage_needs_reindexing = False
        vector_store = FAISS.load_local(VECTOR_STORAGE, embeddings, allow_dangerous_deserialization=True)
        with open(BM25_STORAGE, "rb") as f:
            bm25_index_data = pickle.load(f)
    
    if args.sources:
        faiss.omp_set_num_threads(4)
        in_memory_vector_store = FAISS.from_documents(source_splits, embeddings)
        if args.use_existing_storage:
            vector_store.merge_from(in_memory_vector_store)
        else:
            vector_store = in_memory_vector_store
        if args.append_to_storage:
            print(f"DEBUG: Storing context data and adding embeddings to it")
            vector_store.save_local(VECTOR_STORAGE)
            bm25_index_data = updateRankingStorage(source_splits, True)
        else:
            bm25_index_data = updateRankingStorage(source_splits, False)

    
    print(f"DEBUG: running the LLM with augmented context for query: {args.query}")
    graph_builder = StateGraph(LLMState).add_sequence([
        ("analyzeQuery", partial(analyzeQuery, llm=llm)),
        ("retrieveContext", partial(retrieveContextFromStore, reranker_llm=reranker_llm, vector_store=vector_store, bm25_index_data=bm25_index_data)),
        ("generateResponse", partial(generateResponse, llm=llm))
    ])
    graph_builder.add_edge(START, "analyzeQuery")
    graph = graph_builder.compile()

    for step in graph.stream(
        {"question": args.query},
        stream_mode="updates",
    ):
        print(f"{step}\n\n----------------\n")
    
    

if __name__ == "__main__":
    main()

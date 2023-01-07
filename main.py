from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Tuple
import pickle
import numpy as np
import time
import os
from dotenv import load_dotenv

# ! Get Embeddings Imports
from sentence_transformers import SentenceTransformer
from torch import save
import torch

# ! Get Tokens Imports
from transformers import GPT2Tokenizer

# ! Get Response Imports
import openai

# ! S3 Imports
import boto3

# ! Periodic Tasks
from fastapi_utils.session import FastAPISessionMaker
from fastapi_utils.tasks import repeat_every

# ! START CONFIG -------------------------------------

app = FastAPI()

origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv(".env")
api_key = os.environ.get('OPENAI-API-KEY')
openai.api_key = api_key

# ? Tokenizer
try:
    tokenizer = torch.load("tokenizer.pt")
    print("Tokenizer Successfully Loaded")
except:
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    print("Tokenizer Downloaded")
    torch.save(tokenizer, "tokenizer.pt")

# ? Embeddings Model
try:
    model = torch.load("model.pt")
    print("Model Successfully Loaded")
except:
    model = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-dot-v1')
    print("Model Downloaded")
    torch.save(model, "model.pt")

# ? Config S3
s3_client = boto3.client('s3')
s3_resource = boto3.resource('s3')

# ! END CONFIG -------------------------------------
# ! START CLASSES -------------------------------------
class Text(BaseModel):
    text: str

class Query(BaseModel):
    query: str
    doc_name: str

# ! END CLASSES -------------------------------------
# ! START FUNCTIONS -------------------------------------
# ? START FUNCTION DEFINITION -------------------------------------
# ! Generate Embeddings
def get_embedding(data: str, model) -> List[float]:
    print("Generating Embedding")
    embedding = model.encode(data)
    print("Embedding Successful")
    return embedding

def find_path(doc_name: str) -> str:
    for item in documents:
        if item['name'] == doc_name:
            return item['path']
        print(f"FOLDER NAMES:")
        print([doc['name'] for doc in documents])
        print(f"SELECTED DOC NAME: {doc_name}")
    raise ValueError("Document Not Found")

# ! Load Doc
def load_doc(fpath: str):
    with open(fpath, 'rb') as f:
        data = pickle.load(f)
    print("Complete")
    return data

# ! Scan Local Documents
def scan_local_documents():
    documents_folder_path = "documents/"
    # load all filepaths in the documents folder
    documents = []
    # { 'name' : "document name", 'path' : "filepath"}
    for item in os.listdir(documents_folder_path):
        item_path = os.path.join(documents_folder_path, item)
        item_name = item.split(".")[0]
        temp = {"name": item_name, "path": item_path}
        documents.append(temp)
    
    return documents

# ! Check S3 for new documents
def check_for_new_documents(documents, bucket_name):
    bucket = s3_resource.Bucket(bucket_name)
    s3_documents = []
    for obj in bucket.objects.all():
        # documents/filename.pkl
        filename = obj.key.split("/")[-1]
        # filename with the extension
        temp = {"name": filename, "key": obj.key}
        s3_documents.append(temp)

    print(s3_documents)

    # ? Download New Documents from S3
    for s3_doc in s3_documents:
        # if s3 doc isnt local, download it
        name_no_ext = s3_doc['name'].split(".")[0]
        if name_no_ext not in [doc['name'] for doc in documents]:
            print(f"Downloading {s3_doc['name']} from S3 bucket")
            s3_client.download_file(bucket_name, s3_doc['key'], f"documents/{s3_doc['name']}")

# ! Unpack Dict
def unpack(doc_data: List[Dict]):
    # unpack each every dict value for each key into a list
    doc_text = [doc_data[i]['text'] for i in range(len(doc_data))]
    doc_page = [doc_data[i]['page'] for i in range(len(doc_data))]
    doc_emb = np.array([doc_data[i]['embedding'] for i in range(len(doc_data))])
    # return all three lists
    return doc_page, doc_text, doc_emb

# ! Calculate Similarity
def dot_product_similarity(doc_data: List[Dict], query_data : Dict) -> List[Tuple[int, float]]:
    query_embedding = query_data['embedding']
    doc_embeddings = [page['embedding'] for page in doc_data]

    tuples_list = []
    for index, embedding in enumerate(doc_embeddings):
        page = doc_data[index]['page']
        similarity = np.dot(query_embedding, embedding)
        tuples_list.append((page, similarity))

    ordered_tuples = sorted(tuples_list, key=lambda x: x[1], reverse=True)
    # get top 5 and flip the values
    top_five_tuples = ordered_tuples[:5]

    return top_five_tuples

# ! Add Embedding to Data
def get_single_embedding(data: Dict) -> Dict:
    query = data['query']
    if 'embedding' not in data:
        data['embedding'] = get_embedding(query, model)
    return data

# ! Get Most Relevant Context Data
def get_context(query: str, doc_data : List[Dict]) -> List[Dict]:
    # create the query data
    query_data = {'query': query}
    # get the embedding and add it to the data
    query_data = get_single_embedding(query_data)
    # unit test
    assert query_data['embedding'].all()
    # get top five pages
    top_five_tuples = dot_product_similarity(doc_data, query_data)

    # generate the context object
    context = []
    for item in top_five_tuples:
        page_no = item[0]
        # get first 50 chars of text in doc_data with matching page number
        text = doc_data[page_no - 1]['text']
        # make the dict
        data = {'page': page_no, 'similarity' : item[1], 'text': text}
        # append the dict
        context.append(data)

    return context

# ! Get Number of Tokens
def get_tokens(text: str, tokenizer):
    start = time.perf_counter()
    tokens = tokenizer.tokenize(text)
    elapsed = time.perf_counter() - start
    return (len(tokens), elapsed)

# ! Get Response from OpenAI
def get_response(prompt: str) -> str:
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0,
        max_tokens=1000,
    )
    return response['choices'][0]['text'].strip(" \n")

# ! Format the Context Data
def get_formatted_context(context):
    formatted_text = ''
    pages = []
    print(f"Context length: {len(context)}")
    for idx, item in enumerate(context):
        page_no = item['page']
        text = item['text']
        formatted_text += f"PAGE: {page_no} - {text}\n\n"
        pages.append(page_no)

        tokens = get_tokens(formatted_text, tokenizer)[0]
        # print(f"Current Context Tokens: {tokens}")
        if tokens > 2800:
            while tokens > 2800:
                formatted_text = formatted_text[:-1000]
                tokens = get_tokens(formatted_text, tokenizer)[0]
            break
    
    print(f"Final Context Tokens: {get_tokens(formatted_text, tokenizer)[0]}")
    return formatted_text, pages

# ! Build the Prompt
def build_prompt(query: str, context: List[Dict], *, examples=None) -> str:
    header = "You are a college business management professor. You are teaching your students. Answer the query with a lengthy, deatiled reponse, to the best of your ability based on the provided context. If you want, include examples of how one should apply the concept to the real world. If you dont know something, say 'I don't know.' If the question doesn't make sense, say 'I don't understand the question. Can you please clarify?'"
    book_text, context_pages = get_formatted_context(context)

    if examples is not None:
        prompt = f"HEADER:\n{header}\n\nCONTEXT:\n{book_text}\n\nEXAMPLES:\nHere are some examples, use them as a general example, dont copy them.\n{examples}\n\nQUERY:\n{query}\n\nOUTPUT:\n"
    else:
        prompt = f"HEADER:\n{header}\n\nCONTEXT:\n{book_text}\n\nQUERY:\n{query}\n\nOUTPUT:\n"
    
    tokens = get_tokens(prompt, tokenizer)
    print(f"Prompt tokens: {tokens[0]}")
    return prompt, context_pages

# ! END FUNCTIONS -------------------------------------
# ! START MAIN -------------------------------------

# ? Load Documents, Check for new documents
documents = scan_local_documents()
check_for_new_documents(documents, "evertech-app")

# ? sync the documents with S3 every 12 hours
@app.on_event("startup")
@repeat_every(seconds=60 * 60 * 12, wait_first=True)
async def run_check_for_new_documents():
    check_for_new_documents(documents, "evertech-app")
    print('Synced all documents')

# ! START ROUTES -------------------------------------
@app.get("/")
def root():
    return {"message": "API is up and running."}

@app.get("/embeddings")
def route_embedding_info():
    return {"message": "This endpoint expects a POST request with a JSON body containing a text field. It returns a vector embedding of dimensions (768,)."}

@app.post("/embeddings")
async def handle_embedding(data: Text):
    embedding = await get_embedding(data.text, model)
    response_embedding = embedding.tolist()
    return {"embedding": response_embedding}

@app.get("/tokens")
def route_tokens_info():
    return {"message": "This endpoint expects a POST request with a JSON body containing a text field. It returns the number of tokens."}

@app.post("/tokens")
async def handle_tokens(data: Text):
    tokens = get_tokens(data.text, tokenizer)
    token_len = len(tokens)
    return {"tokens": token_len}

@app.get("/docs/list")
def route_get_docs():
    return {"message": "This endpoint returns a list of all the documents in the database.", "docs": [document['name'] for document in documents]}

@app.get("/docs/sync")
def handle_refresh_docs():
    check_for_new_documents(documents, "evertech-app")
    return {"message": "Documents have been refreshed."}

@app.get("/semantic-qa")
def route_semantic_qa_info():
    return {"message": "This endpoint expects a POST request with a JSON body containing a query and doc_name field. It returns a response to the query.", "availible_docs": [document['name'] for document in documents]}

@app.post(f"/semantic-qa")
async def handle_semantic_qa(data: Query, examples=None):
    start = time.perf_counter()

    fpath = find_path(data.doc_name)
    doc_data = load_doc(fpath)
    context = get_context(data.query, doc_data)
    prompt, context_pages = build_prompt(data.query, context, examples=examples)
    response = get_response(prompt)

    elapsed = time.perf_counter() - start
    tokens = get_tokens(prompt + response, tokenizer)[0]
    est_cost = tokens / 1000 * 0.02

    print(f"Query:\n{data.query}\n")
    print(f"Response:\n{response}\n")
    print("DEBUG:\n")
    print(f"Context Pages: {context_pages}")
    print(f"Time: {elapsed:.2f} seconds")
    print(f"Total Tokens: {tokens}")
    print(f"Cost: ${est_cost:.2f}")
    rounded_cost = round(est_cost, 2)
    rounded_time = round(elapsed, 2)
    
    # convert page numbers to strings
    # context_pages = [str(page) for page in context_pages]

    return {"time": rounded_time, "tokens": tokens, "cost": rounded_cost, "response": response, "context_pages": context_pages}
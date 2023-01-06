from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Tuple
import pickle
import numpy as np
import time

# ! Get Embeddings Imports
from sentence_transformers import SentenceTransformer
from torch import save
import torch

# ! Get Tokens Imports
from transformers import GPT2Tokenizer

# ! Get Response Imports
import openai

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

# load the tokenizer
print("Loading Tokenizer...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
print("Tokenizer Loaded")

# ! END CONFIG -------------------------------------
# ! START MODEL ------------------------------------
# Load the model
# Enable tracemalloc to track memory usage    
try:
    model = torch.load("model.pt")
    print("Model Successfully Loaded")
except:
    model = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-dot-v1')
    print("Model Downloaded")
    torch.save(model, "model.pt")

# ! END MODEL -------------------------------------
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
async def get_embedding(data: str, model) -> List[float]:
    print("Generating Embedding")
    embedding = model.encode(data)
    print("Embedding Successful")
    return embedding

# ! Load Doc
def load_doc(fpath: str):
    with open(fpath, 'rb') as f:
        data = pickle.load(f)
    print("Complete")
    return data

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
        embedding = get_embedding(query)['embedding']
        data['embedding'] = embedding
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

        tokens = get_tokens(formatted_text)[0]
        # print(f"Current Context Tokens: {tokens}")
        if tokens > 2800:
            while tokens > 2800:
                formatted_text = formatted_text[:-1000]
                tokens = get_tokens(formatted_text)[0]
            break
    
    print(f"Final Context Tokens: {get_tokens(formatted_text)[0]}")
    return formatted_text, pages

# ! Build the Prompt
def build_prompt(query: str, context: List[Dict], *, examples=None) -> str:
    header = "You are a college business management professor. You are teaching your students. Answer the query with a lengthy, deatiled reponse, to the best of your ability based on the provided context. If you want, include examples of how one should apply the concept to the real world. If you dont know something, say 'I don't know.' If the question doesn't make sense, say 'I don't understand the question. Can you please clarify?'"
    book_text, context_pages = get_formatted_context(context)

    if examples is not None:
        prompt = f"HEADER:\n{header}\n\nCONTEXT:\n{book_text}\n\nEXAMPLES:\nHere are some examples, use them as a general example, dont copy them.\n{examples}\n\nQUERY:\n{query}\n\nOUTPUT:\n"
    else:
        prompt = f"HEADER:\n{header}\n\nCONTEXT:\n{book_text}\n\nQUERY:\n{query}\n\nOUTPUT:\n"
    
    tokens = get_tokens(prompt)
    print(f"Prompt tokens: {tokens[0]}")
    return prompt, context_pages

# ! END FUNCTIONS -------------------------------------
# ! START ROUTES -------------------------------------
@app.get("/")
def root():
    return {"message": "API is up and running. Use the /embeddings endpoint to generate singular embeddings."}

@app.get("/embeddings")
def get_embedding_info():
    return {"message": "This endpoint expects a POST request with a JSON body containing a text field. It returns a vector embedding of dimensions (768,)."}

@app.post("/embeddings")
async def handle_embedding(data: Text):
    embedding = await get_embedding(data.text, model)
    response_embedding = [str(x) for x in embedding]
    return {"embedding": response_embedding}

@app.get("/tokens")
def get_tokens_info():
    return {"message": "This endpoint expects a POST request with a JSON body containing a text field. It returns the number of tokens."}

@app.post("/tokens")
async def handle_tokens(data: Text):
    tokens = get_tokens(data.text, tokenizer)
    token_len = len(tokens)
    return {"tokens": token_len}

# @app.get(f"/semantic-qa/{doc_name}")
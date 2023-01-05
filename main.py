from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from torch import save
import torch

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

# ! END CONFIG -------------------------------------
# ! START MODEL ------------------------------------
# Load the model
try:
    model = torch.load("model.pt")
    print("Model Successfull Loaded")
except:
    model = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-dot-v1')
    print("Model Downloaded")
    torch.save(model, "model.pt")
# ! END MODEL -------------------------------------
# ! START FUNCTIONS -------------------------------------

async def get_embedding(data: str) -> List[float]:
    print("Generating Embedding")
    embedding = model.encode(data)
    print("Embedding Successful")
    return embedding

# ! END FUNCTIONS -------------------------------------
# ! START CLASSES -------------------------------------
class Data(BaseModel):
    text: str

# ! START ROUTES -------------------------------------
@app.get("/")
def root():
    return {"message": "API is up and running. Use the /embeddings endpoint to generate singular embeddings."}

@app.get("/embeddings")
def get_embedding_info():
    return {"message": "This endpoint expects a POST request with a JSON body containing a text field. It returns a vector embedding of dimensions (768,)."}

@app.post("/embeddings")
async def handle_embedding(data: Data):
    embedding = await get_embedding(data.text)
    response_embedding = [str(x) for x in embedding]
    return {"embedding": response_embedding}
from typing import Dict, List, Union, Tuple, Any, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from realestate_vss.models.embedding import OpenClipTextEmbeddingModel, OpenClipImageEmbeddingModel
from realestate_vss.search.engine import ListingSearchEngine, SearchMode

import realestate_core.common.class_extensions
from realestate_core.common.utils import join_df
from realestate_vision.common.utils import get_listingId_from_image_name

from concurrent.futures import ThreadPoolExecutor
import asyncio, io, json
import torch
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image



app = FastAPI()
# uvicorn main:app --reload &  # run this in terminal to start the server

# Set up CORS middleware configuration
app.add_middleware(
  CORSMiddleware,
  allow_origins=["http://localhost:3000"],  # Allows all origins
  allow_credentials=True,
  allow_methods=["*"],  # Allows all methods
  allow_headers=["*"],  # Allows all headers
)

# Global variable to hold the search engine instance
search_engine = None

# non global variables
local_project_home = Path('/Volumes/Samsung_T7/jumptools_gdrive/NLImageSearch')
model_name = 'ViT-L-14'
pretrained = 'laion2b_s32b_b82k'

async def load_images_dataframe():
  loop = asyncio.get_event_loop()
  with ThreadPoolExecutor() as pool:
    # Asynchronously read the feather file
    image_embeddings_df = await loop.run_in_executor(
        pool,
        pd.read_feather,
        local_project_home / 'deployment_listing_images' / f'{model_name}_{pretrained}' / 'embeddings_df'
    )

    # Asynchronously stack the embeddings
    # image_embeddings = await loop.run_in_executor(
    #     pool,
    #     lambda: np.stack(image_embeddings_df.embedding.values)
    # )

  return image_embeddings_df

async def load_texts_dataframe():
  loop = asyncio.get_event_loop()
  with ThreadPoolExecutor() as pool:
    text_embeddings_df = await loop.run_in_executor(
      pool,
      pd.read_feather,
      local_project_home / 'avm_listing_info' / f'{model_name}_{pretrained}' / 'text_embeddings_df'
    )

    listing_df = await loop.run_in_executor(
      pool,
      pd.read_feather,
      local_project_home / 'avm_listing_info' / 'listing_df'
    )

    # text_embeddings = await loop.run_in_executor(
    #   pool,
    #   lambda: np.stack(text_embeddings_df.embedding.values)
    # )

    text_embeddings_df = await loop.run_in_executor(
      pool,
      join_df,
      listing_df, text_embeddings_df, 'jumpId', 'listing_id'
    )

  return text_embeddings_df

@app.on_event("startup")
async def startup_event():
  global search_engine

  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
  image_embedder = OpenClipImageEmbeddingModel(model_name=model_name, pretrained=pretrained, device=device)
  text_embedder = OpenClipTextEmbeddingModel(embedding_model=image_embedder)

  # Async retrieval of embeddings
  image_embeddings_df = await load_images_dataframe()
  text_embeddings_df = await load_texts_dataframe()

  if 'listing_id' not in image_embeddings_df.columns:    
    image_embeddings_df['listing_id'] = image_embeddings_df.image_name.apply(get_listingId_from_image_name)

  search_engine = ListingSearchEngine(
        image_embeddings_df=image_embeddings_df, 
        text_embeddings_df=text_embeddings_df, 
        image_embedder=image_embedder, 
        text_embedder=text_embedder
        )

class ListingData(BaseModel):
  jumpId: str
  city: str
  provState: str
  postalCode: str
  lat: float
  lng: float
  streetName: Optional[str] = None
  beds: str
  bedsInt: int
  baths: str
  bathsInt: int
  sizeInterior: Optional[str] = None
  sizeInteriorUOM: Optional[str] = None
  lotSize: Optional[str] = None
  lotUOM: Optional[str] = None
  propertyFeatures: List[str]
  propertyType: str
  transactionType: str
  carriageTrade: bool
  price: int
  leasePrice: int
  pool: bool
  garage: bool
  waterFront: bool
  fireplace: bool
  ac: bool
  remarks: Optional[str] = None
  listing_id: str

@app.get("/listing/{listingId}")
async def get_listing(listingId: str) -> ListingData:
  listing_data = search_engine.get_listing(listingId)

  if not listing_data:
      raise HTTPException(status_code=404, detail="Listing not found")

  return ListingData(**listing_data)

@app.post("/search-by-image/")
async def search_by_image(file: UploadFile = File(...)) -> List[Dict[str, Union[str, float , List[str]]]]:
    """
    Using the provided image file as query to the search engine, return a list of listings with images
    that are most similar to it.

    The returned list is sorted by average similarity score in descending order.
    E.g.
    [
      {
        "listingId": "21125523",
        "avg_score": "0.7692008018493652",
        "image_names": [
          "21125523/21125523_4.jpg"
        ]
      },
      {
        "listingId": "20953965",
        "avg_score": "0.7561193406581879",
        "image_names": [
          "20953965/20953965_0.jpg",
          "20953965/20953965_1.jpg",
          "20953965/20953965_4.jpg",
          "20953965/20953965_9.jpg"
        ]
      },
      {
        "listingId": "20911679",
        "avg_score": "0.7544260919094086",
        "image_names": [
          "20911679/20911679_48.jpg",
          "20911679/20911679_49.jpg"
        ]
      },...
    ]

    """
    image_data = await file.read()

    try:
      image = Image.open(io.BytesIO(image_data))
    except Exception as e:
      return f'error: Invalid image file'

    try:
      image_names, scores = search_engine.image_search(image, topk=50)
    except Exception as e:
      return f'search engine error: {e}'

    # listingIds = [get_listingId_from_image_name(image_name) for image_name in image_names]
    # image names are of format {listingId}_{imageId}.jpg, we want to organize by listingId
    # such that we get a dict whose keys are listing_ids and values are list of image names
    # and another dict whose keys are listing_ids and values are list of corresponding scores
    listingId_to_image_names = {}
    listingId_to_scores = {}
    for image_name, score in zip(image_names, scores):
      listingId = get_listingId_from_image_name(image_name)
      if listingId not in listingId_to_image_names:
        listingId_to_image_names[listingId] = []
        listingId_to_scores[listingId] = []

      listingId_to_image_names[listingId].append(image_name)
      listingId_to_scores[listingId].append(score)

    listings = []
    for listingId, image_names in listingId_to_scores.items():
      avg_score = np.mean(np.array(listingId_to_scores[listingId]))      
      image_names = [f"{listingId}/{image_name}" for image_name in listingId_to_image_names[listingId]]
      listings.append({
        'listingId': listingId,
        "avg_score": float(avg_score),
        "image_names": image_names,
      })

    # Sort the listings by average score in descending order
    listings = sorted(listings, key=lambda x: x['avg_score'], reverse=True)

    return listings

@app.get("/images/{listingId}/{image_name}")
async def get_image(listingId: str, image_name: str) -> FileResponse:
  image_path = local_project_home / 'deployment_listing_images' / listingId / image_name
  if not image_path.is_file():
      raise HTTPException(status_code=404, detail="Image not found")
  return FileResponse(image_path)

@app.post("/search-by-text/")
async def search_by_text(query: Dict[str, Union[str, Optional[int], Optional[List[Optional[int]]]]], 
                         mode: SearchMode = Query(SearchMode.VSS_ONLY), 
                         lambda_val: float = Query(0.8, description="Required if mode is SOFT_MATCH_AND_VSS."), 
                         alpha_val: float = Query(0.5, description="Required if mode is SOFT_MATCH_AND_VSS.")
):
  
  phrase = query.get('phrase', None)
  if phrase is not None:
    del query['phrase']   # remove key phrase from query after extracting it

  provState = query.get('provState', None)
  
  print(f'query: {query}')
  if provState is None:
    raise HTTPException(status_code=404, detail="provState must be provided")

  if mode == SearchMode.VSS_ONLY or mode == SearchMode.VSS_RERANK_ONLY:
    results = search_engine.text_search(mode, topk=20, phrase=phrase, return_df=False, **query)
  elif mode == SearchMode.SOFT_MATCH_AND_VSS:
    results = search_engine.text_search(mode, topk=20, return_df=False, lambda_val=lambda_val, alpha_val=alpha_val, phrase=phrase, **query)
  else:
    raise HTTPException(status_code=404, detail="Invalid search mode")
  
  return results

@app.post("/text-to-image-search/")
async def text_to_image_search(query: Dict[str, Any]) -> List[Dict[str, Union[str, float , List[str]]]]:
  try:
    image_names, scores = search_engine.text_2_image_search(phrase=query['phrase'], topk=50)
  except Exception as e:
    return f'search engine error: {e}'

  listingId_to_image_names = {}
  listingId_to_scores = {}
  for image_name, score in zip(image_names, scores):
    listingId = get_listingId_from_image_name(image_name)
    if listingId not in listingId_to_image_names:
      listingId_to_image_names[listingId] = []
      listingId_to_scores[listingId] = []

    listingId_to_image_names[listingId].append(image_name)
    listingId_to_scores[listingId].append(score)

  listings = []
  for listingId, image_names in listingId_to_scores.items():
    avg_score = np.mean(np.array(listingId_to_scores[listingId]))      
    image_names = [f"{listingId}/{image_name}" for image_name in listingId_to_image_names[listingId]]
    listings.append({
      'listingId': listingId,
      "avg_score": float(avg_score),
      "image_names": image_names,
    })

  # Sort the listings by average score in descending order
  listings = sorted(listings, key=lambda x: x['avg_score'], reverse=True)

  return listings


# for testing before UI is built
@app.post("/search-by-image-html/", response_class=HTMLResponse)
async def search_by_image_html(file: UploadFile = File(...)) -> HTMLResponse:
    listings = await search_by_image(file)

    # Generate an HTML page with the images embedded
    html_content = "<html><body>"
    for listing in listings:
        for image_name in listing["image_names"]:
            html_content += f'<img src="/images/{image_name}" alt="{image_name}"><br>'
    html_content += "</body></html>"

    return HTMLResponse(content=html_content)

@app.get("/sanity_check")
async def sanity_check() -> str:
  # global image_embeddings_df, text_embeddings_df
  n_images = search_engine.image_embeddings_df.shape[0]
  n_listings = search_engine.image_embeddings_df.listing_id.nunique()

  n_remarks = search_engine.text_embeddings_df.shape[0]
  
  message = f'Searching {n_images} images in {n_listings} listings\n'
  message += f'Searching remarks for {n_remarks} listings'

  return message
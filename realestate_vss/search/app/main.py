from typing import Dict, List, Union, Tuple, Any, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel
from realestate_vss.models.embedding import OpenClipTextEmbeddingModel, OpenClipImageEmbeddingModel
from realestate_vss.search.engine import ListingSearchEngine, SearchMode

import realestate_core.common.class_extensions
from realestate_core.common.utils import join_df
from realestate_vision.common.utils import get_listingId_from_image_name
from realestate_vss.data.index import FaissIndex

from concurrent.futures import ThreadPoolExecutor

import httpx
import asyncio, io, json, os
import torch
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image

from dotenv import load_dotenv, find_dotenv

from google.cloud import storage
from google.auth.exceptions import DefaultCredentialsError

app = FastAPI()
# uvicorn main:app --reload &  # run this in terminal to start the server
# uvicorn main:app --host 0.0.0.0 --port 8002 --reload  (demo on GCP)

# Set up CORS middleware configuration
_ = load_dotenv(find_dotenv())
if "ALLOW_ORIGINS" in os.environ:
  allow_origins = os.environ["ALLOW_ORIGINS"].split(',')
else:
  raise Exception("ALLOW_ORIGINS environment variable not set in .env file")

app.add_middleware(
  CORSMiddleware,
  allow_origins=allow_origins,  # Allows all origins
  allow_credentials=True,
  allow_methods=["*"],  # Allows all methods
  allow_headers=["*"],  # Allows all headers
)

# Global variable to hold the search engine instance
search_engine = None

# non global variables
if "LOCAL_PROJECT_HOME" in os.environ:
  local_project_home = Path(os.getenv("LOCAL_PROJECT_HOME"))
else:
  raise Exception("LOCAL_PROJECT_HOME environment variable not set in .env file")

model_name = 'ViT-L-14'
pretrained = 'laion2b_s32b_b82k'

if "GCS_PROJECT_ID" in os.environ and "GCS_BUCKET_NAME" in os.environ:
  GCS_PROJECT_ID = os.getenv("GCS_PROJECT_ID")
  GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
  try:
    storage_client = storage.Client(project=GCS_PROJECT_ID)
    bucket = storage_client.bucket(GCS_BUCKET_NAME)
  except DefaultCredentialsError as e:
    print(f'Error connecting to GCS: {e}')
    bucket = None
  except Exception as e:
    print(f'Error connecting to GCS: {e}')
    bucket = None
else:
  bucket = None

# check if FAISS index folder is defined, if so, use FAISS index to later instantiate search engine
# and not use the dataframes
if "FAISS_INDEX_FOLDER" in os.environ:
  faiss_index_folder = Path(os.getenv("FAISS_INDEX_FOLDER"))
  if not faiss_index_folder.is_dir():
    raise Exception("FAISS_INDEX_FOLDER is not a valid directory")
  print(f'Using FAISS index from {faiss_index_folder}')
else:
  faiss_index_folder = None
  print('Not using FAISS index')

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

    text_embeddings_df = await loop.run_in_executor(
      pool,
      join_df,
      listing_df, text_embeddings_df, 'jumpId', 'listing_id', '_y', 'inner'
    )

  return text_embeddings_df

async def load_faiss_indexes():
  loop = asyncio.get_event_loop()
  with ThreadPoolExecutor() as pool:
    faiss_image_index = await loop.run_in_executor(
      pool,
      FaissIndex,
      None,
      None,
      None,
      faiss_index_folder / 'faiss_image_index'
    )

    faiss_text_index = await loop.run_in_executor(
      pool,
      FaissIndex,
      None,
      None,
      None,
      faiss_index_folder / 'faiss_text_index'
    )

  return faiss_image_index, faiss_text_index

async def load_listing_df():
  loop = asyncio.get_event_loop()
  if faiss_index_folder is None:
    with ThreadPoolExecutor() as pool:
      listing_df = await loop.run_in_executor(
        pool,
        pd.read_feather,
        local_project_home / 'avm_listing_info' / 'listing_df'
      )
  else:
    with ThreadPoolExecutor() as pool:
      listing_df = await loop.run_in_executor(
        pool,
        pd.read_feather,
        faiss_index_folder / 'listing_df'
      )


  return listing_df

@app.on_event("startup")
async def startup_event():
  global search_engine

  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
  image_embedder = OpenClipImageEmbeddingModel(model_name=model_name, pretrained=pretrained, device=device)
  text_embedder = OpenClipTextEmbeddingModel(embedding_model=image_embedder)

  if faiss_index_folder is None:
    # Async retrieval of embeddings
    image_embeddings_df = await load_images_dataframe()
    text_embeddings_df = await load_texts_dataframe()
    listing_df = await load_listing_df()

    if 'listing_id' not in image_embeddings_df.columns:    
      image_embeddings_df['listing_id'] = image_embeddings_df.image_name.apply(get_listingId_from_image_name)

    search_engine = ListingSearchEngine(
          image_embeddings_df=image_embeddings_df, 
          text_embeddings_df=text_embeddings_df, 
          image_embedder=image_embedder, 
          text_embedder=text_embedder,
          partition_by_province=False
          )
  else:
    faiss_image_index, faiss_text_index = await load_faiss_indexes()
    listing_df = await load_listing_df()
    search_engine = ListingSearchEngine(
          image_embedder=image_embedder, 
          text_embedder=text_embedder,
          partition_by_province=False,
          faiss_image_index=faiss_image_index, 
          faiss_text_index=faiss_text_index, 
          listing_df=listing_df
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
  propertyType: Optional[str] = None
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
  photo: Optional[str] = None

class PartialListingData(BaseModel):
  jumpId: str
  image_name: Optional[str] = None

@app.get("/listing/{listingId}")
async def get_listing(listingId: str) -> Union[ListingData, PartialListingData]:
  listing_data = search_engine.get_listing(listingId)

  if not listing_data:
      raise HTTPException(status_code=404, detail="Listing not found")
  
  if set(listing_data.keys()) == set(['listing_id', 'image_name']):
    # Replace 'listing_id' with 'jumpId'
    listing_data['jumpId'] = listing_data.pop('listing_id')
    return PartialListingData(**listing_data)
  else:
    return ListingData(**listing_data)

@app.get("/images/{listingId}")
async def read_images(listingId: str) -> List[str]:
  image_names = search_engine.get_imagenames(listingId)
  image_names = [f"{listingId}/{image_name}" for image_name in image_names]
  return image_names

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
      # image_names, scores = search_engine.image_search(image, topk=50)
      listings = search_engine.image_search(image, topk=50, group_by_listingId=True)
    except Exception as e:
      return f'search engine error: {e}'

    return listings

@app.get("/images/{listingId}/{image_name}")
async def get_image(listingId: str, image_name: str) -> FileResponse:
  image_path = local_project_home / 'deployment_listing_images' / listingId / image_name
  if image_path.is_file():
    # serve from local directory
    return FileResponse(image_path)
  elif bucket is not None:           # check and serve from GCS (if available)    
    blob = bucket.blob(f"VSS/{listingId}/{image_name}")
    if blob.exists():
      # raise HTTPException(status_code=404, detail=f"Image not found: {listingId}/{image_name}")
    
      def generate_image():
        yield blob.download_as_bytes()

      content_type = 'image/jpeg'

      return StreamingResponse(generate_image(), media_type=content_type)
    
  # If the image is not found locally or in the GCS bucket, try fetching from the jumptools URL
  try:
    # use _lg.jpg for lower resolution 
    image_name_wo_ext = Path(image_name).stem
    ext = Path(image_name).suffix
    image_name = f'{image_name_wo_ext}_lg{ext}'

    photo_url = 'https:' + str(Path(search_engine.get_listing(listingId)['photo']).parent) + '/' + image_name

    async with httpx.AsyncClient() as client:
      response = await client.get(photo_url)
      response.raise_for_status()
      return StreamingResponse(io.BytesIO(response.content), media_type='image/jpeg')
  except Exception as e:
    raise HTTPException(status_code=404, detail=f"Image not found: {listingId}/{image_name}")


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
  if provState is None and (mode == SearchMode.VSS_RERANK_ONLY or mode == SearchMode.SOFT_MATCH_AND_VSS):
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
    # image_names, scores = search_engine.text_2_image_search(phrase=query['phrase'], topk=50)
    listings = search_engine.text_2_image_search(phrase=query['phrase'], topk=50, group_by_listingId=True)
  except Exception as e:
    return f'search engine error: {e}'

  return listings

@app.post("/text-to-image-text-search/")
async def text_to_image_text_search(query: Dict[str, Any]) -> List[Dict[str, Union[str, float , List[str], str]]]:
  try:
    listings = search_engine.text_2_image_text_search(phrase=query['phrase'], topk=50)
  except Exception as e:
    return f'search engine error: {e}'
  
  return listings


@app.post("/image_to_text_search")
async def image_2_text_search(file: UploadFile = File(...)):
  image_data = await file.read()

  try:
    image = Image.open(io.BytesIO(image_data))
  except Exception as e:
    return f'error: Invalid image file'

  try:
    listings = search_engine.image_2_text_search(image, topk=50)
    print(f'len(listings): {len(listings)}')
  except Exception as e:
    return f'search engine error: {e}'

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
from typing import Dict, List, Union, Tuple, Any, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel
from realestate_vss.models.embedding import OpenClipTextEmbeddingModel, OpenClipImageEmbeddingModel
from realestate_vss.search.engine import ListingSearchEngine, SearchMode #, ScoreAggregationMethod

import realestate_core.common.class_extensions
from realestate_core.common.utils import join_df
from realestate_vision.common.utils import get_listingId_from_image_name
from realestate_vss.data.index import FaissIndex

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

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

import logging
from logging.handlers import RotatingFileHandler

logging.basicConfig(
  level=logging.INFO,
  format='%(asctime)s [%(levelname)s] [Logger: %(name)s]: %(message)s',
  datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

''' Use this for production logging
log_folder = Path("/path/to/your/log/folder")
log_filename = log_folder / "main.log"
logger.setLevel(logging.INFO)
handler = RotatingFileHandler(
    filename=str(log_filename),
    maxBytes=10_000_000,  # 10MB
    backupCount=5,
    encoding='utf-8'
)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Optionally, to prevent log messages from being propagated to the root logger
# (which might have been configured differently elsewhere)
logger.propagate = False
'''

# app = FastAPI()
# uvicorn main:app --port 8002 --reload &  # run this in terminal to start the server
# uvicorn main:app --host 0.0.0.0 --port 8002 --reload  (demo on GCP)

# Set up CORS middleware configuration
_ = load_dotenv(find_dotenv())
if "ALLOW_ORIGINS" in os.environ:
  allow_origins = os.environ["ALLOW_ORIGINS"].split(',')
else:
  raise Exception("ALLOW_ORIGINS environment variable not set in .env file")

# Global variable to hold the search engine instance
search_engine = None
datastore = None
image_embedder, text_embedder = None, None
use_process_pool = True
pool = None

# non global variables
if "LOCAL_PROJECT_HOME" in os.environ:
  local_project_home = Path(os.getenv("LOCAL_PROJECT_HOME"))
else:
  raise Exception("LOCAL_PROJECT_HOME environment variable not set in .env file")

if "USE_REDIS" in os.environ:
  use_redis = (os.getenv("USE_REDIS").lower() == 'true')
  logger.info(f'Using Redis: {use_redis}')
  if use_redis:
    REDIS_HOST = os.getenv("REDIS_HOST")
    REDIS_PORT = int(os.getenv("REDIS_PORT"))
    logger.info(f'Using Redis server: {REDIS_HOST}:{REDIS_PORT}')
else:
  use_redis = False

if use_redis:
  import redis
  from realestate_vss.data.redis_datastore import RedisDataStore

if "USE_WEAVIATE" in os.environ:
  use_weaviate = (os.getenv("USE_WEAVIATE").lower() == 'true')
  logger.info(f'Using Weaviate: {use_weaviate}')
  if use_weaviate:
    WEAVIATE_HOST = os.getenv("WEAVIATE_HOST")
    WEAVIATE_PORT = int(os.getenv("WEAVIATE_PORT")) if os.getenv("WEAVIATE_PORT") is not None else None
    if WEAVIATE_HOST is not None and WEAVIATE_PORT is not None:
      use_local_weaviate = True
      logger.info(f'Using local Weaviate server: {WEAVIATE_HOST}:{WEAVIATE_PORT}')
    else:
      use_local_weaviate = False
      WCS_URL = os.getenv("WCS_URL")
      WCS_API_KEY = os.getenv("WCS_API_KEY")
      if WCS_URL is None or WCS_API_KEY is None:
        logger.error('WCS_URL and WCS_API_KEY not found in .env')
        raise Exception("WCS_URL and WCS_API_KEY not found in .env")
      logger.info('Using Weaviate server on cloud')
else:
  use_weaviate = False

if use_weaviate:
  import weaviate
  from realestate_vss.data.weaviate_datastore import WeaviateDataStore_v4 as WeaviateDataStore
  from realestate_vss.data.weaviate_datastore import AsyncWeaviateDataStore_v4 as AsyncWeaviateDataStore

model_name = 'ViT-L-14'
pretrained = 'laion2b_s32b_b82k'

if "GCS_PROJECT_ID" in os.environ and "GCS_BUCKET_NAME" in os.environ:
  GCS_PROJECT_ID = os.getenv("GCS_PROJECT_ID")
  GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
  try:
    storage_client = storage.Client(project=GCS_PROJECT_ID)
    bucket = storage_client.bucket(GCS_BUCKET_NAME)
  except DefaultCredentialsError as e:
    logger.error(f'Error connecting to GCS: {e}')
    bucket = None
  except Exception as e:
    logger.error(f'Error connecting to GCS: {e}')
    bucket = None
else:
  bucket = None

# check if FAISS index folder is defined, if so, use FAISS index to later instantiate search engine

if "USE_FAISS" in os.environ:
  use_faiss = (os.getenv("USE_FAISS").lower() == 'true')
  logger.info(f'Using FAISS: {use_faiss}')
else:
  use_faiss = False

if use_faiss:
  if "FAISS_IMAGE_INDEX" in os.environ and "FAISS_TEXT_INDEX" in os.environ:
    faiss_image_index_path = Path(os.getenv("FAISS_IMAGE_INDEX"))
    # if not faiss_index_folder.is_dir():
    #   raise Exception("FAISS_INDEX_FOLDER is not a valid directory")
    logger.info(f'Using FAISS image index from {faiss_image_index_path}')

    faiss_text_index_path = Path(os.getenv("FAISS_TEXT_INDEX"))
    logger.info(f'Using FAISS text index from {faiss_text_index_path}')
  else:
    faiss_image_index_path = None
    faiss_text_index_path = None
    logger.info('Not using FAISS index')

  if "LISTING_DF" in os.environ:
    listing_df_path = Path(os.getenv("LISTING_DF"))
    logger.info(f'Using listing dataframe from {listing_df_path}')
  else:
    listing_df_path = None

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
      # faiss_index_folder / 'faiss_image_index'
      faiss_image_index_path
    )

    faiss_text_index = await loop.run_in_executor(
      pool,
      FaissIndex,
      None,
      None,
      None,
      # faiss_index_folder / 'faiss_text_index'
      faiss_text_index_path
    )

  return faiss_image_index, faiss_text_index

async def load_listing_df():
  loop = asyncio.get_event_loop()
  if listing_df_path is None:
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
        listing_df_path
      )

  return listing_df

def setup_weaviate():
  global image_embedder, text_embedder, datastore

  if use_local_weaviate:
    client = weaviate.connect_to_local(WEAVIATE_HOST, WEAVIATE_PORT)
  else:
    client = weaviate.connect_to_wcs(
      cluster_url=WCS_URL,
      auth_credentials=weaviate.auth.AuthApiKey(WCS_API_KEY)
    )

  datastore = WeaviateDataStore(client=client, 
                                  image_embedder=image_embedder, 
                                  text_embedder=text_embedder)
  
async def setup_async_weaviate():
  global image_embedder, text_embedder, datastore

  if use_local_weaviate:
    async_client = weaviate.use_async_with_local(WEAVIATE_HOST, WEAVIATE_PORT)
    await async_client.connect()
  else:
    async_client = weaviate.use_async_with_weaviate_cloud(
      cluster_url=WCS_URL,
      auth_credentials=weaviate.auth.AuthApiKey(WCS_API_KEY)
    )

  datastore = AsyncWeaviateDataStore(async_client=async_client, 
                                    image_embedder=image_embedder, 
                                    text_embedder=text_embedder)

def do_nothing():
  pass

def create_embedders():
  global image_embedder, text_embedder
  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')

  logger.info("Initiating image and text (clip based) embedders for process pool ...")
  image_embedder = OpenClipImageEmbeddingModel(model_name=model_name, pretrained=pretrained, device=device)
  text_embedder = OpenClipTextEmbeddingModel(embedding_model=image_embedder)

  print(f'image_embedder in create_embedders: {image_embedder}')

if use_process_pool:
  pool = ProcessPoolExecutor(max_workers=1, initializer=create_embedders)

async def startup_event():
  global image_embedder, text_embedder
  global search_engine, datastore
  global pool

  if not use_process_pool:    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')

    logger.info("Initiating image and text (clip based) embedders...")
    image_embedder = OpenClipImageEmbeddingModel(model_name=model_name, pretrained=pretrained, device=device)
    text_embedder = OpenClipTextEmbeddingModel(embedding_model=image_embedder)
  else:
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(pool, do_nothing)  # dummy to trigger call to create_embedders

  if use_redis:
    pool = redis.ConnectionPool(
                  host=REDIS_HOST,
                  port=REDIS_PORT,
                  db=0
              )
    # redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
    redis_client = redis.Redis(connection_pool=pool)
    datastore = RedisDataStore(client=redis_client, image_embedder=image_embedder, text_embedder=text_embedder)

  elif use_weaviate:
    # setup_weaviate()
    await setup_async_weaviate()
  elif use_faiss:
    if faiss_image_index_path is None:
      # Async retrieval of embeddings dataframes
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
            listing_df=listing_df,
            score_aggregation_method='max'
            )
      
  else:
    raise Exception("No datastore or search engine (using faiss) specified")
  
  logger.info('Warming up')
  warmup_text_embedding = None
  if use_process_pool:
    dummy_image = Image.new('RGB', (224, 224))
    await embed_image(dummy_image)
    warmup_text_embedding = await embed_text('dummy warmup text')

  if use_weaviate:
    try:
      
      await datastore._search_text_2_text(
        phrase='dummy warmup text' if not use_process_pool else None,
        embedding=warmup_text_embedding, 
        topk=1, 
        group_by_listingId=True, 
        include_all_fields=True, 
        **{}
      )
    except Exception as e:
      logger.warning(f'Warmup search failed: {e}')
  logger.info('Warmup done')

async def shutdown_event():
  logger.info('Shutting and calling datastore.close ...')
  global datastore
  if use_weaviate or use_redis:
    await datastore.close()

@asynccontextmanager
async def lifespan(app: FastAPI):
  await startup_event()
  try:
    yield
  finally:
    await shutdown_event()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
  CORSMiddleware,
  allow_origins=allow_origins,  # Allows all origins
  allow_credentials=True,
  allow_methods=["*"],  # Allows all methods
  allow_headers=["*"],  # Allows all headers
)


class ListingData(BaseModel):
  jumpId: Optional[str] = None
  city: Optional[str] = None    # TODO: city should not be None, we may refine this later
  provState: Optional[str] = None   # TODO: provState should not be None, we may refine this later
  postalCode: Optional[str] = None
  lat: Optional[float] = None
  lng: Optional[float] = None
  streetName: Optional[str] = None
  beds: Optional[str] = None
  bedsInt: Optional[int] = None
  baths: Optional[str] = None
  bathsInt: Optional[int] = None
  sizeInterior: Optional[str] = None
  sizeInteriorUOM: Optional[str] = None
  lotSize: Optional[str] = None
  lotUOM: Optional[str] = None
  propertyFeatures: Optional[List[str]] = None
  propertyType: Optional[str] = None
  transactionType: Optional[str] = None
  carriageTrade: Optional[bool] = None
  price: Optional[float] = None
  leasePrice: Optional[int] = None
  pool: Optional[bool] = None
  garage: Optional[bool] = None
  waterFront: Optional[bool] = None
  fireplace: Optional[bool] = None
  ac: Optional[bool] = None
  remarks: Optional[str] = None
  listing_id: str
  listingId: Optional[str] = None
  photo: Optional[str] = None
  listingDate: Optional[str] = None
  lastUpdate: Optional[str] = None
  lastPhotoUpdate: Optional[str] = None
  

class ListingSearchResult(ListingData):
  agg_score: float
  image_names: Optional[List[str]] = None
  remark_chunk_ids: Optional[List[str]] = None

class PartialListingData(BaseModel):
  jumpId: Optional[str] = None
  image_name: Optional[str] = None

def embed_image_in_process_pool(image: Image.Image) -> np.ndarray:
  global image_embedder
  return image_embedder.embed_from_single_image(image)

def embed_text_in_process_pool(phrase: str) -> np.ndarray:
  global text_embedder
  return text_embedder.embed_from_texts([phrase], 1)[0]


async def embed_image(image: Image.Image) -> List[float]:
  # global image_embedder
  # print(f'image_embedder in embed_image: {image_embedder}')

  loop = asyncio.get_event_loop()
  img_tensor_np = await loop.run_in_executor(pool, embed_image_in_process_pool, image)

  return img_tensor_np.flatten().tolist()


async def embed_text(phrase: str) -> List[float]:
  # global text_embedder
  loop = asyncio.get_event_loop()
  txt_tensor_np = await loop.run_in_executor(pool, embed_text_in_process_pool, phrase)
  # return txt_tensor_np[0].flatten().tolist()
  return txt_tensor_np.flatten().tolist()

@app.get("/listing/{listingId}")
async def get_listing(listingId: str) -> Union[ListingData, PartialListingData]:
  if use_faiss:
    listing_data = search_engine.get_listing(listingId)
  elif use_redis or use_weaviate:
    listing_data = await datastore.get_listing(listingId)

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
  if use_faiss:
    image_names = search_engine.get_imagenames(listingId)
  elif use_redis or use_weaviate:
    image_names = await datastore.get_imagenames(listingId)

  image_names = [f"{listingId}/{image_name}" for image_name in image_names]
  return image_names

@app.post("/search-by-image/")
# async def search_by_image(file: UploadFile = File(...)) -> List[Dict[str, Union[str, float , List[str]]]]:
async def search_by_image(file: UploadFile = File(...)) -> List[ListingSearchResult]:
    """
    Using the provided image file as query to the search engine, return a list of listings with images
    that are most similar to it.

    The returned list is sorted by aggregated similarity score in descending order.
    E.g.
    [
      {
        "listingId": "21125523",
        "agg_score": "0.7692008018493652",
        "image_names": [
          "21125523/21125523_4.jpg"
        ]
      },
      {
        "listingId": "20953965",
        "agg_score": "0.7561193406581879",
        "image_names": [
          "20953965/20953965_0.jpg",
          "20953965/20953965_1.jpg",
          "20953965/20953965_4.jpg",
          "20953965/20953965_9.jpg"
        ]
      },
      {
        "listingId": "20911679",
        "agg_score": "0.7544260919094086",
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
      if use_faiss:
        # image_names, scores = search_engine.image_search(image, topk=50)
        listings = search_engine.image_search(image, topk=50, group_by_listingId=True)
      elif use_redis or use_weaviate:
        if use_process_pool:
          embedding = await embed_image(image)
          listings = await datastore._search_image_2_image(embedding=embedding, topk=50, group_by_listingId=True, include_all_fields=True, **{})
        else:
          listings = await datastore._search_image_2_image(image=image, topk=50, group_by_listingId=True, include_all_fields=True, **{})
    except Exception as e:
      return f'Error: {e}'

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

    if use_faiss:
      photo_url_prefix = search_engine.get_listing(listingId)['photo']
      photo_url = 'https:' + str(Path(photo_url_prefix).parent) + '/' + image_name
    elif use_redis or use_weaviate:
      _photo_url_prefix = await datastore.get_listing(listingId)
      photo_url_prefix = _photo_url_prefix['photo']
      photo_url = 'https:' + str(Path(photo_url_prefix).parent) + '/' + image_name

    async with httpx.AsyncClient() as client:
      response = await client.get(photo_url)
      response.raise_for_status()
      return StreamingResponse(io.BytesIO(response.content), media_type='image/jpeg')
  except Exception as e:
    logger.error(e)
    raise HTTPException(status_code=404, detail=f"Image not found: {listingId}/{image_name}")

def cleanup_query(query: str) -> str:
  """
    Clean up query for redis, weaviate, etc. 
    Empty value are removed, numeric stuff are expanded as range search, etc.

    Parameters:
    query (dict): The query to be cleaned up.

    Returns:
    dict: The cleaned up query.
  """
  for key in list(query.keys()):
      val = query[key]
      if val is None or val == '':
        del query[key]
      elif isinstance(val, list) or isinstance(val, tuple):
        if val[0] is None and val[1] is None:
          del query[key]
      elif isinstance(val, int) or isinstance(val, float):  # NumericField
        # somehow, only range search works, so expand x -> [x x]
        query[key] = [val, val]
      elif isinstance(val, str):
        try:
          num_val = float(val)
          query[key] = [num_val, num_val]
        except ValueError:
          pass
  return query

@app.post("/search-by-text/")
async def search_by_text(query: Dict[str, Union[str, Optional[int], Optional[List[Optional[int]]]]], 
                         mode: SearchMode = Query(SearchMode.VSS_ONLY), 
                         lambda_val: float = Query(0.8, description="Required if mode is SOFT_MATCH_AND_VSS."), 
                         alpha_val: float = Query(0.5, description="Required if mode is SOFT_MATCH_AND_VSS.")
):
  """
  Old
  """
  if search_engine is not None:
  
    phrase = query.get('phrase', None)
    if phrase is not None:
      del query['phrase']   # remove key phrase from query after extracting it

    provState = query.get('provState', None)
    
    logger.info(f'query: {query}')
    if provState is None and (mode == SearchMode.VSS_RERANK_ONLY or mode == SearchMode.SOFT_MATCH_AND_VSS):
      raise HTTPException(status_code=404, detail="provState must be provided")

    if mode == SearchMode.VSS_ONLY or mode == SearchMode.VSS_RERANK_ONLY:
      results = search_engine.text_search(mode, topk=20, phrase=phrase, return_df=False, **query)
    elif mode == SearchMode.SOFT_MATCH_AND_VSS:
      results = search_engine.text_search(mode, topk=20, return_df=False, lambda_val=lambda_val, alpha_val=alpha_val, phrase=phrase, **query)
    else:
      raise HTTPException(status_code=404, detail="Invalid search mode")
    
  elif use_redis:
    phrase = query.get('phrase', None)
    if phrase is not None:
      del query['phrase']   # remove key phrase from query after extracting it
   
    # clean up query to conform to valid redis query
    logger.info(f'before cleanup: {query}')
    query = cleanup_query(query)
    logging.info(f'after cleanup: {query}')

    results = datastore._search_text_2_text(phrase=phrase, topk=50, group_by_listingId=True, include_all_fields=True, **query)
    # add "listing_id" key to each result
    for result in results:
      result['listing_id'] = result['listingId']
      
  return results

@app.post("/text-to-text-search/")
async def text_to_text_search(query: Dict[str, Any]) -> List[ListingSearchResult]:
  phrase = query.get('phrase', None)
  if phrase is not None:
    del query['phrase']   # remove key phrase from query after extracting it

  if use_faiss:
    listings = search_engine._search_text_2_text(phrase=query['phrase'], topk=50, group_by_listingId=False, include_all_fields=False, **query)
  elif use_redis or use_weaviate:
    logger.info(f'before cleanup: {query}')
    query = cleanup_query(query)
    logger.info(f'after cleanup: {query}')

    if use_process_pool:
      embedding = await embed_text(phrase)
      listings = await datastore._search_text_2_text(embedding=embedding, topk=50, group_by_listingId=True, include_all_fields=True, **query)
    else:
      listings = await datastore._search_text_2_text(phrase=phrase, topk=50, group_by_listingId=True, include_all_fields=True, **query)
  
  for listing in listings:
    listing['listing_id'] = listing['listingId']

  return listings


@app.post("/text-to-image-search/")
# async def text_to_image_search(query: Dict[str, Any]) -> List[Dict[str, Union[str, float , List[str]]]]:
async def text_to_image_search(query: Dict[str, Any]) -> List[ListingSearchResult]:
  phrase = query.get('phrase', None)
  if phrase is not None:
    del query['phrase']   # remove key phrase from query after extracting it

  if use_faiss:
    try:
      # image_names, scores = search_engine.text_2_image_search(phrase=query['phrase'], topk=50)
      # listings = search_engine.text_2_image_search(phrase=query['phrase'], topk=50, group_by_listingId=True)
      listings = search_engine._search_text_2_image(phrase=phrase, topk=50, group_by_listingId=True, include_all_fields=True, **query)
    except Exception as e:
      return f'search engine error: {e}'
  elif use_redis or use_weaviate:    
    logger.info(f'before cleanup: {query}')
    query = cleanup_query(query)
    logger.info(f'after cleanup: {query}')

    try:
      if use_process_pool:
        embedding = await embed_text(phrase)
        listings = await datastore._search_text_2_image(embedding=embedding, topk=50, group_by_listingId=True, include_all_fields=True, **query)
      else:
        listings = await datastore._search_text_2_image(phrase=phrase, topk=50, group_by_listingId=True, include_all_fields=True, **query)
    except Exception as e:
      return f'search engine error: {e}'

  return listings

'''
@app.post("/text-to-image-text-search/")
async def text_to_image_text_search(query: Dict[str, Any]) -> List[Dict[str, Union[str, float , List[str], str]]]:
  
  try:
    if search_engine is not None:
      listings = search_engine.text_2_image_text_search(phrase=query['phrase'], topk=50)
    elif use_redis:
      phrase = query.get('phrase', None)
      if phrase is not None:
        del query['phrase']

      listings = datastore.search(phrase=phrase, topk=50, group_by_listingId=True, **query)

  except Exception as e:
    return f'Error: {e}'
  
  return listings
'''

@app.post("/image-to-text-search")
async def image_2_text_search(file: UploadFile = File(...)) -> List[ListingSearchResult]:
  
  image_data = await file.read()

  try:
    image = Image.open(io.BytesIO(image_data))
  except Exception as e:
    return f'error: Invalid image file'

  try:
    if use_faiss:
      # listings = search_engine.image_2_text_search(image, topk=50)
      listings = search_engine._search_image_2_text(image=image, topk=50, group_by_listingId=True, include_all_fields=True, **{})

    elif use_redis or use_weaviate:
      if use_process_pool:
        embedding = await embed_image(image)
        listings = await datastore._search_image_2_text(embedding=embedding, topk=50, group_by_listingId=True, include_all_fields=True, **{})
      else:
        listings = await datastore._search_image_2_text(image=image, topk=50, group_by_listingId=True, include_all_fields=True, **{})
  except Exception as e:
    return f'Error: {e}'

  return listings

'''
@app.post("/image-to-image-text-search")
async def image_2_image_text_search(file: UploadFile = File(...)):

  image_data = await file.read()

  try:
    image = Image.open(io.BytesIO(image_data))
  except Exception as e:
    return f'error: Invalid image file'

  try:
    if search_engine is not None:
      listings = search_engine.image_2_image_text_search(image, topk=50)
    elif use_redis:
      listings = datastore.search(image=image, topk=50, group_by_listingId=True, **{})
  except Exception as e:
    return f'search engine error: {e}'

  return listings
'''

'''
@app.post("/many-image-search")
async def many_image_search(files: List[UploadFile] = File(...)) -> List[Dict[str, Union[str, float , List[str]]]]:
  """
  Using the provided image files as query to the search engine, return a list of listings with images
  that are most similar to it.

  The returned list is sorted by aggregated similarity score in descending order.
  """
  if search_engine is None:
    raise HTTPException(status_code=503, detail="Search engine not initialized or not configured for use.")
  
  images = []
  for file in files:
    image_data = await file.read()

    try:
      image = Image.open(io.BytesIO(image_data))
      images.append(image)
    except Exception as e:
      return f'error: Invalid image file {file.filename}'

  try:
    listings = search_engine.many_image_search(images, topk=50, group_by_listingId=True)
  except Exception as e:
    return f'search engine error: {e}'

  return listings
'''

# class QueryModel(BaseModel):
#   query: dict

@app.post("/multi-image-search")
async def multi_image_search(query_body: Optional[str] = Form(None), files: List[UploadFile] = File(...)) -> List[ListingSearchResult]:
  
  images: List[Image.Image] = []
  for file in files:
    image_data = await file.read()
    try:
      image = Image.open(io.BytesIO(image_data))
      images.append(image)
    except Exception as e:
      return f'error: Invalid image file {file.filename}'
    
  if query_body is not None:
    try:
      query = json.loads(query_body)
      logger.info(f'before cleanup: {query}')

      query = cleanup_query(query)
      logger.info(f'after cleanup: {query}')
    except json.JSONDecodeError:
      return {"error": f"Invalid JSON format in query_body {query_body}"}
    
    phrase = query.get('phrase', None)
    logger.info(f'phrase: {phrase}')
    if phrase is not None:
      del query['phrase']   # remove key phrase from query after extracting it
  else:
    phrase = None
    query = {}

  try:
    if use_faiss:
      listings = search_engine.multi_image_search(images, phrase=phrase, topk=50, group_by_listingId=True, include_all_fields=True, **query)
    elif use_redis or use_weaviate:
      if use_process_pool:
        embeddings = await asyncio.gather(*[embed_image(image) for image in images])
        image_embedding = np.mean(embeddings, axis=0).tolist()

        text_embedding = await embed_text(phrase) if phrase is not None else None

        listings = await datastore.multi_image_search(image_embedding=image_embedding, 
                                                      text_embedding=text_embedding, 
                                                      topk=50, 
                                                      group_by_listingId=True, 
                                                      include_all_fields=True, 
                                                      **query)
      else:
        listings = await datastore.multi_image_search(images, phrase=phrase, topk=50, group_by_listingId=True, include_all_fields=True, **query)
  except Exception as e:
    return f'Error: {e}'
  
  return listings

@app.post("/search")
# async def search(file: Optional[UploadFile] = None, query: Optional[Dict[str, Any]] = None):
# async def search(query_body: Optional[str] = Form(None), file: Optional[UploadFile] = File(None)):
async def search(query_body: Optional[str] = Form(None), file: UploadFile = None) -> List[ListingSearchResult]:
  """
  One search to rule them all (full cross modality)

  Parameters:
  file (UploadFile): The image file to search by image.
  query_body (dict): The query to search by text in format of {"phrase": "<search term>"}

  """
  
  image = None
  if file is not None:
    image_data = await file.read()
    try:
      image = Image.open(io.BytesIO(image_data))
    except Exception as e:
      image = None
      return f'error: Invalid image file {file.filename}'
    
  if query_body is not None:
    try:
      query = json.loads(query_body)
      logger.info(f'before cleanup: {query}')
      query = cleanup_query(query)
      logger.info(f'after cleanup: {query}')
    except json.JSONDecodeError:
      return {"error": f"Invalid JSON format in query_body {query_body}."}
    
    phrase = query.get('phrase', None)
    logger.info(f'phrase: {phrase}')
    if phrase is not None:
      del query['phrase']   # remove key phrase from query after extracting it
  else:
    phrase = None
    query = {}
    
  # try:
  if use_faiss:
    listings = search_engine.search(image=image, phrase=phrase, topk=50, group_by_listingId=True, include_all_fields=True, **query)
  elif use_redis or use_weaviate:
    logger.info(f'phrase: {phrase}')
    logger.info(f'query: {query}')

    if use_process_pool:
      image_embedding = await embed_image(image) if image is not None else None
      text_embedding = await embed_text(phrase) if phrase is not None else None
      listings = await datastore.search(image_embedding=image_embedding, 
                                        text_embedding=text_embedding, 
                                        topk=50, 
                                        group_by_listingId=True, 
                                        include_all_fields=True, 
                                        **query)
    else:
      listings = await datastore.search(image=image, phrase=phrase, topk=50, group_by_listingId=True, include_all_fields=True, **query)

  # except Exception as e:
  #   return f'Error: {e}'
  
  return listings


@app.get("/health")
async def health_check():
  if use_weaviate:
    is_connected = await datastore.ping()
    if not is_connected:
      raise HTTPException(status_code=503, detail="Weaviate connection lost")
  return {"status": "healthy"}

# ---------------------------------------------------------------------------------------------------------------
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
  if search_engine is None:
    raise HTTPException(status_code=503, detail="Search engine not initialized or not configured for use.")
  
  # global image_embeddings_df, text_embeddings_df
  n_images = search_engine.image_embeddings_df.shape[0]
  n_listings = search_engine.image_embeddings_df.listing_id.nunique()

  n_remarks = search_engine.text_embeddings_df.shape[0]
  
  message = f'Searching {n_images} images in {n_listings} listings\n'
  message += f'Searching remarks for {n_remarks} listings'

  return message
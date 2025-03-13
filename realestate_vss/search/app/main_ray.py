from typing import Dict, List, Union, Tuple, Any, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel
import ray
from ray import serve
import httpx
import asyncio, io, json, os
import torch
import numpy as np
from pathlib import Path
import pandas as pd
from PIL import Image
from io import BytesIO

from dotenv import load_dotenv, find_dotenv

from google.cloud import storage
from google.auth.exceptions import DefaultCredentialsError

import weaviate
from realestate_vss.data.weaviate_datastore import WeaviateDataStore_v4 as WeaviateDataStore
from realestate_vss.data.weaviate_datastore import AsyncWeaviateDataStore_v4 as AsyncWeaviateDataStore

import logging
from logging.handlers import RotatingFileHandler

# Setup Ray and Ray Serve
ray.init(ignore_reinit_error=True)
serve.start(http_options={"port": 8005})  # Different port from FastAPI

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
# uvicorn main_ray:app --port 8002 --reload &  # run this in terminal to start the server
# uvicorn main_ray:app --host 0.0.0.0 --port 8002 --reload  (demo on GCP)

# Set up CORS middleware configuration
_ = load_dotenv(find_dotenv())
if "ALLOW_ORIGINS" in os.environ:
  allow_origins = os.environ["ALLOW_ORIGINS"].split(',')
else:
  raise Exception("ALLOW_ORIGINS environment variable not set in .env file")

if "LOCAL_PROJECT_HOME" in os.environ:
  local_project_home = Path(os.getenv("LOCAL_PROJECT_HOME"))
else:
  raise Exception("LOCAL_PROJECT_HOME environment variable not set in .env file")

# Weaviate configuration

use_weaviate = (os.getenv("USE_WEAVIATE").lower() == 'true')
logger.info(f'Using Weaviate: {use_weaviate}')
assert use_weaviate, "Weaviate is required for this application"

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



# GCS configuration
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

model_name = 'ViT-L-14'
pretrained = 'laion2b_s32b_b82k'

# Determine whether to use a GPU automatically based on PyTorch's CUDA availability
num_gpus = 1 if torch.cuda.is_available() else 0   # on a mac, its 0 since its mps

# Ray Serve deployment for model inference
@serve.deployment(num_replicas=1, ray_actor_options={"num_gpus": num_gpus})
class OpenCLIPModelServer:
  def __init__(self):
    from realestate_vss.models.embedding import OpenClipImageEmbeddingModel, OpenClipTextEmbeddingModel
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    logger.info(f"Using device: {device}")
    
    self.image_embedder = OpenClipImageEmbeddingModel(model_name=model_name, pretrained=pretrained, device=device)
    self.text_embedder = OpenClipTextEmbeddingModel(embedding_model=self.image_embedder)
    logger.info("Models loaded successfully")

  def embed_image(self, image_bytes: bytes) -> List[float]:
    try:
      # Open and convert the image
      image = Image.open(BytesIO(image_bytes)).convert("RGB")
      
      # Use the embedder to get embeddings
      embedding = self.image_embedder.embed_from_single_image(image)
      
      # Return the embedding as a list
      return embedding.flatten().tolist()
    except Exception as e:
      logger.error(f"Error embedding image: {e}")
      raise ValueError(f"Failed to embed image: {str(e)}")
    
  def embed_text(self, text: str) -> List[float]:
    try:
      # Use the text embedder to get embeddings
      embedding = self.text_embedder.embed_from_texts([text], 1)[0]
      
      # Return the embedding as a list
      return embedding.flatten().tolist()
    except Exception as e:
      logger.error(f"Error embedding text: {e}")
      raise ValueError(f"Failed to embed text: {str(e)}")
      
  async def __call__(self, request):
    try:
      if isinstance(request, dict):
        data = request
      else:
        data = await request.json()
      
      if "type" not in data:
        return {"error": "Missing 'type' field. Must be 'image' or 'text'"}
      
      if data["type"] == "image":
        if "image_bytes" not in data:
          return {"error": "Missing 'image_bytes' field for image embedding"}
        
        # Decode base64 if needed
        import base64
        image_bytes = base64.b64decode(data["image_bytes"])
        
        embedding = self.embed_image(image_bytes)
        return {"embedding": embedding}
        
      elif data["type"] == "text":
        if "text" not in data:
          return {"error": "Missing 'text' field for text embedding"}
          
        embedding = self.embed_text(data["text"])
        return {"embedding": embedding}
        
      else:
        return {"error": f"Unknown type: {data['type']}. Must be 'image' or 'text'"}
    
    except Exception as e:
      logger.error(f"Error processing request: {e}")
      return {"error": str(e)}
    
# Deploy Ray serve model
clip_server = OpenCLIPModelServer.bind()
serve.run(clip_server)

# Get a handle to the deployment
handle = serve.get_app_handle("default")

# Create global variable for datastore
datastore = None

class ListingData(BaseModel):
  jumpId: Optional[str] = None
  city: Optional[str] = None
  provState: Optional[str] = None
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
  remark_chunk_pos: Optional[List[List[int]]] = None

class PartialListingData(BaseModel):
  jumpId: Optional[str] = None
  image_name: Optional[str] = None

async def setup_async_weaviate():
  global datastore
  
  if use_local_weaviate:
    async_client = weaviate.use_async_with_local(WEAVIATE_HOST, WEAVIATE_PORT)    
  else:
    async_client = weaviate.use_async_with_weaviate_cloud(
      cluster_url=WCS_URL,
      auth_credentials=weaviate.auth.AuthApiKey(WCS_API_KEY)
    )

  await async_client.connect()

  datastore = AsyncWeaviateDataStore(async_client=async_client, image_embedder=None, text_embedder=None)
  return datastore

async def startup_event():
  global datastore
  await setup_async_weaviate()

  # Warmup
  logger.info('Warming up')
  try:
    # Send warmup requests to Ray
    warmup_text = "dummy warmup text"
    warmup_response = await handle.remote({"type": "text", "text": warmup_text})
    
    # Create a small dummy image
    dummy_image = Image.new('RGB', (224, 224))
    buffer = BytesIO()
    dummy_image.save(buffer, format="JPEG")
    import base64
    image_bytes_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    image_warmup_response = await handle.remote({"type": "image", "image_bytes": image_bytes_b64})
    
    # Also warm up the datastore
    await datastore._search_text_2_text(
      embedding=warmup_response["embedding"],
      topk=1,
      group_by_listingId=True,
      include_all_fields=True,
      **{}
    )
  except Exception as e:
    logger.warning(f'Warmup search failed: {e}')
  
  logger.info('Warmup done')


async def shutdown_event():
  logger.info('Shutting down and calling datastore.close ...')
  global datastore
  await datastore.close()
  
  # Shutdown Ray and Ray Serve
  serve.shutdown()
  ray.shutdown()


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
  allow_origins=allow_origins,
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

# Helper function to cleanup API query parameters
def cleanup_query(query: dict) -> dict:
  """
  Clean up query for weaviate.
  Empty values are removed, numeric stuff are expanded as range search, etc.
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

# API endpoints
@app.get("/listing/{listingId}")
async def get_listing(listingId: str) -> Union[ListingData, PartialListingData]:
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
  image_names = await datastore.get_imagenames(listingId)
  image_names = [f"{listingId}/{image_name}" for image_name in image_names]
  return image_names


@app.get("/images/{listingId}/{image_name}")
async def get_image(listingId: str, image_name: str) -> FileResponse:
  """
  Endpoint to retrieve an image for a given listing.
  This function attempts to retrieve an image from three possible sources:
  1. Local directory
  2. Google Cloud Storage (GCS) bucket
  3. External jumptools URL
  Args:
    listingId (str): The ID of the listing.
    image_name (str): The name of the image file.
  Returns:
    FileResponse: The image file response if found locally.
    StreamingResponse: The image file response if found in GCS or fetched from the jumptools URL.
  Raises:
    HTTPException: If the image is not found in any of the sources.
  """
  image_path = local_project_home / 'deployment_listing_images' / listingId / image_name
  if image_path.is_file():
    # serve from local directory
    return FileResponse(image_path)
  elif bucket is not None:           # check and serve from GCS (if available)    
    blob = bucket.blob(f"VSS/{listingId}/{image_name}")
    if blob.exists():
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

# Helper function to encode image for Ray Serve
async def prepare_image_for_ray(image: Image.Image):
  # Resize image to optimize transfer to Ray
  width, height = image.size
  # Compute new dimensions: shortest side = 224 (preserving aspect ratio)
  if width < height:
    new_width = 224
    new_height = int((height / width) * 224)
  else:
    new_height = 224
    new_width = int((width / height) * 224)

  # Resize using LANCZOS filter for high-quality downsampling
  resized_image = image.resize((new_width, new_height), Image.LANCZOS)

  buffer = BytesIO()
  resized_image.save(buffer, format="JPEG", quality=95)
  buffer.seek(0)

  import base64
  return base64.b64encode(buffer.getvalue()).decode('utf-8')


@app.post("/multi-image-search")
async def multi_image_search(query_body: Optional[str] = Form(None), files: List[UploadFile] = File(...)) -> List[ListingSearchResult]:
  images: List[Image.Image] = []
  for file in files:
    image_data = await file.read()
    try:
      image = Image.open(BytesIO(image_data))
      images.append(image)
    except Exception as e:
      raise HTTPException(status_code=400, detail=f"Invalid image file: {file.filename}")
    
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
    # Get embeddings for each image from Ray
    image_embeddings = []
    for image in images:
      image_bytes_b64 = await prepare_image_for_ray(image)
      response = await handle.remote({"type": "image", "image_bytes": image_bytes_b64})
      
      if "error" in response:
        raise Exception(f"Ray Serve error: {response['error']}")
      
      image_embeddings.append(response["embedding"])
    
    # Average the embeddings
    image_embedding = np.mean(image_embeddings, axis=0).tolist()
    
    # Get text embedding if phrase exists
    text_embedding = None
    if phrase is not None:
      text_response = await handle.remote({"type": "text", "text": phrase})
      
      if "error" in text_response:
        raise Exception(f"Ray Serve error: {text_response['error']}")
      
      text_embedding = text_response["embedding"]
    
    listings = await datastore.multi_image_search(
      image_embedding=image_embedding, 
      text_embedding=text_embedding, 
      topk=50, 
      group_by_listingId=True, 
      include_all_fields=True, 
      **query
    )
  except Exception as e:
    return f'Error: {e}'
  
  return listings


@app.post("/search")
async def search(query_body: Optional[str] = Form(None), file: UploadFile = None) -> List[ListingSearchResult]:
  """
  One search to rule them all (full cross modality)
  """
  image = None
  image_embedding = None
  if file is not None:
    image_data = await file.read()
    try:
      image = Image.open(BytesIO(image_data))
      
      # Get embedding from Ray Serve for the image
      image_bytes_b64 = await prepare_image_for_ray(image)
      response = await handle.remote({"type": "image", "image_bytes": image_bytes_b64})
      
      if "error" in response:
        raise Exception(f"Ray Serve error: {response['error']}")
      
      image_embedding = response["embedding"]
    except Exception as e:
      image = None
      raise HTTPException(status_code=400, detail=f"Invalid image file: {file.filename}")
    
  text_embedding = None
  if query_body is not None:
    try:
      query = json.loads(query_body)
      logger.info(f'before cleanup: {query}')
      query = cleanup_query(query)
      logger.info(f'after cleanup: {query}')
      
      phrase = query.get('phrase', None)
      logger.info(f'phrase: {phrase}')
      
      if phrase is not None:
        # Get embedding from Ray Serve for the text
        text_response = await handle.remote({"type": "text", "text": phrase})
        
        if "error" in text_response:
          raise Exception(f"Ray Serve error: {text_response['error']}")
        
        text_embedding = text_response["embedding"]
        del query['phrase']
        
    except json.JSONDecodeError:
      return {"error": f"Invalid JSON format in query_body {query_body}."}
  else:
    query = {}

  try:
    listings = await datastore.search(
      image_embedding=image_embedding, 
      text_embedding=text_embedding, 
      topk=50, 
      group_by_listingId=True, 
      include_all_fields=True, 
      **query
    )
  except Exception as e:
    return f'Error: {e}'
  
  return listings


@app.get("/health")
async def health_check():
  is_connected = await datastore.ping()
  if not is_connected:
    raise HTTPException(status_code=503, detail="Weaviate connection lost")
  return {"status": "healthy"}

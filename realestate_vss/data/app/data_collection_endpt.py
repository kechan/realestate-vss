from typing import Optional, List, Dict, Any, Tuple
from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException, Body, Query
from fastapi.responses import JSONResponse, PlainTextResponse
from pathlib import Path
import shutil, os, gzip, json
from datetime import datetime
import uvicorn

import pandas as pd

import weaviate
from weaviate.classes.init import AdditionalConfig, Timeout
from realestate_vss.data.weaviate_datastore import WeaviateDataStore_v4 as WeaviateDataStore

from dotenv import load_dotenv, find_dotenv

from celery.result import AsyncResult
from celery_unstack import unstack
# from celery_embed import embed_listings, embed_listings_from_avm, remove_all_embed_listings_task_ids
# from celery_update_embeddings import update_embeddings, update_inactive_embeddings
from celery_embed_index import embed_and_index_task
from celery_delete_inactive import delete_inactive_listings_task

import logging

# use this to establish a public endpt for image tagging service pipeline to upload image to
# ./ngrok http 8000 

# Use this for loca.lt
# echo "weak-loops-prove.loca.lt" > non_ml_host
# gsutil cp non_ml_host gs://ai-tests/tmp 

# uvicorn data_collection_endpt:app --port 8001 --reload

listing_fields = ['jumpId', 'city', 'provState', 'postalCode', 'lat', 'lng', 'streetName',
                  'beds', 'bedsInt', 'baths', 'bathsInt', 'sizeInterior',
                  'sizeInteriorUOM', 'lotSize', 'lotUOM', 'propertyFeatures',
                  'propertyType', 'transactionType', 'carriageTrade', 'price',
                  'leasePrice', 'pool', 'garage', 'waterFront', 'fireplace', 'ac',
                  'remarks', 'photo', 'listingDate', 'lastUpdate', 'lastPhotoUpdate']


class UnsyncFileManager:
  def __init__(self, 
    unsync_folder: Path,
    es_host: str,
    es_port: int,
    es_index: str,
    expiration_days: int = 30
  ):
    self.logger = logging.getLogger(__name__)
    self.unsync_folder = unsync_folder
    self.expiration_days = expiration_days

    from realestate_vss.data.es_client import ESClient
    self.es_client = ESClient(
      host=es_host,
      port=es_port,
      index_name=es_index,
      fields=['jumpId']
    )

  def get_expired_files(self) -> List[Tuple[Path, datetime]]:
    """Find files older than expiration_days."""
    expired_files = []
    for file in self.unsync_folder.glob("unsync_image_embeddings_df.*"):
      try:
        # Extract timestamp from filename
        timestamp_str = file.name.split('.')[-1]
        timestamp = datetime.strptime(timestamp_str, "%Y%m%d%H%M")
        
        # Check if file is expired
        age_days = (datetime.now() - timestamp).days
        if age_days > self.expiration_days:
          expired_files.append((file, timestamp))
      except Exception as e:
        self.logger.warning(f"Error processing file {file}: {e}")
    
    return expired_files

  def cleanup_expired_files(self) -> List[Dict]:
    """
    Wrapper method that ensures ES client cleanup.
    """
    try:
      return self._do_cleanup()
    finally:
      if hasattr(self, 'es_client'):
        self.es_client.close()

  def _do_cleanup(self) -> List[Dict]:
    """
    Main cleanup implementation
    
    Clean up expired unsync files if none of their listings exist in ES.
    """
    expired_files = self.get_expired_files()
    results = []
    
    for file_path, timestamp in expired_files:
      try:
        # Read the file and get unique listings
        df = pd.read_feather(file_path)
        unique_listings = df['listing_id'].unique().tolist()
        
        # Check if any listings exist in ES
        existing_listings = self.es_client.get_active_listings(unique_listings)
        
        if not existing_listings:  # No listings exist in ES
          size_mb = file_path.stat().st_size / (1024 * 1024)
          age_days = (datetime.now() - timestamp).days
          
          file_path.unlink()
          
          results.append({
            'filename': file_path.name,
            'age_days': age_days,
            'size_mb': round(size_mb, 2),
            'total_listings': len(unique_listings),
            'listings': unique_listings,
            'action': 'deleted',
            'reason': f"Age: {age_days} days, no listings found in ES"
          })
        else:
          results.append({
            'filename': file_path.name,
            'action': 'preserved',
            'reason': f"Found {len(existing_listings)} active listings in ES"
          })
          
      except Exception as e:
        self.logger.error(f"Error processing file {file_path}: {e}")
        results.append({
          'filename': file_path.name,
          'action': 'error',
          'error': str(e)
        })
    
    return results

app = FastAPI()

_ = load_dotenv(find_dotenv())
if "IMG_CACHE_FOLDER" in os.environ:
  img_cache_folder = Path(os.environ["IMG_CACHE_FOLDER"])
  print(f"img_cache_folder: {img_cache_folder}")
else:
  raise ValueError("IMG_CACHE_FOLDER not found in .env")

@app.get("/")
async def root():
  return PlainTextResponse("ok")

@app.get("/weaviate/count")
async def get_weaviate_doc_count():
  load_dotenv(find_dotenv())
  
  WEAVIATE_HOST = os.getenv("WEAVIATE_HOST")
  WEAVIATE_PORT = int(os.getenv("WEAVIATE_PORT")) if os.getenv("WEAVIATE_PORT") is not None else None
  WCS_URL = os.getenv("WCS_URL")
  WCS_API_KEY = os.getenv("WCS_API_KEY")
  
  if WEAVIATE_HOST and WEAVIATE_PORT:
    client = weaviate.connect_to_local(
        host=WEAVIATE_HOST, 
        port=WEAVIATE_PORT
    )
  elif WCS_URL and WCS_API_KEY:
    client = weaviate.connect_to_wcs(
        cluster_url=WCS_URL,
        auth_credentials=weaviate.auth.AuthApiKey(WCS_API_KEY)
    )
  else:
    raise HTTPException(status_code=500, detail="Weaviate configuration not found")

  datastore = WeaviateDataStore(client=client, image_embedder=None, text_embedder=None)
  
  try:
      count = datastore.count_all()
      return JSONResponse(content={"total_docs": count})
  except Exception as e:
      raise HTTPException(status_code=500, detail=str(e))
  finally:
      datastore.close()

# Endpoint for submitting images and metadata
@app.post("/submit/")
async def submit(
      file: UploadFile = File(...),
      # timestamp: str = Form(...),
      listingId: str = Form(...),
      comma_sep_image_id: str = Form(None),
      comma_sep_photo_id: str = Form(None),
      comma_sep_aspect_ratio: str = Form(None),
      remarks: str = Form(None)
  ):

  # Create listing directory if it does not exist
  listing_path = img_cache_folder / listingId
  listing_path.mkdir(exist_ok=True)

  # Save the file
  img_filename = listing_path / file.filename
  with img_filename.open("wb") as buffer:
    file.file.seek(0)
    shutil.copyfileobj(file.file, buffer)

  unstack.apply_async(args=[str(img_cache_folder/listingId), 
                            comma_sep_image_id, 
                            comma_sep_photo_id, 
                            comma_sep_aspect_ratio, 
                            remarks], 
                            queue='unstack_queue')


  # Respond with status of the upload
  return JSONResponse(content={"message": f"file {file.filename} saved with metadata."})

''' Not needed for now
@app.post("/submit_listing_jsons/")
async def submit_listing_jsons(body: bytes = Body(...), start_date: Optional[str] = None, end_date: Optional[str] = None):
  """
  Submission of listing jsons in gzip format. For now, they come from data dump from the AVM monitoring service.
  """
  
  try:
    decompressed_data = gzip.decompress(body)
    json_data: List[Dict[str, Any]] = json.loads(decompressed_data)

    # print(type(json_data[0]['propertyFeatures']))
    # start_date = '2024-03-25'
    # end_date = None
    print(f"start_date: {start_date}, end_date: {end_date}")
    embed_listings_from_avm.apply_async(args=[str(img_cache_folder), json_data, start_date, end_date], queue='embed_queue')
      
    return {"message": "Data processed successfully", "received_records": len(json_data)}
  except Exception as e:
    raise HTTPException(status_code=400, detail=f"Error processing data: {str(e)}")
'''

''' Not needed for now
@app.get("/remove_all_embed_listings_task_id")
async def remove_all_embed_listings_task_id():
  remove_all_embed_listings_task_ids.apply_async(args=[], queue='embed_queue')

  return JSONResponse(content={"message": "Removing all embed listings task ids from Redis."})
''' 

''' we will do /embed and /update_vec_index in one step by /embed_and_index
@app.get("/embed")
async def embed():
  # embed all the images under img_cache_folder
  # embed_images.delay(img_cache_folder=str(img_cache_folder))
  # Inside the embed endpoint
  embed_listings.apply_async(args=[str(img_cache_folder), listing_fields], queue='embed_queue')

  return JSONResponse(content={"message": "Embedding images."})

@app.get("/update_vec_index")
async def update_vec_index():
  """
  Consolidate all embeddings in img_cache_folder and update the vector index for the search service.
  """
  update_embeddings.apply_async(args=[str(img_cache_folder)], queue='update_embed_queue')

  return JSONResponse(content={"message": "Updating vector index."})
'''

@app.get("/embed_and_index")
async def embed_and_index(image_batch_size: int = Query(32), text_batch_size: int = Query(128), num_workers: int = Query(4)):
  task = embed_and_index_task.apply_async(args=[str(img_cache_folder), 
                                                listing_fields, 
                                                image_batch_size, 
                                                text_batch_size, 
                                                num_workers
                                                ], queue='embed_index_queue')
  
  return JSONResponse(content={"message": "Embedding and indexing task started.", "task_id": task.id})

''' Not needed
@app.get("/update_inactive_vec_index")
async def update_inactive_vec_index():
  """
  Consolidate all embeddings in img_cache_folder and update the vector index for the search service.
  """
  update_inactive_embeddings.apply_async(args=[str(img_cache_folder)], queue='update_embed_queue')

  return JSONResponse(content={"message": "Updating inactive vector index."})
'''

@app.get("/delete_inactive")
async def delete_inactive(batch_size: int = Query(20), sleep_time: float = Query(0.5)):
  """
  Endpoint to trigger deletion of inactive listings from Weaviate.
  This task will delete listings that have been marked as inactive, delisted, or sold
  based on BigQuery data since the last run.
  
  Args:
    batch_size: Number of listings to delete in each batch
    sleep_time: Sleep time between batches in seconds
  """
  task = delete_inactive_listings_task.apply_async(
    args=[str(img_cache_folder), batch_size, sleep_time],
    queue='delete_inactive_queue'
  )
  
  return JSONResponse(content={
    "message": "Inactive listings deletion task started.",
    "task_id": task.id
  })


@app.get("/embed_and_index_task_status/{task_id}")
async def get_embed_and_index_task_status(task_id: str):
  task_result = embed_and_index_task.AsyncResult(task_id)
  response = {
      "task_id": task_id,
      "status": task_result.status,
      "result": task_result.result if task_result.ready() else None
  }
  return JSONResponse(content=response)

@app.get("/delete_inactive_listings_task_status/{task_id}")
async def get_delete_inactive_listings_task_status(task_id: str):
  task_result = delete_inactive_listings_task.AsyncResult(task_id)
  response = {
      "task_id": task_id,
      "status": task_result.status,
      "result": task_result.result if task_result.ready() else None
  }
  return JSONResponse(content=response)

# Add the new endpoint
@app.get("/delete_old_unsync")
async def delete_old_unsync(expiration_days: int = Query(30)):
  """
  Endpoint to cleanup old unsync files.
  
  Args:
    dry_run: If True, only simulate deletion
    expiration_days: Number of days after which files are considered expired
  
  Returns:
    JSON response with cleanup results
  """
  try:
    # Get ES configuration from environment
    es_host = os.getenv("ES_HOST")
    es_port = int(os.getenv("ES_PORT"))
    es_index = os.getenv("ES_LISTING_INDEX_NAME")
    
    if not all([es_host, es_port, es_index]):
      raise HTTPException(
        status_code=500,
        detail="Missing required ES configuration in environment"
      )
    
    # Initialize manager and run cleanup
    unsync_folder = img_cache_folder / 'unsync'
    manager = UnsyncFileManager(
      unsync_folder=unsync_folder,
      es_host=es_host,
      es_port=es_port,
      es_index=es_index,
      expiration_days=expiration_days
    )
    
    results = manager.cleanup_expired_files()
    
    # Summarize results
    summary = {
      'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
      'expiration_days': expiration_days,
      'total_files_processed': len(results),
      'files_to_delete': len([r for r in results if r['action'] in ('deleted', 'would_delete')]),
      'files_preserved': len([r for r in results if r['action'] == 'preserved']),
      'errors': len([r for r in results if r['action'] == 'error']),
      'details': results
    }
    
    return JSONResponse(content=summary)
    
  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))
  
# Run the server and reload on changes
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)

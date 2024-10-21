from typing import Optional, List, Dict, Any
from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException, Body, Query
from fastapi.responses import JSONResponse, PlainTextResponse
from pathlib import Path
import shutil, os, gzip, json
from datetime import datetime
import uvicorn

import pandas as pd
from dotenv import load_dotenv, find_dotenv

from celery.result import AsyncResult
from celery_unstack import unstack
from celery_embed import embed_listings, embed_listings_from_avm, remove_all_embed_listings_task_ids
from celery_update_embeddings import update_embeddings, update_inactive_embeddings
from celery_embed_index import embed_and_index_task

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
async def embed_and_index(image_batch_size: int = Query(32), text_batch_size: int = Query(128), num_workers: int = Query(4), delete_incoming: bool = Query(True)):
  task = embed_and_index_task.apply_async(args=[str(img_cache_folder), 
                                                listing_fields, 
                                                image_batch_size, 
                                                text_batch_size, 
                                                num_workers,
                                                delete_incoming], queue='embed_index_queue')
  
  return JSONResponse(content={"message": "Embedding and indexing task started.", "task_id": task.id})

@app.get("/update_inactive_vec_index")
async def update_inactive_vec_index():
  """
  Consolidate all embeddings in img_cache_folder and update the vector index for the search service.
  """
  update_inactive_embeddings.apply_async(args=[str(img_cache_folder)], queue='update_embed_queue')

  return JSONResponse(content={"message": "Updating inactive vector index."})


@app.get("/task_status/{task_id}")
async def get_task_status(task_id: str):
  task_result = AsyncResult(task_id)
  response = {
      "task_id": task_id,
      "status": task_result.status,
      "result": task_result.result if task_result.ready() else None
  }
  return JSONResponse(content=response)

# Run the server and reload on changes
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)

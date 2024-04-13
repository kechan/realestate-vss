from typing import Optional, List, Dict, Any
from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException, Body
from fastapi.responses import JSONResponse, PlainTextResponse
from pathlib import Path
import shutil, os, gzip, json
from datetime import datetime
import uvicorn

import pandas as pd
from dotenv import load_dotenv, find_dotenv

from celery_unstack import unstack
from celery_embed import embed_listings, embed_listings_from_avm
from celery_update_embeddings import update_embeddings, update_inactive_embeddings

# use this to establish a public endpt for image tagging service pipeline to upload image to
# ./ngrok http 8000 

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
  return PlainTextResponse("not_ok")

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
async def submit_listing_jsons(body: bytes = Body(...)):
  """
  Submission of listing jsons in gzip format. For now, they come from data dump from the AVM monitoring service.
  """
  
  try:
    decompressed_data = gzip.decompress(body)
    json_data: List[Dict[str, Any]] = json.loads(decompressed_data)

    # print(type(json_data[0]['propertyFeatures']))

    embed_listings_from_avm.apply_async(args=[json_data], queue='embed_queue')
      
    return {"message": "Data processed successfully", "received_records": len(json_data)}
  except Exception as e:
    raise HTTPException(status_code=400, detail=f"Error processing data: {str(e)}")


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

@app.get("/update_inactive_vec_index")
async def update_inactive_vec_index():
  """
  Consolidate all embeddings in img_cache_folder and update the vector index for the search service.
  """
  update_inactive_embeddings.apply_async(args=[str(img_cache_folder)], queue='update_embed_queue')

  return JSONResponse(content={"message": "Updating inactive vector index."})



# Run the server and reload on changes
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)

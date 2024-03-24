from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, PlainTextResponse
from pathlib import Path
import shutil, os
from datetime import datetime
import uvicorn
from dotenv import load_dotenv, find_dotenv

from celery_unstack import unstack
from celery_embed import embed_images
from celery_update_embeddings import update_embeddings

# use this to establish a public endpt for image tagging service pipeline to upload image to
# ./ngrok http 8000 

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

@app.get("/embed")
async def embed():
  # embed all the images under img_cache_folder
  # embed_images.delay(img_cache_folder=str(img_cache_folder))
  # Inside the embed endpoint
  embed_images.apply_async(args=[str(img_cache_folder)], queue='embed_queue')

  return JSONResponse(content={"message": "Embedding images."})

@app.get("/update_vec_index")
async def update_vec_index():
  """
  Consolidate all embeddings in img_cache_folder and update the vector index for the search service.
  """
  update_embeddings.apply_async(args=[str(img_cache_folder)], queue='update_embed_queue')

  return JSONResponse(content={"message": "Updating vector index."})


# Run the server and reload on changes
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)

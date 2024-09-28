from typing import Optional, List, Dict, Any

import shutil, os, requests, time
from pathlib import Path
import pandas as pd
from PIL import Image
import pillow_heif
from io import BytesIO
from datetime import datetime

from celery import Celery
from celery.utils.log import get_task_logger

import torch

import realestate_core.common.class_extensions
from realestate_vision.common.utils import get_listingId_from_image_name
from realestate_core.common.utils import flatten_list
from realestate_vss.models.embedding import OpenClipImageEmbeddingModel, OpenClipTextEmbeddingModel

from realestate_vss.data.es_client import ESClient
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import NotFoundError
from elasticsearch.helpers import scan

from dotenv import load_dotenv, find_dotenv

import redis

celery = Celery('embed_app', broker='pyamqp://guest@localhost//')
celery.conf.worker_cancel_long_running_tasks_on_connection_loss = True
celery_logger = get_task_logger(__name__)

# celery -A celery_embed.celery worker --loglevel=info --logfile=celery_embed.log --detach -P solo
# celery -A celery_embed.celery worker --loglevel=info --logfile=celery_embed.log -P solo -Q embed_queue --detach --hostname=embed_worker@%h

# ps aux | grep 'celeryd' | awk '{print $2}' | xargs kill -9
# ps aux | grep 'celeryd' | grep embed | awk '{print $2}' | xargs kill -9

# rabbitmqctl list_queues|grep embed_queue
# rabbitmqctl purge_queue embed_queue

model_name = 'ViT-L-14'
pretrained='laion2b_s32b_b82k'

IMAGE_BATCH_SIZE = 128    # num of images to accumulate before sending them to embedding model
TASK_ID_PREFIX = 'embed_listings_task:'
LISTING_IMAGES_LASTUPDATE_PREFIX = 'embed_images:'
LISTING_TEXT_LASTUPDATE_PREFIX = 'embed_text:'

@celery.task(bind=True, acks_late=False, max_retries=2)
def embed_listings(self, img_cache_folder: str, es_fields: List[str], listing_start_num: Optional[int] = None, listing_end_num: Optional[int] = None):
  """
  Embed images and text for listings in img_cache_folder. img_cache_folder contains folders with listing images with name of folder the listing_id.
  Given the images, this will use the model to embed the images. The remarks of the listing will be retrieved from ES and embedded as well in sentence chunks.
  """
  _ = load_dotenv(find_dotenv())
  if "ES_HOST" in os.environ and "ES_PORT" in os.environ and "ES_LISTING_INDEX_NAME" in os.environ:
    es_host = Path(os.environ["ES_HOST"])
    es_port = Path(os.environ["ES_PORT"])
    listing_index_name = Path(os.environ["ES_LISTING_INDEX_NAME"])
  else:
    raise ValueError("ES_HOST, ES_PORT and ES_LISTING_INDEX_NAME not found in .env")

  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')

  img_cache_folder = Path(img_cache_folder)
  listing_folders = img_cache_folder.lfre(r'^\d+$')
  celery_logger.info(f'Total # of listings in {img_cache_folder}: {len(set(listing_folders))}')

  if len(listing_folders) == 0:
    celery_logger.info('No listings found. Exiting...')
    return

  if listing_start_num is not None and listing_end_num is not None:
    listing_folders = listing_folders[listing_start_num:listing_end_num]

  pattern = r'(?<!stacked_resized_)\d+_\d+\.jpg'   # skip the stacked_resized*.jpg files
  image_paths = flatten_list([folder.lfre(pattern) for folder in listing_folders])
  celery_logger.info(f'# of images getting embedded: {len(image_paths)}')

  image_embedding_model = OpenClipImageEmbeddingModel(model_name=model_name, pretrained=pretrained, device=device)
  embeddings_df = image_embedding_model.embed(image_paths=image_paths)

  # Get the job_id from the celery task itself
  job_id = self.request.id

  (img_cache_folder/f'{model_name}_{pretrained}').mkdir(parents=True, exist_ok=True)
  embeddings_df.to_feather(img_cache_folder/f'{model_name}_{pretrained}'/f'{job_id}_image_embeddings_df')

  # get ES json for each listing, and embed the remarks
  # es = Elasticsearch([f'http://{es_host}:{es_port}/'])   
  es = ESClient(host=es_host, port=es_port, index_name=listing_index_name, fields=es_fields)
  if not es.ping():
    celery_logger.info('ES is not accessible. Exiting...')
    return

  listing_jsons = es.get_active_listings([folder.parts[-1] for folder in listing_folders])

  if len(listing_jsons) > 0:
    listing_df = pd.DataFrame(listing_jsons)

    listing_df.remarks = listing_df.remarks.fillna('')
    listing_df.to_feather(img_cache_folder/f'{model_name}_{pretrained}'/f'{job_id}_listing_df')

    text_embedder = OpenClipTextEmbeddingModel(embedding_model=image_embedding_model)
    text_embeddings_df = text_embedder.embed(df=listing_df, tokenize_sentences=True)
    text_embeddings_df.to_feather(img_cache_folder/f'{model_name}_{pretrained}'/f'{job_id}_text_embeddings_df')

  # delete all listing folders
  for folder in listing_folders:
    shutil.rmtree(folder)

@celery.task(bind=True, acks_late=True, max_retries=2)
def embed_listings_from_avm(self, 
                            img_cache_folder: str,
                            json_data: List[Dict[str, Any]], 
                            start_date: Optional[str] = None, 
                            end_date: Optional[str] = None):

  img_cache_folder = Path(img_cache_folder)
  img_archive_folder = img_cache_folder/'image_archives'

  pillow_heif.register_heif_opener()

  celery_logger.info(f'len(json_data): {len(json_data)}')
  celery_logger.info(f'start_date: {start_date}, end_date: {end_date}')

  try:
    start_date = datetime.strptime(start_date, '%Y-%m-%d') if start_date else None
    end_date = datetime.strptime(end_date, '%Y-%m-%d') if end_date else None
  except ValueError:
    celery_logger.error(f'Invalid date format. Expected YYYY-MM-DD. Got start_date: {start_date} or end_date: {end_date}')
    return

  # check if the task has already been processed, task status are stored on redis
  r = redis.Redis(host='localhost', port=6379, db=0)  #TODO: read from .env later
  task_id = TASK_ID_PREFIX + self.request.id
  if r.exists(task_id):
    celery_logger.info(f'Task {task_id} already processed. Exiting...')
    return
  
  r.set(task_id, 'processing 0%')

  # embed images and remarks
  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
  image_embedding_model = OpenClipImageEmbeddingModel(model_name=model_name, pretrained=pretrained, device=device)  

  # Initialize lists for batch processing
  batch_images, batch_image_names, batch_listing_ids = [], [], []

  image_embeddings_df, listing_df = [], []
  # json_data = json_data[:5]   # TODO: dev/debug, remove later
  for i, listing in enumerate(json_data):
    progress = (i+1)/len(json_data)*100
    r.set(task_id, f'processing {progress:.2f}%')  # so we can monitor this

    # check if photos updated date is newer than the specified date
    lastUpdate = listing.get('lastUpdate')
    lastPhotoUpdate = listing.get('lastPhotoUpdate')
    celery_logger.info(f'listing_id: {listing["jumpId"]}')

    # lastUpdate is a string that looks like 24-01-11:02:16:16, which is year, month, date convert it to a datetime object
    lastUpdate = datetime.strptime(lastUpdate, '%y-%m-%d:%H:%M:%S') if lastUpdate is not None else None
    lastPhotoUpdate = datetime.strptime(lastPhotoUpdate, '%y-%m-%d:%H:%M:%S') if lastPhotoUpdate is not None else None

    # IMPORTANT: it seems that lastPhotoUpdate is None if image has never been updated since the listing first published. 
    # so it seems that we should use assign lastPhotoUpdate to lastUpdate if lastPhotoUpdate is None 
    # This may need some revisions before deployment.
    if lastPhotoUpdate is None: lastPhotoUpdate = lastUpdate

    celery_logger.info(f'lastUpdate: {lastUpdate}, lastPhotoUpdate: {lastPhotoUpdate}')

    skip_image_embedding = False
    skip_text_embedding = False

    # if start_date and end_date are provided, check if the listing is within the date range
    # if only start_date is proviced, then check if listing is later than start_date
    # if only end_date is provided, then check if listing is earlier than end_date
    if start_date and end_date:
      if lastUpdate and (lastUpdate < start_date or lastUpdate > end_date):
        celery_logger.info(f'Listing {listing["jumpId"]} lastUpdate not within the date range. Skipping...')
        skip_text_embedding = True
      if lastPhotoUpdate and (lastPhotoUpdate < start_date or lastPhotoUpdate > end_date):
        celery_logger.info(f'Listing {listing["jumpId"]} lastPhotoUpdate not within the date range. Skipping...')
        skip_image_embedding = True

    elif start_date:
      if lastUpdate and lastUpdate < start_date:
        celery_logger.info(f'Listing {listing["jumpId"]} lastUpdate not within the date range. Skipping...')
        skip_text_embedding = True
      if lastPhotoUpdate and lastPhotoUpdate < start_date:
        celery_logger.info(f'Listing {listing["jumpId"]} lastPhotoUpdate not within the date range. Skipping...')
        skip_image_embedding = True
    elif end_date:
      if lastUpdate and lastUpdate > end_date:
        celery_logger.info(f'Listing {listing["jumpId"]} lastUpdate not within the date range. Skipping...')
        skip_text_embedding = True
      if lastPhotoUpdate and lastPhotoUpdate > end_date:
        celery_logger.info(f'Listing {listing["jumpId"]} lastPhotoUpdate not within the date range. Skipping...')
        skip_image_embedding = True

    # also check the updated time in redis cache 
    bLastUpdate = r.get(f"{LISTING_IMAGES_LASTUPDATE_PREFIX}{listing['jumpId']}") 
    if bLastUpdate is not None and datetime.strptime(bLastUpdate.decode('utf-8'), '%y-%m-%d:%H:%M:%S') > lastPhotoUpdate:
      celery_logger.info(f'Listing {listing["jumpId"]} image embedding already updated. Skipping...')
      skip_image_embedding = True
    
    bLastUpdate = r.get(f"{LISTING_TEXT_LASTUPDATE_PREFIX}{listing['jumpId']}")
    if bLastUpdate is not None and datetime.strptime(bLastUpdate.decode('utf-8'), '%y-%m-%d:%H:%M:%S') > lastUpdate:
      celery_logger.info(f'Listing {listing["jumpId"]} text embedding already updated. Skipping...')
      skip_text_embedding = True

    if not skip_image_embedding:

      # Temporary lists to hold data for the current listing
      temp_images, temp_image_names, temp_listing_ids = [], [], []
      celery_logger.info(f"listing_id: {listing['jumpId']}, # of images: {len(listing['photos'])}")

      now_in_str = datetime.now().strftime('%y-%m-%d:%H:%M:%S')
      r.set(f"{LISTING_IMAGES_LASTUPDATE_PREFIX}{listing['jumpId']}", f"{now_in_str}")   # mark the image embedding updated time

      # Construct the URLs and names for images in the current listing
      for image_name in listing['photos']:
        if image_name == 'photo_not_avail': continue

        url = f'https:{Path(listing["photo"]).parent}/{image_name}_lg.jpg'
        image_full_name = f'{image_name}.jpg'
        listing_id = get_listingId_from_image_name(image_full_name)

        try:
          response = requests.get(url)
          img = Image.open(BytesIO(response.content)).convert("RGB")
        except Exception as e:
          celery_logger.error(f'Error downloading image from {url}: {str(e)}')
          img = Image.new('RGB', (224, 224), (255, 255, 255))  # use a placeholder image on error
        time.sleep(0.1)  # sleep for 100ms to avoid rate limiting

        # save image in HEIF format
        (img_archive_folder/f'{listing_id}').mkdir(parents=True, exist_ok=True)
        heic_path = img_archive_folder/f'{listing_id}/{image_full_name.replace(".jpg", ".heic")}'
        img.save(heic_path, format="HEIF")
        celery_logger.info(f'Saved image {heic_path}')

        temp_images.append(img)
        temp_image_names.append(image_full_name)
        temp_listing_ids.append(listing_id)

      # Append current listing's images to the batch
      batch_images.extend(temp_images)
      batch_image_names.extend(temp_image_names)
      batch_listing_ids.extend(temp_listing_ids)

    # Process the batch if it reaches the size of IMAGE_BATCH_SIZE
    if len(batch_images) >= IMAGE_BATCH_SIZE:
      # Process the batch
      image_embeddings = image_embedding_model.embed_from_images(images=batch_images, return_df=False)
      image_embeddings_df.append(pd.DataFrame(data={
          'listing_id': batch_listing_ids,
          'image_name': batch_image_names,
          'embedding': image_embeddings.tolist()
      }))
      
      # Clear the batch for next set of images
      batch_images = []
      batch_image_names = []
      batch_listing_ids = []

    if not skip_text_embedding:  # add to listing for later embedding
      _listing_df = pd.DataFrame([listing])
      _listing_df.remarks = _listing_df.remarks.fillna('')
      listing_df.append(_listing_df)
      
      now_in_str = datetime.now().strftime('%y-%m-%d:%H:%M:%S')
      r.set(f"{LISTING_TEXT_LASTUPDATE_PREFIX}{listing['jumpId']}", f"{now_in_str}")   # mark the text embedding updated time
      
  # Check for any remaining images after all listings have been processed
  if batch_images:
    # Process the remaining images in the batch
    image_embeddings = image_embedding_model.embed_from_images(images=batch_images, return_df=False)
    image_embeddings_df.append(pd.DataFrame(data={
        'listing_id': batch_listing_ids,
        'image_name': batch_image_names,
        'embedding': image_embeddings.tolist()
    }))

  if len(image_embeddings_df) > 0:
    image_embeddings_df = pd.concat(image_embeddings_df, axis=0, ignore_index=True)
    image_embeddings_df.to_feather(img_cache_folder/f'{model_name}_{pretrained}'/f'{self.request.id}_image_embeddings_df')  # save to manual review for dev and debug

  # embed text all at once
  text_embedding_model = OpenClipTextEmbeddingModel(embedding_model=image_embedding_model)   # same underlying (clip) model as image embedding
  if len(listing_df) > 0:
    listing_df = pd.concat(listing_df, axis=0, ignore_index=True)
    listing_df.to_feather(img_cache_folder/f'{model_name}_{pretrained}'/f'{self.request.id}_listing_df')

    text_embeddings_df = text_embedding_model.embed(df=listing_df, tokenize_sentences=True)
    text_embeddings_df.to_feather(img_cache_folder/f'{model_name}_{pretrained}'/f'{self.request.id}_text_embeddings_df')  # save to manual review for dev and debug

  # mark the task as processed
  # r.set(task_id, 'completed')
  r.setex(task_id, 86400, 'completed')

@celery.task
def remove_all_embed_listings_task_ids():
  r = redis.Redis(host='localhost', port=6379, db=0)  #TODO: read from .env later
  for key in r.scan_iter(f"{TASK_ID_PREFIX}*"):
    r.delete(key)




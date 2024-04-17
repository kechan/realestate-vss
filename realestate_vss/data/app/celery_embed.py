from typing import Optional, List, Dict, Any

import shutil, os, requests, time
from pathlib import Path
import pandas as pd
from PIL import Image
from io import BytesIO

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

r = redis.Redis(host='localhost', port=6379, db=0)  # read from .env later

celery = Celery('embed_app', broker='pyamqp://guest@localhost//')
celery.conf.worker_cancel_long_running_tasks_on_connection_loss = True
celery_logger = get_task_logger(__name__)

# celery -A celery_embed.celery worker --loglevel=info --logfile=celery_embed.log --detach -P solo
# celery -A celery_embed.celery worker --loglevel=info --logfile=celery_embed.log -P solo -Q embed_queue --detach --hostname=embed_worker@%h

# ps aux | grep 'celeryd' | awk '{print $2}' | xargs kill -9
# ps aux | grep 'celeryd' | grep embed | awk '{print $2}' | xargs kill -9

model_name = 'ViT-L-14'
pretrained='laion2b_s32b_b82k'

counter = 0

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
  listing_folders = img_cache_folder.lfre('^\d+$')
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

@celery.task(bind=True, acks_late=False, max_retries=2)
def embed_listings_from_avm(self, json_data: List[Dict[str, Any]]):
  global counter
  celery_logger.info(f'len(json_data): {len(json_data)}')

  # check if the task has already been processed
  task_id = 'embed_listings_task:' + self.request.id
  if r.exists(task_id):
    celery_logger.info(f'Task {task_id} already processed. Exiting...')
    # delete the task from redis
    r.delete(task_id)
    return

  # pd.DataFrame(json_data).to_feather(Path.home()/'tmp'/f'rt_{counter}_listing_df')  # save to manual review for dev and debug

  # embed images
  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
  image_embedding_model = OpenClipImageEmbeddingModel(model_name=model_name, pretrained=pretrained, device=device)

  image_embeddings_df = []
  json_data = json_data[:3]   # TODO: dev/debug, remove later
  for listing in json_data:    
    # check if photos updated date is newer than the specified date
    lastUpdate = listing['lastUpdate']
    lastPhotoUpdate = listing['lastPhotoUpdate']
    celery_logger.info(f'listing_id: {listing["jumpId"]}, lastUpdate: {lastUpdate}, lastPhotoUpdate: {lastPhotoUpdate}')
    celery_logger.info(f'{type(lastUpdate)}')
    celery_logger.info(f'{type(lastPhotoUpdate)}')
    return


    image_urls = ['https:' + str(Path(listing['photo']).parent) + f'/{image_name}.jpg' for image_name in listing['photos']]
    image_names = [f'{image_name}.jpg' for image_name in listing['photos']]
    listing_ids = [get_listingId_from_image_name(image_name) for image_name in image_names]

    celery_logger.info(f'listing_id: {listing["jumpId"]}, # of images: {len(image_urls)}')

    imgs = []
    for url in image_urls:
      try:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        
      except Exception as e:
        celery_logger.error(f'Error downloading image from {url}: {str(e)}')
        img = Image.new('RGB', (224, 224), (255, 255, 255))  # white image

      imgs.append(img) 
      time.sleep(0.2)

    image_embeddings = image_embedding_model.embed_from_images(images=imgs, return_df=False)

    image_embeddings_df.append(pd.DataFrame(data={'listing_id': listing_ids, 'image_name': image_names, 'embedding': image_embeddings.tolist()}))
    

  image_embeddings_df = pd.concat(image_embeddings_df, axis=0, ignore_index=True)
  image_embeddings_df.to_feather(Path.home()/'tmp'/f'rt_{counter}_image_embeddings_df')  # save to manual review for dev and debug
  counter += 1

  # mark the task as processed
  r.set(task_id, 1)






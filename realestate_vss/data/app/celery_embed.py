from typing import Optional

import shutil, os
from pathlib import Path
import pandas as pd

from celery import Celery
from celery.utils.log import get_task_logger

import torch

import realestate_core.common.class_extensions
from realestate_core.common.utils import flatten_list
from realestate_vss.models.embedding import OpenClipImageEmbeddingModel, OpenClipTextEmbeddingModel

from realestate_vss.data.es_client import ESClient
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import NotFoundError
from elasticsearch.helpers import scan

from dotenv import load_dotenv, find_dotenv

celery = Celery('embed_app', broker='pyamqp://guest@localhost//')
celery_logger = get_task_logger(__name__)

# celery -A celery_embed.celery worker --loglevel=info --logfile=celery_embed.log --detach -P solo
# celery -A celery_embed.celery worker --loglevel=info --logfile=celery_embed.log -P solo -Q embed_queue --detach --hostname=embed_worker@%h

# ps aux | grep 'celeryd' | awk '{print $2}' | xargs kill -9
# ps aux | grep 'celeryd' | grep embed | awk '{print $2}' | xargs kill -9

model_name = 'ViT-L-14'
pretrained='laion2b_s32b_b82k'

@celery.task(bind=True, acks_late=False, max_retries=2)
def embed_images(self, img_cache_folder: str, listing_start_num: Optional[int] = None, listing_end_num: Optional[int] = None):
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
  es = ESClient(host=es_host, port=es_port, index_name=listing_index_name)
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


from typing import Optional, List, Dict, Union, Any
import os, shutil
from datetime import datetime, timedelta

from celery import Celery
from celery.utils.log import get_task_logger
from pathlib import Path
import pandas as pd

import weaviate
import torch

from realestate_vss.models.embedding import OpenClipImageEmbeddingModel, OpenClipTextEmbeddingModel
from realestate_vss.data.weaviate_datastore import WeaviateDataStore_v4 as WeaviateDataStore

from realestate_vss.data.es_client import ESClient
from realestate_core.common.utils import join_df, flatten_list

from realestate_analytics.data.bq import BigQueryDatastore

# from elasticsearch import Elasticsearch
# from elasticsearch.exceptions import NotFoundError
# from elasticsearch.helpers import scan

from dotenv import load_dotenv, find_dotenv

celery = Celery('embed_index_app', broker='pyamqp://guest@localhost//')
celery.conf.worker_cancel_long_running_tasks_on_connection_loss = True
celery_logger = get_task_logger(__name__)

LAST_RUN_FILE = 'celery_embed_index.last_run.log'

def get_last_run_time():
  try:
    with open(LAST_RUN_FILE, 'r') as f:
      last_run_str = f.read().strip()
      return datetime.fromisoformat(last_run_str)
  except FileNotFoundError:
    # If it's the first run, use a date thats a month ago
    celery_logger.warning(f"File {LAST_RUN_FILE} not found. Using default.")
    return datetime.now() - timedelta(days=39)
  except ValueError:
    # If the date string is invalid, also use a date far in the past
    celery_logger.warning(f"Invalid date in {LAST_RUN_FILE}. Using default.")
    return datetime.now() - timedelta(days=30)

def set_last_run_time(a_datetime: datetime):
  with open(LAST_RUN_FILE, 'w') as f:
    f.write(a_datetime.isoformat())

def process_and_batch_insert_to_datastore(embeddings_df: pd.DataFrame, 
                       listingIds: List[str], 
                       datastore: WeaviateDataStore,
                       aux_key: str,
                       listing_df: pd.DataFrame,
                       embedding_type: str = 'I') -> None:
  """
  Function to process embeddings and perform operations(add/delete) on Redis or Weaviate datastore.

  Parameters:
  embeddings_df (pd.DataFrame): DataFrame containing the embeddings.
  listingIds (List[Any]): List of listing IDs to be processed.
  datastore (Any): The Redis/Weaviate datastore object where docs are to be added/deleted.
  aux_key (str): The column name in the DataFrame that corresponds to the auxiliary key (image name or remark chunk ID).
                 This can also be thought of as the primary key to auxilliary information.  
  listing_df (pd.DataFrame): DataFrame containing the detail listing data.                 

  """
  items_to_process = list(embeddings_df.q("listing_id.isin(@listingIds)")[aux_key].values)
  processed_embeddings_df = embeddings_df.q(f"{aux_key}.isin(@items_to_process)")
  _df = join_df(processed_embeddings_df, listing_df, left_on='listing_id', right_on='jumpId', how='left').drop(columns=['jumpId'])
  listing_jsons = _df.to_dict(orient='records')
  datastore.batch_insert(listing_jsons, embedding_type=embedding_type)

@celery.task(bind=True)
def embed_and_index_task(self, img_cache_folder: str, es_fields: List[str], image_batch_size: int, text_batch_size: int, num_workers: int):

  # Gather environment variables for Weaviate and Elasticsearch
  _ = load_dotenv(find_dotenv())

  if not (os.getenv("USE_WEAVIATE").lower() == 'true'):
    raise ValueError("USE_WEAVIATE not set to 'true' in .env, this task is only for Weaviate")
  if "ES_HOST" in os.environ and "ES_PORT" in os.environ and "ES_LISTING_INDEX_NAME" in os.environ:
    es_host = Path(os.environ["ES_HOST"])
    es_port = Path(os.environ["ES_PORT"])
    listing_index_name = Path(os.environ["ES_LISTING_INDEX_NAME"])
  else:
    raise ValueError("ES_HOST, ES_PORT and ES_LISTING_INDEX_NAME not found in .env")

  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
  celery_logger.info(f'device: {device}')

  last_run = get_last_run_time()
  task_start_time = datetime.now()

  # Initialize models
  image_embedding_model = OpenClipImageEmbeddingModel(model_name='ViT-L-14', pretrained='laion2b_s32b_b82k', device=device)
  text_embedding_model = OpenClipTextEmbeddingModel(embedding_model=image_embedding_model, device=device)
  
  # Initialize Weaviate client and datastore and Elasticsearch client
  WEAVIATE_HOST = os.getenv("WEAVIATE_HOST")
  WEAVIATE_PORT = int(os.getenv("WEAVIATE_PORT"))
  celery_logger.info(f'weaviate_host: {WEAVIATE_HOST}, weaviate_port: {WEAVIATE_PORT}')
  client = weaviate.connect_to_local(WEAVIATE_HOST, WEAVIATE_PORT)  # TODO: change this before deployment
  datastore = WeaviateDataStore(client=client, image_embedder=None, text_embedder=None)

  es = ESClient(host=es_host, port=es_port, index_name=listing_index_name, fields=es_fields)
  if not es.ping():
    celery_logger.info('ES is not accessible. Exiting...')
    return

  # Resolve listings to be processed
  img_cache_folder = Path(img_cache_folder)
  listing_folders = img_cache_folder.lfre(r'^\d+$')
  celery_logger.info(f'Total # of listings in {img_cache_folder}: {len(set(listing_folders))}')
  if len(listing_folders) == 0:
    celery_logger.info('No listings found. Exiting...')
    return
  
  # Process images
  pattern = r'(?<!stacked_resized_)\d+_\d+\.jpg'   # skip the stacked_resized*.jpg files
  image_paths = flatten_list([folder.lfre(pattern) for folder in listing_folders])
  celery_logger.info(f'# of images getting embedded: {len(image_paths)}')
  image_embeddings_df = image_embedding_model.embed(image_paths=image_paths, 
                                                    batch_size=image_batch_size, 
                                                    num_workers=num_workers)
  
  # Process text
  listing_ids = [folder.parts[-1] for folder in listing_folders]
  listing_df = get_listing_data(es, listing_ids, es_fields)

  celery_logger.info(f'# of listings getting embedded: {len(listing_df)}')
  text_embeddings_df = text_embedding_model.embed(df=listing_df, 
                                                  batch_size=text_batch_size, 
                                                  tokenize_sentences=True, 
                                                  use_dataloader=True, 
                                                  num_workers=num_workers)
  
  # there can be dups in image_embeddings_df and text_embeddings_df, we keep the latest
  image_embeddings_df.drop_duplicates(subset=['image_name'], keep='last', inplace=True)
  image_embeddings_df.reset_index(drop=True, inplace=True)
  text_embeddings_df.drop_duplicates(subset=['remark_chunk_id'], keep='last', inplace=True)  
  text_embeddings_df.reset_index(drop=True, inplace=True)

  incoming_image_listingIds = set(image_embeddings_df.listing_id.unique())
  incoming_text_listingIds = set(text_embeddings_df.listing_id.unique())

  # Process image embeddings
  celery_logger.info(f'Processing  {len(incoming_image_listingIds)} listing image embeddings')
  datastore.delete_listings(listing_ids=incoming_image_listingIds, embedding_type='I')  
  process_and_batch_insert_to_datastore(
    embeddings_df=image_embeddings_df, 
    listingIds=incoming_image_listingIds,
    datastore=datastore, 
    aux_key='image_name', 
    listing_df=listing_df, 
    embedding_type='I'
  )

  # Process text embeddings
  celery_logger.info(f'Processing {len(incoming_text_listingIds)} listing text embeddings')
  datastore.delete_listings(listing_ids=incoming_text_listingIds, embedding_type='T')
  process_and_batch_insert_to_datastore(
    embeddings_df=text_embeddings_df, 
    listingIds=incoming_text_listingIds, 
    datastore=datastore, 
    aux_key='remark_chunk_id', 
    listing_df=listing_df, 
    embedding_type='T'
  )

  # remove deleted (delisted, sold, or inactive) listings from Weaviate
  bq_datastore = BigQueryDatastore()   # figure this out from big query
  deleted_listings_df = bq_datastore.get_deleted_listings(start_time=last_run)
  if len(deleted_listings_df) > 0:
    deleted_listing_ids = deleted_listings_df['listingId'].tolist()
    celery_logger.info(f'Removing {len(deleted_listing_ids)} deleted listings from Weaviate')
    datastore.delete_listings(listing_ids=deleted_listing_ids)
  
  set_last_run_time(task_start_time)

  """
  # delete all processed listing folders
  for folder in listing_folders:
    shutil.rmtree(folder)
  """

  datastore.close()

  return "Embedding and indexing completed successfully"

def get_listing_data(es: ESClient, listing_ids: List[str], es_fields: List[str]) -> pd.DataFrame:
  # Get listing info from ES using listing_ids
  listing_jsons = es.get_active_listings(listing_ids)
  listing_df = pd.DataFrame(listing_jsons)

  # Clean up some data  
  listing_df.remarks = listing_df.remarks.fillna('')

  # Convert lat to numeric (float), coercing non-numeric values to NaN
  listing_df.lat = pd.to_numeric(listing_df.lat, errors='coerce')
  listing_df.lng = pd.to_numeric(listing_df.lng, errors='coerce')
  listing_df.price = pd.to_numeric(listing_df.price, errors='coerce')
  listing_df.leasePrice = pd.to_numeric(listing_df.leasePrice, errors='coerce')

  return listing_df
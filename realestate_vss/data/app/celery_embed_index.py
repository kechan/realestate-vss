from typing import Optional, List, Dict, Union, Any
import os, shutil
from datetime import datetime, timedelta

from celery import Celery
from celery.utils.log import get_task_logger
from pathlib import Path
import pandas as pd

import weaviate
from weaviate.classes.init import AdditionalConfig, Timeout

import torch

from realestate_vss.models.embedding import OpenClipImageEmbeddingModel, OpenClipTextEmbeddingModel
from realestate_vss.data.weaviate_datastore import WeaviateDataStore_v4 as WeaviateDataStore

from realestate_vss.data.es_client import ESClient
import realestate_core.common.class_extensions
from realestate_core.common.utils import join_df, flatten_list

from realestate_analytics.data.bq import BigQueryDatastore

# from elasticsearch import Elasticsearch
# from elasticsearch.exceptions import NotFoundError
# from elasticsearch.helpers import scan

from dotenv import load_dotenv, find_dotenv

celery = Celery('embed_index_app', broker='pyamqp://guest@localhost//')
# celery.conf.worker_cancel_long_running_tasks_on_connection_loss = True

# Set log level for the Celery app
celery.conf.update(
  task_acks_late=True,
  worker_cancel_long_running_tasks_on_connection_loss=True,
  task_acks_on_failure_or_timeout=True,
  task_reject_on_worker_lost=True,
  worker_log_format="%(levelname)s: %(message)s",
  worker_task_log_format="%(levelname)s: %(message)s",
  worker_redirect_stdouts_level='INFO'  # Redirect stdout/stderr to Celery log at ERROR level
)

celery_logger = get_task_logger(__name__)
celery_logger.setLevel('INFO')

LAST_RUN_FILE = 'celery_embed_index.last_run.log'
RUN_LOG_FILE = 'celery_embed_index.run_log.csv'

def get_last_run_time():
  try:
    with open(LAST_RUN_FILE, 'r') as f:
      last_run_str = f.read().strip()
      return datetime.fromisoformat(last_run_str)
  except FileNotFoundError:
    # If it's the first run, use a date thats a month ago
    celery_logger.warning(f"File {LAST_RUN_FILE} not found. Using default.")
    # return datetime.now() - timedelta(days=5)
    return None
  except ValueError:
    # If the date string is invalid, also use a date far in the past
    celery_logger.warning(f"Invalid date in {LAST_RUN_FILE}. Using default.")
    # return datetime.now() - timedelta(days=5)
    return None

def set_last_run_time(a_datetime: datetime):
  with open(LAST_RUN_FILE, 'w') as f:
    f.write(a_datetime.isoformat())

def log_run(start_time: datetime, end_time: Optional[datetime], status: str):
  duration = (end_time - start_time).total_seconds()

  run_log_path = Path(RUN_LOG_FILE)
  run_data = {
    'start_time': [start_time],
    'end_time': [end_time],
    'status': [status],
    'duration': [duration]
  }
  run_df = pd.DataFrame(run_data)
  if run_log_path.exists():
    existing_df = pd.read_csv(run_log_path)
    updated_df = pd.concat([existing_df, run_df], ignore_index=True)
  else:
    updated_df = run_df
  updated_df.to_csv(run_log_path, index=False)

def process_and_batch_insert_to_datastore(embeddings_df: pd.DataFrame, 
                       listingIds: List[str], 
                       datastore: WeaviateDataStore,
                       aux_key: str,
                       listing_df: pd.DataFrame,
                       embedding_type: str = 'I') -> int:
  """
  Function to process embeddings and perform operations(add/delete) on Redis or Weaviate datastore.

  Parameters:
  embeddings_df (pd.DataFrame): DataFrame containing the embeddings.
  listingIds (List[Any]): List of listing IDs to be processed.
  datastore (Any): The Redis/Weaviate datastore object where docs are to be added/deleted.
  aux_key (str): The column name in the DataFrame that corresponds to the auxiliary key (image name or remark chunk ID).
                 This can also be thought of as the primary key to auxilliary information.  
  listing_df (pd.DataFrame): DataFrame containing the detail listing data.    

  Returns the number of documents inserted.
  """
  items_to_process = list(embeddings_df.q("listing_id.isin(@listingIds)")[aux_key].values)
  processed_embeddings_df = embeddings_df.q(f"{aux_key}.isin(@items_to_process)")
  _df = join_df(processed_embeddings_df, listing_df, left_on='listing_id', right_on='jumpId', how='left').drop(columns=['jumpId'])
  listing_jsons = _df.to_dict(orient='records')
  datastore.batch_insert(listing_jsons, embedding_type=embedding_type)
  # datastore.batch_upsert(listing_jsons, embedding_type=embedding_type)

  return len(listing_jsons)

@celery.task(bind=True, max_retries=3)
def embed_and_index_task(self, img_cache_folder: str, es_fields: List[str], image_batch_size: int, text_batch_size: int, num_workers: int):

  datastore, image_embeddings_df, text_embeddings_df = None, None, None
  task_status = "Failed"
  error_message = None
  img_cache_folder = Path(img_cache_folder)

  # Statistics
  stats = {
    "total_listings_processed": 0,
    "image_embeddings_inserted": 0,
    "image_listings_deleted": 0,
    "text_embeddings_inserted": 0,
    "text_listings_deleted": 0,
    "total_embeddings_inserted": 0,
    "total_listings_deleted": 0
  }

  # Gather environment variables for Weaviate and Elasticsearch
  _ = load_dotenv(find_dotenv())
  task_start_time = datetime.now()

  try:
    
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
    celery_logger.info(f'Last run time: {last_run}')
    
    # Initialize models
    image_embedding_model = OpenClipImageEmbeddingModel(model_name='ViT-L-14', pretrained='laion2b_s32b_b82k', device=device)
    text_embedding_model = OpenClipTextEmbeddingModel(embedding_model=image_embedding_model, device=device)
    
    # Initialize Weaviate client/datastore, and Elasticsearch client
    # Try connect_to_local looking for WEAVIATE_HOST and WEAVIATE_PORT first, if not found
    # then try connect_to_wcs looking for WCS_URL and WCS_API_KEY
    WEAVIATE_HOST = os.getenv("WEAVIATE_HOST")
    WEAVIATE_PORT = int(os.getenv("WEAVIATE_PORT")) if os.getenv("WEAVIATE_PORT") is not None else None
    if WEAVIATE_HOST is not None and WEAVIATE_PORT is not None:
      celery_logger.info(f'weaviate_host: {WEAVIATE_HOST}, weaviate_port: {WEAVIATE_PORT}')
      client = weaviate.connect_to_local(WEAVIATE_HOST, WEAVIATE_PORT)  # TODO: change this before deployment
    else:
      WCS_URL = os.getenv("WCS_URL")
      WCS_API_KEY = os.getenv("WCS_API_KEY")
      celery_logger.info(f'From .env, wcs_url: {WCS_URL}, wcs_api_key: {WCS_API_KEY}')

      if WCS_URL is None or WCS_API_KEY is None:
        celery_logger.error('WCS_URL and WCS_API_KEY not found in .env')
        raise Exception("WCS_URL and WCS_API_KEY not found in .env")
      
      # TODO: additional config is to resolve slow network issues, investigate if we should keep it for production
      client = weaviate.connect_to_wcs(
        additional_config=AdditionalConfig(timeout=Timeout(init=30, query=60, insert=120)),
        # skip_init_checks=True,
        cluster_url=WCS_URL,
        auth_credentials=weaviate.auth.AuthApiKey(WCS_API_KEY)
      )

    datastore = WeaviateDataStore(client=client, image_embedder=None, text_embedder=None)
    if not datastore.ping():
      celery_logger.info('Weaviate is not accessible. Exiting...')
      return

    es = ESClient(host=es_host, port=es_port, index_name=listing_index_name, fields=es_fields)
    if not es.ping():
      celery_logger.info('ES is not accessible. Exiting...')
      return

    # check for existing embedding files (if last run has exceptions and failed to complete)
    image_embeddings_file = img_cache_folder / 'image_embeddings_df'
    text_embeddings_file = img_cache_folder / 'text_embeddings_df'

    if image_embeddings_file.exists() and text_embeddings_file.exists():
      celery_logger.info(f"Loading existing embedding files")
      image_embeddings_df = pd.read_feather(image_embeddings_file)
      text_embeddings_df = pd.read_feather(text_embeddings_file)
    else:
      # Resolve listings to be processed    
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

    stats["total_listings_processed"] = len(incoming_image_listingIds.union(incoming_text_listingIds))

    # Process image embeddings
    celery_logger.info(f'Processing {len(incoming_image_listingIds)} listing image embeddings')
    if last_run is not None:
      celery_logger.info(f'Deleting incoming listing IDs before batch_insert (reconsider if this is needed)')
      datastore.delete_listings(listing_ids=incoming_image_listingIds, embedding_type='I')
      stats["image_listings_deleted"] = len(incoming_image_listingIds)
    else:
      celery_logger.info(f'Skipping deletion of incoming listing IDs (first run or intentional skip)')
    
    stats["image_embeddings_inserted"] = process_and_batch_insert_to_datastore(
      embeddings_df=image_embeddings_df, 
      listingIds=incoming_image_listingIds,
      datastore=datastore, 
      aux_key='image_name', 
      listing_df=listing_df, 
      embedding_type='I'
    )

    # Process text embeddings
    celery_logger.info(f'Processing {len(incoming_text_listingIds)} listing text embeddings')
    if last_run is not None:
      celery_logger.info(f'Deleting incoming listing IDs before batch_insert (reconsider if this is needed)')
      datastore.delete_listings(listing_ids=incoming_text_listingIds, embedding_type='T')
      stats["text_listings_deleted"] = len(incoming_text_listingIds)
    else:
      celery_logger.info(f'Skipping deletion of incoming listing IDs (first run or intentional skip)')

    stats["text_embeddings_inserted"] = process_and_batch_insert_to_datastore(
      embeddings_df=text_embeddings_df, 
      listingIds=incoming_text_listingIds, 
      datastore=datastore, 
      aux_key='remark_chunk_id', 
      listing_df=listing_df, 
      embedding_type='T'
    )

    # remove deleted (delisted, sold, or inactive) listings from Weaviate
    if last_run is not None:
      bq_datastore = BigQueryDatastore()   # figure this out from big query
      deleted_listings_df = bq_datastore.get_deleted_listings(start_time=last_run)
      if len(deleted_listings_df) > 0:
        deleted_listing_ids = deleted_listings_df['listingId'].unique().tolist()
        celery_logger.info(f'Removing {len(deleted_listing_ids)} deleted listings from Weaviate since {last_run}')
        # datastore.delete_listings(listing_ids=deleted_listing_ids)
        datastore.delete_listings_by_batch(listing_ids=deleted_listing_ids, batch_size=10)
        stats["total_listings_deleted"] = len(deleted_listing_ids)
    else:
      celery_logger.info(f'Skipping deletion of deleted listings (first run or intentional skip)')
    
    set_last_run_time(task_start_time)

    # delete all processed listing folders
    # for folder in listing_folders:
    processed_listing_ids = incoming_image_listingIds.union(incoming_text_listingIds)
    for listing_id in processed_listing_ids:
      listing_folder_path = img_cache_folder / str(listing_id)
      try:
        shutil.rmtree(listing_folder_path)    
        # shutil.move(listing_folder_path, img_cache_folder/'done')   # TODO: for dev temporarily
      except Exception as e:
        celery_logger.warning(f'Unable to remove {listing_folder_path}')


    # Calculate total statistics
    stats["total_embeddings_inserted"] = stats["image_embeddings_inserted"] + stats["text_embeddings_inserted"]
    stats["total_listings_deleted"] += stats["image_listings_deleted"] + stats["text_listings_deleted"]

    task_status = "Completed"

  except Exception as e:
    celery_logger.error(f"Error: {str(e)}")
    error_message = str(e)
    
    # if there's an error, try save the embeddings
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    if image_embeddings_df is not None and not image_embeddings_df.empty:
      image_embeddings_df.to_feather(image_embeddings_file)
    if text_embeddings_df is not None and not text_embeddings_df.empty:
      text_embeddings_df.to_feather(text_embeddings_file)
  finally:
    if datastore:
      datastore.close()

    task_end_time = datetime.now()
    # don't log if there's no listing and the status is Failed, these are mostly likely rerun due to timeout with rabbitmq
    if not (task_status == "Failed" and error_message is None and len(listing_folders) == 0):
      log_run(task_start_time, task_end_time, task_status)

    if task_status == "Completed":
      # Remove the temporary embedding files
      for file in [image_embeddings_file, text_embeddings_file]:
        if file.exists():
          try:
            file.unlink()
          except Exception as e:
            celery_logger.warning(f"Failed to delete {file}: {str(e)}")
      return {"status": "Completed", 
              "message": "Embedding and indexing completed successfully",              
              "stats": stats,
              "start_time": task_start_time.strftime("%Y-%m-%d %H:%M:%S"),
              "end_time": task_end_time.strftime("%Y-%m-%d %H:%M:%S")
              }
    else:
      return {"status": "Failed", 
              "error": error_message,
              "start_time": task_start_time.strftime("%Y-%m-%d %H:%M:%S"),
              "end_time": task_end_time.strftime("%Y-%m-%d %H:%M:%S")
              }


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
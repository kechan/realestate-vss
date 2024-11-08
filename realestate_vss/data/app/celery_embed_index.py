from typing import Optional, List, Dict, Union, Any, Tuple
import os, shutil, gc
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
from realestate_core.common.utils import join_df, flatten_list, save_to_pickle, load_from_pickle

from realestate_analytics.data.bq import BigQueryDatastore

# from elasticsearch import Elasticsearch
# from elasticsearch.exceptions import NotFoundError
# from elasticsearch.helpers import scan

# Restart workers after processing a certain number of tasks to free up memory.
# celery -A your_app worker --max-tasks-per-child=100

from dotenv import load_dotenv, find_dotenv

DELETE_INCOMING_IMAGE_EMBEDDINGS_SLEEP_TIME = 0.5
DELETE_INCOMING_TEXT_EMBEDDINGS_SLEEP_TIME = 0.5
# DELETE_BQ_LISTINGS_SLEEP_TIME = 0.5
BATCH_INSERT_SLEEP_TIME = 3

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
  worker_redirect_stdouts_level='INFO',  # Redirect stdout/stderr to Celery log at ERROR level

  # For connection handling
  broker_connection_timeout=60,          # 60 seconds connection timeout
  broker_heartbeat=120,                 # Heartbeat every 2 minutes
  broker_transport_options={
      'socket_timeout': 60.0,           # Socket timeout 60 seconds
      'socket_keepalive': True,         # Enable TCP keepalive
  }
)

celery_logger = get_task_logger(__name__)
celery_logger.setLevel('INFO')

LAST_RUN_FILE = 'celery_embed_index.last_run.log'
RUN_LOG_FILE = 'celery_embed_index.run_log.csv'

# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
# # Initialize models
# image_embedding_model = OpenClipImageEmbeddingModel(model_name='ViT-L-14', pretrained='laion2b_s32b_b82k', device=device)
# text_embedding_model = OpenClipTextEmbeddingModel(embedding_model=image_embedding_model, device=device)

device = None
image_embedding_model = None
text_embedding_model = None

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
  datastore.batch_insert(listing_jsons, 
                         embedding_type=embedding_type, 
                         batch_size=1000, 
                         sleep_time=BATCH_INSERT_SLEEP_TIME)   # do we need this sleep if not for using free weaviate cloud.
  # datastore.batch_upsert(listing_jsons, embedding_type=embedding_type)

  return len(listing_jsons)


def split_image_embeddings_by_listing_df(embeddings_df: pd.DataFrame, 
                                        listing_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
  """
  Splits image embeddings based on whether their listings exist in the listing data.
  Modifies embeddings_df in place to contain only embeddings with synchronized listings.

  By "synchronized", we mean that the image embeddings have corresponding listings in the listing data.
  
  Args:
    embeddings_df: DataFrame containing image embeddings, will be modified in place
    listing_df: DataFrame containing listing data
  
  Returns:
    Tuple of (synchronized_df, unsynchronized_df)
    synchronized_df is the modified original embeddings_df (contains only "synced" listings)
    unsynchronized_df contains embeddings whose listings were not found
  """
  # Get list of synchronized listing IDs
  sync_listing_ids = set(listing_df['jumpId'].unique())
  
  # Get boolean mask for unsynchronized entries
  is_unsynced = ~embeddings_df['listing_id'].isin(sync_listing_ids)
  
  if is_unsynced.any():
    # Extract unsynchronized records and reset index
    unsync_df = embeddings_df.loc[is_unsynced].reset_index(drop=True)
    
    # Remove unsynchronized records from original dataframe
    embeddings_df.drop(index=embeddings_df[is_unsynced].index, inplace=True)
    embeddings_df.reset_index(drop=True, inplace=True)
  else:
    unsync_df = pd.DataFrame()
  
  return embeddings_df, unsync_df


def process_unsync_embedding_file(unsync_embedding_file: Path,
                                  datastore: WeaviateDataStore,
                                  es: ESClient,
                                  es_fields: List[str],
                                  text_batch_size=128,
                                  num_workers=4) -> Dict[str, int]:
  """
  Processes a single "unsynchronized" image embeddings file (a dataframe) by attempting to find their listings
  attributes from ES server and do text embeddings on these listings.
  
  And finally follow by overwriting the file with remaining unsynchronized embeddings.
  
  By "unsynchronized", we mean that the image embeddings have not been indexed in Weaviate yet 'cos their
  corresponding listing wasn't in ES at the time of the last embed/indexing run.
  
  Args:
    unsync_embedding_file: Path to unsync embeddings file
    datastore: Weaviate datastore instance
    es: ESClient instance
    es_fields: List of fields to retrieve from ES
  
  Returns:
    Dictionary with processing statistics for this file
  """
  stats = {
    "total_processed": 0,
    "newly_synced_images": 0,
    "newly_synced_texts": 0,
    "still_unsync": 0
  }

  try:
    embeddings_df = pd.read_feather(unsync_embedding_file)
    if embeddings_df.empty:
      unsync_embedding_file.unlink()  # Remove empty file
      return stats
      
    stats["total_processed"] = len(embeddings_df)
    
    # Get current listing data
    listing_ids = embeddings_df['listing_id'].unique().tolist()
    listing_df = get_listing_data(es, listing_ids, es_fields)   # get listing data from ES

    if len(listing_df) == 0:
      # No listings found, keep the file as is      
      stats["still_unsync"] = len(embeddings_df)
      return stats
    
    # Split into synchronized and still unsynchronized
    to_be_indexed_embeddings_df, still_unsync_df = split_image_embeddings_by_listing_df(
      embeddings_df,
      listing_df
    )
    
    stats["newly_synced_images"] = len(to_be_indexed_embeddings_df)
    stats["still_unsync"] = len(still_unsync_df)
    
    if len(to_be_indexed_embeddings_df) > 0:
      celery_logger.info(f"Begin batch insert {len(to_be_indexed_embeddings_df)} image embeddings to weaviate")
      process_and_batch_insert_to_datastore(
        embeddings_df=to_be_indexed_embeddings_df,
        listingIds=to_be_indexed_embeddings_df['listing_id'].unique(),
        datastore=datastore,
        aux_key='image_name',
        listing_df=listing_df,
        embedding_type='I'
      )
      celery_logger.info(f"Ended batch insert {len(to_be_indexed_embeddings_df)} image embeddings to weaviate")

      # Generate and process text embeddings for the listings in listing_df
      celery_logger.info(f'Begin text embedding for {len(listing_df)} listings')
      text_embeddings_df = text_embedding_model.embed(df=listing_df, 
                                                      batch_size=text_batch_size, 
                                                      tokenize_sentences=True, 
                                                      # use_dataloader=True, 
                                                      use_dataloader=False, 
                                                      num_workers=num_workers)
      celery_logger.info(f'Ended text embedding for {len(listing_df)} listings')

      if not text_embeddings_df.empty:
        text_embeddings_df.drop_duplicates(subset=['remark_chunk_id'], keep='last', inplace=True)  
        text_embeddings_df.reset_index(drop=True, inplace=True)

        # Process and index text embeddings
        celery_logger.info(f"Begin batch insert {len(text_embeddings_df)} text embeddings to weaviate")
        stats["newly_synced_texts"] = process_and_batch_insert_to_datastore(
          embeddings_df=text_embeddings_df,
          listingIds=text_embeddings_df['listing_id'].unique(),
          datastore=datastore,
          aux_key='remark_chunk_id',
          listing_df=listing_df,
          embedding_type='T'
        )
        celery_logger.info(f"Ended batch insert {len(text_embeddings_df)} text embeddings to weaviate")
    
    # Handle the unsync file based on results
    if len(still_unsync_df) == 0:
      # All embeddings synced, remove the file
      unsync_embedding_file.unlink()
      celery_logger.info(f"Removed fully synced file: {unsync_embedding_file.name}")
    else:
      # Some embeddings still unsynced, overwrite file with remaining ones
      still_unsync_df.to_feather(unsync_embedding_file)
      celery_logger.info(f"Updated {unsync_embedding_file.name} with {len(still_unsync_df)} remaining unsynced embeddings")
      
  except Exception as e:
    celery_logger.error(f"Error processing unsync file {unsync_embedding_file}: {e}")
    stats["error"] = str(e)
    
  return stats


def process_unsync_image_embeddings(img_cache_folder: Path,
                                    datastore: WeaviateDataStore,
                                    es: ESClient,
                                    es_fields: List[str],
                                    text_batch_size=128,
                                    num_workers=4) -> Dict[str, int]:
  """
  Processes all unsynchronized image embeddings files, attempting to sync their listings.
  Processes each file individually and tracks overall statistics.
  """
  unsync_folder = img_cache_folder / 'unsync'
  if not unsync_folder.exists():
    unsync_folder.mkdir(exist_ok=True)
    return {"files_processed": 0, "total_processed": 0, "newly_synced_images": 0, "newly_synced_texts": 0, "still_unsync": 0}
  
  unsync_files = list(unsync_folder.lf("unsync_image_embeddings_df.*"))
  if not unsync_files:
    return {"files_processed": 0, "total_processed": 0, "newly_synced_images": 0, "newly_synced_texts": 0, "still_unsync": 0}
  
  aggregate_stats = {
    "files_processed": len(unsync_files),
    "total_processed": 0,
    "newly_synced_images": 0,
    "newly_synced_texts": 0,
    "still_unsync": 0
  }
  
  for unsync_file in unsync_files:
    celery_logger.info(f"Processing unsync file: {unsync_file.name}")
    file_stats = process_unsync_embedding_file(unsync_file, 
                                               datastore, 
                                               es, 
                                               es_fields,
                                               text_batch_size,
                                               num_workers
                                               )
    
    # Aggregate statistics
    aggregate_stats["total_processed"] += file_stats["total_processed"]
    aggregate_stats["newly_synced_images"] += file_stats["newly_synced_images"]
    aggregate_stats["newly_synced_texts"] += file_stats["newly_synced_texts"]
    aggregate_stats["still_unsync"] += file_stats["still_unsync"]
    
  return aggregate_stats


@celery.task(bind=True, max_retries=3)
def embed_and_index_task(self, 
                        img_cache_folder: str, 
                        es_fields: List[str], 
                        image_batch_size: int, 
                        text_batch_size: int, 
                        num_workers: int, 
                        delete_incoming=True
                        ):

  global device, image_embedding_model, text_embedding_model
  if device is None:
    device = torch.device('cuda') if torch.cuda.is_available() else \
             torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
  if image_embedding_model is None:
    image_embedding_model = OpenClipImageEmbeddingModel(model_name='ViT-L-14', pretrained='laion2b_s32b_b82k', device=device)
    
  if text_embedding_model is None:
    text_embedding_model = OpenClipTextEmbeddingModel(embedding_model=image_embedding_model, device=device)

  datastore, es, image_embeddings_df, text_embeddings_df = None, None, None, None
  listing_folders = None
  task_status = "Failed"
  error_message = None
  img_cache_folder = Path(img_cache_folder)
  unsync_folder = img_cache_folder / 'unsync'
  unsync_folder.mkdir(exist_ok=True)

  # Statistics
  stats = {
    "total_listings_processed": 0,
    "image_embeddings_inserted": 0,
    "image_listings_deleted": 0,
    "text_embeddings_inserted": 0,
    "text_listings_deleted": 0,
    "total_embeddings_inserted": 0,
    # "total_listings_deleted": 0,
    # "total_embeddings_deleted": 0
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
    
    celery_logger.info(f'device: {device}')

    last_run = get_last_run_time()
    celery_logger.info(f'Last run time: {last_run}')
    
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
    
    # process any existing unsynced image embeddings first
    unsync_stats = process_unsync_image_embeddings(img_cache_folder, datastore, es, es_fields, text_batch_size, num_workers)
    celery_logger.info(f"Unsynced processing stats: {unsync_stats}")

    # check for existing embedding files (if last run has exceptions and failed to complete)
    image_embeddings_file = img_cache_folder / 'image_embeddings_df'
    text_embeddings_file = img_cache_folder / 'text_embeddings_df'
    listing_df_file = img_cache_folder / 'listing_df'
    listing_folders_pickle_file = img_cache_folder / 'listing_folders.pkl'

    # used to mark delete for incoming image listings
    image_delete_marker = img_cache_folder / 'image_delete_completed'
    text_delete_marker = img_cache_folder / 'text_delete_completed'

    if image_embeddings_file.exists():
      celery_logger.info(f"Loading existing image embedding file")
      image_embeddings_df = pd.read_feather(image_embeddings_file)
    else:
      # Resolve listings to be processed    
      listing_folders = img_cache_folder.lfre(r'^\d+$')      
      celery_logger.info(f'Total # of listings in {img_cache_folder}: {len(set(listing_folders))}')
      if len(listing_folders) == 0:
        celery_logger.info('No listings found. Exiting...')
        task_status = "Completed"
        return
      
      # save listing_folders to pickle file for later recovery use if applicable
      save_to_pickle(listing_folders, listing_folders_pickle_file)
      
      # Embed images
      pattern = r'(?<!stacked_resized_)\d+_\d+\.jpg'   # skip the stacked_resized*.jpg files
      image_paths = flatten_list([folder.lfre(pattern) for folder in listing_folders])
      celery_logger.info(f'Begin embedding {len(image_paths)} images')
      image_embeddings_df = image_embedding_model.embed(image_paths=image_paths, 
                                                        batch_size=image_batch_size, 
                                                        num_workers=num_workers)
      celery_logger.info(f'Ended embedding {len(image_paths)} images')

    if text_embeddings_file.exists() and listing_df_file.exists():
      celery_logger.info(f"Loading existing text embedding file")
      text_embeddings_df = pd.read_feather(text_embeddings_file)
      listing_df = pd.read_feather(listing_df_file)
    else:
      # Embed text
      # Note: listing_folders must exist since image embedding run first and it is responsible for instantiing it and/or dumping this pickle
      if listing_folders is None:
        listing_folders = load_from_pickle(listing_folders_pickle_file)  
      listing_ids = [folder.parts[-1] for folder in listing_folders]
      listing_df = get_listing_data(es, listing_ids, es_fields)  # get listing data from ES

      if len(listing_df) > 0:
        celery_logger.info(f'Begin text embedding for {len(listing_df)} listings')
        text_embeddings_df = text_embedding_model.embed(df=listing_df, 
                                                        batch_size=text_batch_size, 
                                                        tokenize_sentences=True, 
                                                        # use_dataloader=True, 
                                                        use_dataloader=False, 
                                                        num_workers=num_workers)
        celery_logger.info(f'Ended text embedding for {len(listing_df)} listings')
      else:
        celery_logger.info('No corresponding listings from images found from ES.')
        # create a dummy df with all required cols to fulfil downstream processing
        text_embeddings_df = pd.DataFrame(columns=['listing_id', 'remark_chunk_id', 'embedding'])

      # there can be dups in image_embeddings_df and text_embeddings_df, we keep the latest
      image_embeddings_df.drop_duplicates(subset=['image_name'], keep='last', inplace=True)
      image_embeddings_df.reset_index(drop=True, inplace=True)
      text_embeddings_df.drop_duplicates(subset=['remark_chunk_id'], keep='last', inplace=True)  
      text_embeddings_df.reset_index(drop=True, inplace=True)

    incoming_image_listingIds = set(image_embeddings_df.listing_id.unique())
    incoming_text_listingIds = set(text_embeddings_df.listing_id.unique())

    stats["total_listings_processed"] = len(incoming_image_listingIds.union(incoming_text_listingIds))

    # Process image embeddings
    # Delete incoming image embeddings
    celery_logger.info(f'Processing {len(incoming_image_listingIds)} listings and {image_embeddings_df.shape[0]} image embeddings')
    if delete_incoming and last_run is not None and not image_delete_marker.exists():
      celery_logger.info(f'Begin deleting incoming listing IDs before batch_insert')
      # datastore.delete_listings(listing_ids=incoming_image_listingIds, embedding_type='I')
      datastore.delete_listings_by_batch(listing_ids=list(incoming_image_listingIds), 
                                         embedding_type='I', 
                                         batch_size=10, 
                                         sleep_time=DELETE_INCOMING_IMAGE_EMBEDDINGS_SLEEP_TIME)
      celery_logger.info(f'Ended deleting incoming listing IDs before batch_insert')

      stats["image_listings_deleted"] = len(incoming_image_listingIds)
      image_delete_marker.touch()
    else:
      celery_logger.info(f'Skipping deletion of incoming listing IDs (first run or intentional skip)')
    
    
    # Embed Image and Text
    if len(listing_df) == 0:
      # just save the entire image_embeddings_df for later indexing
      ts = datetime.now().strftime("%Y%m%d%H%M")
      unsync_file = unsync_folder / f'unsync_image_embeddings_df.{ts}'
      image_embeddings_df.to_feather(unsync_file)
      celery_logger.info(f"Saved {len(image_embeddings_df)} unsynced image embeddings to {unsync_file}")

    else:
      # Insert incoming image embeddings
      # indexed only image embeddings whose listings is in listing_df, saved the otherwise for later indexing
      image_embeddings_df, unsync_df = split_image_embeddings_by_listing_df(image_embeddings_df, listing_df)
      if len(unsync_df) > 0:
        ts = datetime.now().strftime("%Y%m%d%H%M")
        unsync_file = unsync_folder / f'unsync_image_embeddings_df.{ts}'
        unsync_df.to_feather(unsync_file)
        celery_logger.info(f"Saved {len(unsync_df)} unsynced image embeddings to {unsync_file}")

      celery_logger.info("Begin batch insert image embeddings to weaviate")
      stats["image_embeddings_inserted"] = process_and_batch_insert_to_datastore(
        embeddings_df=image_embeddings_df, 
        listingIds=image_embeddings_df['listing_id'].unique(),
        datastore=datastore, 
        aux_key='image_name', 
        listing_df=listing_df, 
        embedding_type='I'
      )
      celery_logger.info("Ended batch insert image embeddings to weaviate")

      # Process text embeddings
      # Delete incoming text embeddings
      celery_logger.info(f'Processing {len(incoming_text_listingIds)} listings and {text_embeddings_df.shape[0]} text embeddings')
      if delete_incoming and last_run is not None and not text_delete_marker.exists():
        celery_logger.info(f'Begin deleting incoming listing IDs before batch_insert')
        # datastore.delete_listings(listing_ids=incoming_text_listingIds, embedding_type='T')
        datastore.delete_listings_by_batch(listing_ids=list(incoming_text_listingIds), 
                                          embedding_type='T', 
                                          batch_size=10, 
                                          sleep_time=DELETE_INCOMING_TEXT_EMBEDDINGS_SLEEP_TIME)
        celery_logger.info(f'Ended deleting incoming listing IDs before batch_insert')

        stats["text_listings_deleted"] = len(incoming_text_listingIds)
        text_delete_marker.touch()
      else:
        celery_logger.info(f'Skipping deletion of incoming listing IDs (first run or intentional skip)')

      # Insert incoming text embeddings
      celery_logger.info("Begin batch insert text embeddings to weaviate")
      stats["text_embeddings_inserted"] = process_and_batch_insert_to_datastore(
        embeddings_df=text_embeddings_df, 
        listingIds=incoming_text_listingIds, 
        datastore=datastore, 
        aux_key='remark_chunk_id', 
        listing_df=listing_df, 
        embedding_type='T'
      )
      celery_logger.info("Ended batch insert text embeddings to weaviate")

    # IMPORTANT: This is now done in celery_delete_inactive, this would be removed eventually.

    # Delete (delisted, sold, or inactive) from Weaviate listings obtained from BQ (for deleted listings)
    # if last_run is not None:
    #   bq_datastore = BigQueryDatastore()   # figure this out from big query
    #   deleted_listings_df = bq_datastore.get_deleted_listings(start_time=last_run)
    #   if len(deleted_listings_df) > 0:
    #     count_before_del = datastore.count_all()
    #     deleted_listing_ids = deleted_listings_df['listingId'].unique().tolist()
    #     celery_logger.info(f'Begin removing {len(deleted_listing_ids)} deleted listings from Weaviate since {last_run}')

    #     datastore.delete_listings_by_batch(listing_ids=deleted_listing_ids, 
    #                                        batch_size=20, 
    #                                        sleep_time=DELETE_BQ_LISTINGS_SLEEP_TIME)
    #     count_after_del = datastore.count_all()
    #     celery_logger.info(f'Ended removing {len(deleted_listing_ids)} deleted listings from Weaviate since {last_run}')

    #     stats["total_listings_deleted"] = len(deleted_listing_ids)
    #     stats["total_embeddings_deleted"] = count_before_del - count_after_del
    # else:
    #   celery_logger.info(f'Skipping deletion of deleted listings (first run or intentional skip)')
    
    set_last_run_time(task_start_time)

    # delete all processed listing folders
    
    # processed_listing_ids = incoming_image_listingIds.union(incoming_text_listingIds)
    if listing_folders is not None:
      celery_logger.info(f'Deleting all processed {len(listing_folders)} listing folders')
      for listing_folder in listing_folders:
        # listing_folder_path = img_cache_folder / str(listing_id)
        try:
          shutil.rmtree(listing_folder)    
          # shutil.move(listing_folder_path, img_cache_folder/'done')   # TODO: for dev temporarily
        except Exception as e:
          celery_logger.warning(f'Unable to remove {listing_folder}')
      celery_logger.info(f'Deleted all processed {len(listing_folders)} listing folders')
    else:
      celery_logger.warning('No listing folders to delete because listing_folders is None (please investigate)')

    # Calculate total statistics
    stats["total_embeddings_inserted"] = stats["image_embeddings_inserted"] + stats["text_embeddings_inserted"]
    # stats["total_listings_deleted"] += stats["image_listings_deleted"] + stats["text_listings_deleted"]

    # TODO: consider commeting these out when deployed
    # Backup
    backup(
      backup_folder=img_cache_folder/'backup',
      image_embeddings_df=image_embeddings_df,
      text_embeddings_df=text_embeddings_df,
      listing_df=listing_df,
      compression='zstd',
      compression_level=3
    )

    # timestamp = datetime.now().strftime("%Y%m%d%H%M")
    # if image_embeddings_df is not None and not image_embeddings_df.empty:
    #   image_embeddings_df.to_feather(backup_folder/f'image_embeddings_df.{timestamp}')
    # if text_embeddings_df is not None and not text_embeddings_df.empty:
    #   text_embeddings_df.to_feather(backup_folder/f'text_embeddings_df.{timestamp}')
    # if listing_df is not None and not listing_df.empty:
    #   listing_df.to_feather(backup_folder/f'listing_df.{timestamp}')

    task_status = "Completed"

  except Exception as e:
    celery_logger.error(f"Error: {str(e)}")
    error_message = str(e)
    
    # if there's an error, try save the embeddings and listing_df
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    if image_embeddings_df is not None and not image_embeddings_df.empty:
      image_embeddings_df.to_feather(image_embeddings_file)
    if text_embeddings_df is not None and not text_embeddings_df.empty:
      text_embeddings_df.to_feather(text_embeddings_file)
    if listing_df is not None and not listing_df.empty:
      listing_df.to_feather(listing_df_file)

  finally:
    if datastore:
      datastore.close()
    if 'es' in locals() and es is not None:
      es.close()

    # if 'bq_datastore' in locals() and bq_datastore is not None:
    #   try:
    #     bq_datastore.close()
    #   except Exception as e:
    #     pass

    del image_embeddings_df
    del text_embeddings_df
    gc.collect()
    try:
      if device.type == 'cuda':
        torch.cuda.empty_cache()
      elif device.type == 'mps':
        torch.mps.empty_cache()
    except Exception as e:
      celery_logger.warning(f"Failed to empty cache: {str(e)}")

    task_end_time = datetime.now()
    # TODO: consider log_run on every attempt.
    # don't log if there's no listing and the status is Failed, these are mostly likely rerun due to timeout with rabbitmq
    if not (task_status == "Failed" and error_message is None and listing_folders is not None and len(listing_folders) == 0):
      log_run(task_start_time, task_end_time, task_status)

    if task_status == "Completed":
      # Remove the temporary embedding files
      for file in [image_embeddings_file, text_embeddings_file, listing_df_file, 
                   image_delete_marker, text_delete_marker, 
                   listing_folders_pickle_file]:
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

  if len(listing_jsons) == 0:
    return pd.DataFrame()
  
  listing_df = pd.DataFrame(listing_jsons)

  # Clean up some data  
  listing_df.remarks = listing_df.remarks.fillna('')

  # Convert lat to numeric (float), coercing non-numeric values to NaN
  listing_df.lat = pd.to_numeric(listing_df.lat, errors='coerce')
  listing_df.lng = pd.to_numeric(listing_df.lng, errors='coerce')
  listing_df.price = pd.to_numeric(listing_df.price, errors='coerce')
  listing_df.leasePrice = pd.to_numeric(listing_df.leasePrice, errors='coerce')

  return listing_df

def backup(backup_folder: Path,
          image_embeddings_df: Optional[pd.DataFrame] = None,
          text_embeddings_df: Optional[pd.DataFrame] = None, 
          listing_df: Optional[pd.DataFrame] = None,
          compression: str = 'zstd',
          compression_level: int = 9) -> None:
  """
  Backs up dataframes with compression support.
  
  Args:
    backup_folder: Path to the backup directory
    image_embeddings_df: DataFrame containing image embeddings
    text_embeddings_df: DataFrame containing text embeddings 
    listing_df: DataFrame containing listing data
    compression: Compression algorithm to use ('zstd' or 'lz4')
    compression_level: Compression level (higher = better compression but slower)
  """
  if compression not in ['zstd', 'lz4']:
    raise ValueError("Compression must be either 'zstd' or 'lz4'")
  
  if not (0 <= compression_level <= 9):
    raise ValueError("Compression level must be between 0 and 9")  

  try:
    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    
    # Create backup folder if it doesn't exist
    backup_folder.mkdir(exist_ok=True, parents=True)
    
    # Helper function to safely save DataFrame with compression
    def save_df_compressed(df: pd.DataFrame, filename: str) -> None:
      try:
        df.to_feather(
          backup_folder / filename,
          compression=compression,
          compression_level=compression_level
        )
      except Exception as e:
        celery_logger.error(f"Failed to save {filename}: {str(e)}")
        # Fallback to uncompressed if compression fails
        try:
          df.to_feather(backup_folder / f"uncompressed_{filename}")
          celery_logger.info(f"Saved uncompressed fallback: {filename}")
        except Exception as e2:
          celery_logger.error(f"Failed to save uncompressed fallback {filename}: {str(e2)}")

    # Backup each DataFrame if it exists and is not empty
    if image_embeddings_df is not None and not image_embeddings_df.empty:
      save_df_compressed(image_embeddings_df, f'image_embeddings_df.{timestamp}')
    
    if text_embeddings_df is not None and not text_embeddings_df.empty:
      save_df_compressed(text_embeddings_df, f'text_embeddings_df.{timestamp}')
    
    if listing_df is not None and not listing_df.empty:
      save_df_compressed(listing_df, f'listing_df.{timestamp}')

  except Exception as e:
    celery_logger.error(f"Backup operation failed: {str(e)}")


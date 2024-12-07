from typing import Optional, List, Dict, Union, Any, Tuple
import os, shutil, html, gc
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
from realestate_vss.utils.email import send_email_alert

from realestate_vss.data.es_client import ESClient
import realestate_core.common.class_extensions
from realestate_core.common.utils import join_df, flatten_list, save_to_pickle, load_from_pickle

# from elasticsearch import Elasticsearch
# from elasticsearch.exceptions import NotFoundError
# from elasticsearch.helpers import scan

# Restart workers after processing a certain number of tasks to free up memory.
# celery -A your_app worker --max-tasks-per-child=100

from dotenv import load_dotenv, find_dotenv

BATCH_INSERT_SLEEP_TIME = 3
MAX_LISTING_TO_EMBED_INDEX = 1000 

_ = load_dotenv(find_dotenv())
# Get Redis host (for result backend) from .env or fall back to 127.0.0.1
REDIS_HOST = os.getenv('CELERY_BACKEND_REDIS_HOST_IP', '127.0.0.1')

if os.getenv('CELERY_ENABLE_RESULT_BACKEND', 'false').lower() == 'true':
  celery = Celery('embed_index_app', broker='pyamqp://guest@localhost//', backend=f'redis://{REDIS_HOST}:6379/1')
  # celery.conf.result_backend = f'redis://{REDIS_HOST}:6379/1'
  celery.conf.result_expires = 86400  # 1 day = 86400 seconds
  celery.conf.task_track_started = True
else:
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
  broker_connection_timeout=3600,          # 1 hour connection timeout
  broker_heartbeat=60,                 # Heartbeat every 1 minutes
  task_time_limit=3600,                  # 1 hr max task runtime
  task_soft_time_limit=3540,             # 1 min before hard time limit
  worker_prefetch_multiplier=1,
  # broker_transport_options={
  #     'socket_timeout': 60.0,           # Socket timeout 60 seconds
  #     'socket_keepalive': True,         # Enable TCP keepalive
  # },
  worker_concurrency=1,                   # Single worker process
  worker_max_tasks_per_child=1,

  broker_transport_options={
      'retry_on_timeout': True,
      'interval_start': 0,
      'interval_step': 1,
      'interval_max': 30,
      'socket_keepalive': True,
  },
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

class WeaviateInsertionError(Exception):
  """
  Custom exception for Weaviate insertion failures.
  
  Attributes:
      message (str): Description of the error.
  """
  def __init__(self, message: str):
    super().__init__(message)
    self.message = message

class ConnectionError(Exception):
  """Custom exception for connection failures for ES and Weaviate."""
  pass

class ConnectionManager:
  def __init__(self):
    self.logger = celery_logger
    self.datastore = None
    self.es_client = None

  def init_conns(self, es_fields: list) -> Tuple[WeaviateDataStore, ESClient]:
    """
    Initialize connections to both Weaviate and Elasticsearch.
    
    Args:
      es_fields: List of fields to retrieve from Elasticsearch
      
    Returns:
      Tuple of (WeaviateDataStore, ESClient)
      
    Raises:
      ConnectionError: If either connection fails
    """
    try:
      # Initialize Weaviate connection
      if not os.getenv("USE_WEAVIATE", "").lower() == 'true':
        raise ConnectionError("USE_WEAVIATE not set to 'true' in .env, this task requires Weaviate")

      weaviate_client = self._setup_weaviate_connection()
      self.datastore = WeaviateDataStore(
        client=weaviate_client, 
        image_embedder=None, 
        text_embedder=None
      )

      if not self.datastore.ping():
        raise ConnectionError("Weaviate is not accessible.")

      # Initialize Elasticsearch connection
      es_config = self._get_es_config()
      self.es_client = ESClient(
        host=es_config['host'],
        port=es_config['port'],
        index_name=es_config['index'],
        fields=es_fields
      )

      if not self.es_client.ping():
        raise ConnectionError("Elasticsearch is not accessible.")

      return self.datastore, self.es_client

    except Exception as e:
      error_msg = f"Connection initialization failed: {str(e)}"
      if "Meta endpoint!" in error_msg:
        error_msg += " Check if the Weaviate cluster is running and accessible."
      self.logger.error(error_msg)
      self._cleanup_connections()
      raise ConnectionError(error_msg)

  def _setup_weaviate_connection(self) -> weaviate.Client:
    """Set up and return Weaviate client based on environment configuration"""
    weaviate_host = os.getenv("WEAVIATE_HOST")
    weaviate_port = int(os.getenv("WEAVIATE_PORT")) if os.getenv("WEAVIATE_PORT") else None

    if weaviate_host and weaviate_port:
      self.logger.info(f'Using local Weaviate: {weaviate_host}:{weaviate_port}')
      return weaviate.connect_to_local(
        host=weaviate_host,
        port=weaviate_port,
        additional_config=AdditionalConfig(
          timeout=Timeout(init=30, query=60, insert=120)
        )
      )
    else:
      wcs_url = os.getenv("WCS_URL")
      wcs_api_key = os.getenv("WCS_API_KEY")
      
      if not (wcs_url and wcs_api_key):
        raise ConnectionError("Neither local nor cloud Weaviate credentials found")
      
      self.logger.info(f'From .env, wcs_url: {wcs_url}, wcs_api_key: {wcs_api_key}')
        
      return weaviate.connect_to_wcs(
        additional_config=AdditionalConfig(
          timeout=Timeout(init=30, query=60, insert=120)
        ),
        cluster_url=wcs_url,
        auth_credentials=weaviate.auth.AuthApiKey(wcs_api_key)
      )

  def _get_es_config(self) -> Dict[str, str]:
    """Get and validate Elasticsearch configuration from environment"""
    required_vars = {
      'host': 'ES_HOST',
      'port': 'ES_PORT',
      'index': 'ES_LISTING_INDEX_NAME'
    }
    
    config = {}
    missing_vars = []
    
    for key, env_var in required_vars.items():
      value = os.getenv(env_var)
      if not value:
        missing_vars.append(env_var)
      config[key] = value
    
    if missing_vars:
      raise ConnectionError(f"Missing required ES environment variables: {', '.join(missing_vars)}")
    
    try:
      config['port'] = int(config['port'])
    except ValueError:
      raise ConnectionError(f"Invalid ES_PORT value: {config['port']}")
    
    return config

  def _cleanup_connections(self):
    """Clean up any existing connections"""
    if self.datastore:
      try:
        self.datastore.close()
      except Exception as e:
        self.logger.warning(f"Error closing Weaviate connection: {e}")
      finally:
        self.datastore = None

    if self.es_client:
      try:
        self.es_client.close()
      except Exception as e:
        self.logger.warning(f"Error closing Elasticsearch connection: {e}")
      finally:
        self.es_client = None


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

def log_run(start_time: datetime, end_time: Optional[datetime], status: str, total_embedding_inserted: int):
  duration = (end_time - start_time).total_seconds()

  run_log_path = Path(RUN_LOG_FILE)
  run_data = {
    'start_time': [start_time],
    'end_time': [end_time],
    'status': [status],
    'duration': [duration],
    'total_embedding_inserted': [total_embedding_inserted]
  }
  run_df = pd.DataFrame(run_data)
  if run_log_path.exists():
    existing_df = pd.read_csv(run_log_path)
    updated_df = pd.concat([existing_df, run_df], ignore_index=True)
  else:
    updated_df = run_df
  updated_df.to_csv(run_log_path, index=False)

def retry_batch_insert(func, max_retries=3, *args, **kwargs) -> Tuple[int, int]:
  """
  Helper function to retry batch insertions.
  Returns:
    Tuple[int, int]: (total_items, failed_count)
  """
  for attempt in range(max_retries):
    total_items, failed_count = func(*args, **kwargs)
    if failed_count == 0:
      return total_items, 0
    celery_logger.warning(f"Batch insert attempt {attempt + 1} had {failed_count} failures, retrying...")
  
  # If we get here, we've exhausted all retries
  celery_logger.error(f"Batch insert failed after {max_retries} attempts")
  return total_items, failed_count


def process_and_batch_insert_to_datastore(embeddings_df: pd.DataFrame, 
                       listingIds: List[str], 
                       datastore: WeaviateDataStore,
                       aux_key: str,
                       listing_df: pd.DataFrame,
                       embedding_type: str = 'I') -> Tuple[int, int]:
  """
  Function to process embeddings and perform operations(add/delete) on Redis or Weaviate datastore.

  Parameters:
  embeddings_df (pd.DataFrame): DataFrame containing the embeddings.
  listingIds (List[Any]): List of listing IDs to be processed.
  datastore (Any): The Redis/Weaviate datastore object where docs are to be added/deleted.
  aux_key (str): The column name in the DataFrame that corresponds to the auxiliary key (image name or remark chunk ID).
                 This can also be thought of as the primary key to auxilliary information.  
  listing_df (pd.DataFrame): DataFrame containing the detail listing data.    

  Note: For image embeddings, remarks are excluded to save space.

  Returns:
    Tuple[int, int]: (total_items_processed, failed_items_count)
  """
  items_to_process = list(embeddings_df.q("listing_id.isin(@listingIds)")[aux_key].values)
  processed_embeddings_df = embeddings_df.q(f"{aux_key}.isin(@items_to_process)")

  _df = join_df(processed_embeddings_df, 
                listing_df, 
                left_on='listing_id', 
                right_on='jumpId', 
                how='left') #.drop(columns=['jumpId'])
  
  # for image embeddings, don't populate remarks (space optimization for weaviate)
  if embedding_type == 'I':
    _df.drop(columns=['remarks'], inplace=True)

  _df.drop(columns=['jumpId'], inplace=True)
  listing_jsons = _df.to_dict(orient='records')
  total_items = len(listing_jsons)

  result = datastore.batch_insert(listing_jsons, 
                         embedding_type=embedding_type, 
                         batch_size=1000, 
                         sleep_time=BATCH_INSERT_SLEEP_TIME)   # do we need this sleep if not for using free weaviate cloud.
  # datastore.batch_upsert(listing_jsons, embedding_type=embedding_type)

  failed_count = len(result['failed_objects'])

  return total_items, failed_count


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
    
    stats["newly_synced_images"] = 0
    stats["still_unsync"] = len(still_unsync_df)
    
    if len(to_be_indexed_embeddings_df) > 0:
      celery_logger.info(f"Begin batch insert {len(to_be_indexed_embeddings_df)} image embeddings to weaviate")
      total_items, failed_count = retry_batch_insert(
        process_and_batch_insert_to_datastore,
        embeddings_df=to_be_indexed_embeddings_df,
        listingIds=to_be_indexed_embeddings_df['listing_id'].unique(),
        datastore=datastore,
        aux_key='image_name',
        listing_df=listing_df,
        embedding_type='I'
      )
      stats["newly_synced_images"] = total_items - failed_count
      if failed_count > 0:
        celery_logger.error(f"Failed to insert {failed_count} image embeddings (unsync)")
      celery_logger.info(f"Ended batch insert {total_items - failed_count} image embeddings to weaviate")

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
        # stats["newly_synced_texts"] = process_and_batch_insert_to_datastore(
        total_items, failed_count = retry_batch_insert(
          process_and_batch_insert_to_datastore,
          embeddings_df=text_embeddings_df,
          listingIds=text_embeddings_df['listing_id'].unique(),
          datastore=datastore,
          aux_key='remark_chunk_id',
          listing_df=listing_df,
          embedding_type='T'
        )
        stats['newly_synced_texts'] = total_items - failed_count
        if failed_count > 0:
          celery_logger.error(f"Failed to insert {failed_count} text embedding (unsync)")
        celery_logger.info(f"Ended batch insert {total_items - failed_count} text embeddings to weaviate")
    
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


def load_image_embeddings(img_cache_folder: Path, logger) -> Tuple[pd.DataFrame, Path]:
  """
  Load image embeddings from either timestamped or non-timestamped file.
  For timestamped files, uses the oldest one.
  Returns (dataframe, file_path) tuple. Both can be None if no valid file found.
  """
  # Try non-timestamped file first
  image_embeddings_file = img_cache_folder / 'image_embeddings_df'
  if image_embeddings_file.exists():
    logger.info(f"Loading image embeddings from {image_embeddings_file}")
    return pd.read_feather(image_embeddings_file), image_embeddings_file
    
  # Try timestamped file
  timestamped_files = list(img_cache_folder.glob('image_embeddings_df.*'))
  if timestamped_files:
    # Take the oldest
    oldest_file = min(timestamped_files, key=lambda x: x.stat().st_mtime)
    logger.info(f"Loading image embeddings from {oldest_file}")
    return pd.read_feather(oldest_file), oldest_file
    
  return None, None


def load_text_embeddings(img_cache_folder: Path, logger) -> Tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
  """
  Load text embeddings and listing df from either timestamped or non-timestamped files.
  For timestamped files, uses the oldest ones.
  Returns (text_df, listing_df, text_file_path, listing_file_path) tuple. Any can be None.

  # TODO: the format without timestamp is considered obsolete, so we should simplify this code.
  """
  # Try non-timestamped files first
  text_embeddings_file = img_cache_folder / 'text_embeddings_df'
  listing_df_file = img_cache_folder / 'listing_df'
  
  if text_embeddings_file.exists() and listing_df_file.exists():
    logger.info(f"Loading text embeddings from {text_embeddings_file}")
    logger.info(f"Loading listing data from {listing_df_file}")
    return pd.read_feather(text_embeddings_file), pd.read_feather(listing_df_file), text_embeddings_file, listing_df_file
    
  # Try timestamped files
  text_timestamped = list(img_cache_folder.glob('text_embeddings_df.*'))
  listing_timestamped = list(img_cache_folder.glob('listing_df.*'))
  
  if text_timestamped and listing_timestamped:
    # Take the oldest ones
    oldest_text = min(text_timestamped, key=lambda x: x.stat().st_mtime)
    oldest_listing = min(listing_timestamped, key=lambda x: x.stat().st_mtime)
    logger.info(f"Loading text embeddings from {oldest_text}")
    logger.info(f"Loading listing data from {oldest_listing}")
    return pd.read_feather(oldest_text), pd.read_feather(oldest_listing), oldest_text, oldest_listing
    
  return None, None, None, None


def send_insert_failure_alert(embedding_stats: Dict[str, int],
                           task_id: str,
                           error_message: Optional[str] = None,
                           embedding_type: str = 'unknown'):
  """
  Send an email alert for failed embedding insertions.

  Args:
    embedding_stats: Dictionary containing embedding/insertion statistics
    task_id: The Celery task ID
    error_message: Optional error message/exception to include in the alert
    embedding_type: Type of embeddings being processed ('I' for image, 'T' for text)
  """
  try:
    _ = load_dotenv(find_dotenv())

    sender_email = os.getenv('VSS_SENDER_EMAIL')
    receiver_emails = os.getenv('VSS_RECEIVER_EMAILS', '')
    if receiver_emails:
      receiver_emails = receiver_emails.split(',')
    email_password = os.getenv('VSS_EMAIL_PASSWORD')

    if not all([sender_email, email_password, receiver_emails]):
      missing_vars = []
      if not sender_email:
        missing_vars.append('VSS_SENDER_EMAIL')
      if not email_password:
        missing_vars.append('VSS_EMAIL_PASSWORD')
      if not receiver_emails:
        missing_vars.append('VSS_RECEIVER_EMAILS')
      
      celery_logger.error(f"Missing required email credentials: {', '.join(missing_vars)}")
      return

    embedding_type_name = 'Image' if embedding_type == 'I' else 'Text' if embedding_type == 'T' else 'N/A or Unknown'
    subject = f"VSS Embed/Index Task Failure Alert - Task ID: {task_id}"

    html_content = f"""
    <html>
    <body>
      <h2>VSS Embed/Index Task Failure Alert</h2>
      <p>Task ID: {task_id}</p>
      <p>Status: <span style="color:red">FAILED</span></p>
      <p>Embedding Type: {embedding_type_name}</p>

      {f'<p>Error: <span style="color:red">{error_message}</span></p>' if error_message else ''}

      <h3>Embedding Statistics:</h3>
      <ul>
        <li>Total Listings Processed: {embedding_stats.get('total_processed', embedding_stats.get('total_listings_processed', 0))}</li>
        <li>Image Embeddings Inserted: {embedding_stats.get('newly_synced_images', embedding_stats.get('image_embeddings_inserted', 0))}</li>
        <li>Text Embeddings Inserted: {embedding_stats.get('newly_synced_texts', embedding_stats.get('text_embeddings_inserted', 0))}</li>
      </ul>

      <p>Please check the logs for more details.</p>
    </body>
    </html>
    """
    
    send_email_alert(subject, html_content, sender_email, receiver_emails, email_password)
  except Exception as e:
    celery_logger.error(f"Failed to send embed/index failure alert: {str(e)}")
    
@celery.task(bind=True, max_retries=3, track_started=True)
def embed_and_index_task(self, 
                        img_cache_folder: str, 
                        es_fields: List[str], 
                        image_batch_size: int, 
                        text_batch_size: int, 
                        num_workers: int
                        ):
  """
    Performs image and text embedding on listing data and indexes them into Weaviate.
    
    Args:
        img_cache_folder: Path to folder containing listing images and data
        es_fields: List of fields to retrieve from Elasticsearch 
        image_batch_size: Batch size for image embedding
        text_batch_size: Batch size for text embedding
        num_workers: Number of worker processes for data loading
    
    The task:
    1. Processes any unsynced image embeddings from previous runs
    2. Embeds images from listing folders
    3. Retrieves listing data from Elasticsearch
    4. Generates text embeddings from listing descriptions
    5. Indexes both image and text embeddings into Weaviate
    6. Backs up processed data and cleans up temporary files
    
    Returns:
        Dict with status, statistics and timing information
  """
  global device, image_embedding_model, text_embedding_model
  if device is None:
    device = torch.device('cuda') if torch.cuda.is_available() else \
             torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
  if image_embedding_model is None:
    image_embedding_model = OpenClipImageEmbeddingModel(model_name='ViT-L-14', pretrained='laion2b_s32b_b82k', device=device)
    
  if text_embedding_model is None:
    text_embedding_model = OpenClipTextEmbeddingModel(embedding_model=image_embedding_model, device=device)

  datastore, es = None, None
  image_embeddings_df, text_embeddings_df, listing_folders = None, None, None
  task_status = "Failed"
  error_message = None
  img_cache_folder = Path(img_cache_folder)
  unsync_folder = img_cache_folder / 'unsync'
  unsync_folder.mkdir(exist_ok=True)

  image_embeddings_file_used, text_embeddings_file_used, listing_df_file_used = None, None, None

  # this file is used to track listing folders that are currently being processed
  # this is used mainly to correctly delete image folders after all processing has been completed successfully
  listing_folders_pickle_file = img_cache_folder / 'listing_folders.pkl'

  # Statistics
  stats = {
    "total_listings_processed": 0,
    "image_embeddings_inserted": 0,
    "text_embeddings_inserted": 0,
    "total_embeddings_inserted": 0
  }

  # Gather environment variables for Weaviate and Elasticsearch
  _ = load_dotenv(find_dotenv())
  task_start_time = datetime.now()

  weaviate_insertion_has_failed = False   # this is empirically the most likely error to occur

  connection_manager = ConnectionManager()  

  try:
    """
    if not (os.getenv("USE_WEAVIATE").lower() == 'true'):
      raise ValueError("USE_WEAVIATE not set to 'true' in .env, this task is only for Weaviate")
    if "ES_HOST" in os.environ and "ES_PORT" in os.environ and "ES_LISTING_INDEX_NAME" in os.environ:
      es_host = os.environ["ES_HOST"]
      es_port = int(os.environ["ES_PORT"])
      listing_index_name = os.environ["ES_LISTING_INDEX_NAME"]
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
      client = weaviate.connect_to_local(host=WEAVIATE_HOST, 
                                         port=WEAVIATE_PORT,
                                         additional_config=AdditionalConfig(timeout=Timeout(init=30, query=60, insert=120)),
                                         )  # TODO: change this before deployment
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
      error_message = "Weaviate is not accessible."
      send_insert_failure_alert(
        embedding_stats=stats,
        task_id=self.request.id,
        error_message=error_message,
        embedding_type='unknown'
      )
      return

    es = ESClient(host=es_host, port=es_port, index_name=listing_index_name, fields=es_fields)
    if not es.ping():
      celery_logger.info('ES is not accessible. Exiting...')
      error_message = "Elasticsearch is not accessible."
      send_insert_failure_alert(
          embedding_stats=stats,
          task_id=self.request.id,
          error_message=error_message,
          embedding_type='unknown'
      )
      return
    """
    
    datastore, es = connection_manager.init_conns(es_fields)

    # process any existing unsynced image embeddings first
    unsync_stats = process_unsync_image_embeddings(img_cache_folder, datastore, es, es_fields, text_batch_size, num_workers)
    celery_logger.info(f"Unsynced processing stats: {unsync_stats}")

    # Check for existing embedding files (if last run has exceptions and failed to complete)
    image_embeddings_df, image_embeddings_file_used = load_image_embeddings(img_cache_folder, celery_logger)
    
    #####################################
    # Begin image embedding inference
    if image_embeddings_df is None:
      # Retrieve listings from cache folder to be processed
      listing_folders = img_cache_folder.lfre(r'^\d+$')
      celery_logger.info(f'Total # of listings in {img_cache_folder}: {len(set(listing_folders))}')

      if len(listing_folders) > MAX_LISTING_TO_EMBED_INDEX:
        celery_logger.info(f'Limiting number of listing folders from {len(listing_folders)} to {MAX_LISTING_TO_EMBED_INDEX}')
        listing_folders = listing_folders[:MAX_LISTING_TO_EMBED_INDEX]

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
    #####################################
    text_embeddings_df, listing_df, text_embeddings_file_used, listing_df_file_used = load_text_embeddings(img_cache_folder, celery_logger)
    
    ####################################
    # Begin text embedding inference
    if text_embeddings_df is None or listing_df is None:
      # Generate embed text
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
        text_embeddings_df = pd.DataFrame(columns=['listing_id', 'remark_chunk_id', 'sentence', 'chunk_start', 'chunk_end', 'embedding'])

      # there can be dups in image_embeddings_df and text_embeddings_df, we keep the latest
      image_embeddings_df.drop_duplicates(subset=['image_name'], keep='last', inplace=True)
      image_embeddings_df.reset_index(drop=True, inplace=True)
      text_embeddings_df.drop_duplicates(subset=['remark_chunk_id'], keep='last', inplace=True)  
      text_embeddings_df.reset_index(drop=True, inplace=True)

    #####################################

    incoming_image_listingIds = set(image_embeddings_df.listing_id.unique())
    incoming_text_listingIds = set(text_embeddings_df.listing_id.unique())

    stats["total_listings_processed"] = len(incoming_image_listingIds.union(incoming_text_listingIds))

    #####################################
    # Begin Batch insert all embeddings
    celery_logger.info(f'Processing {len(incoming_image_listingIds)} listings and {image_embeddings_df.shape[0]} image embeddings')
    
    if len(listing_df) == 0:
      # just save the entire image_embeddings_df for later indexing
      ts = datetime.now().strftime("%Y%m%d%H%M")
      unsync_file = unsync_folder / f'unsync_image_embeddings_df.{ts}'
      image_embeddings_df.to_feather(unsync_file)
      celery_logger.info(f"Saved {len(image_embeddings_df)} unsynced image embeddings to {unsync_file}")

    else:
      # indexed only image embeddings whose listings is in listing_df, saved the otherwise for later indexing
      image_embeddings_df, unsync_df = split_image_embeddings_by_listing_df(image_embeddings_df, listing_df)
      if len(unsync_df) > 0:
        ts = datetime.now().strftime("%Y%m%d%H%M")
        unsync_file = unsync_folder / f'unsync_image_embeddings_df.{ts}'
        unsync_df.to_feather(unsync_file)
        celery_logger.info(f"Saved {len(unsync_df)} unsynced image embeddings to {unsync_file}")

      #####################################
      # Begin Batch insert image embeddings
      celery_logger.info("Begin batch insert image embeddings to weaviate")
      total_items, failed_count = retry_batch_insert(
        process_and_batch_insert_to_datastore,
        embeddings_df=image_embeddings_df, 
        listingIds=image_embeddings_df['listing_id'].unique(),
        datastore=datastore, 
        aux_key='image_name', 
        listing_df=listing_df, 
        embedding_type='I'
      )
      stats["image_embeddings_inserted"] = total_items - failed_count
      if failed_count > 0:
        weaviate_insertion_has_failed = True
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        error_message = f"Failed to insert {failed_count} image embeddings (timestamp: {timestamp})"
        celery_logger.error(error_message)
        send_insert_failure_alert(
          embedding_stats=stats,
          task_id=self.request.id,
          error_message=error_message,
          embedding_type='I'
        )

      celery_logger.info("Ended batch insert image embeddings to weaviate")
      # End Batch insert image embeddings
      #####################################

      #####################################
      # Batch insert text embeddings
      celery_logger.info(f'Processing {len(incoming_text_listingIds)} listings and {text_embeddings_df.shape[0]} text embeddings')
      celery_logger.info("Begin batch insert text embeddings to weaviate")
      total_items, failed_count = retry_batch_insert(
        process_and_batch_insert_to_datastore,
        embeddings_df=text_embeddings_df, 
        listingIds=incoming_text_listingIds, 
        datastore=datastore, 
        aux_key='remark_chunk_id', 
        listing_df=listing_df, 
        embedding_type='T'
      )
      stats["text_embeddings_inserted"] = total_items - failed_count
      if failed_count > 0:
        weaviate_insertion_has_failed = True
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        error_message = f"Failed to insert {failed_count} text embeddings (timestamp: {timestamp})"
        celery_logger.error(error_message)
        send_insert_failure_alert(
          embedding_stats=stats,
          task_id=self.request.id,
          error_message=error_message,
          embedding_type='T'
        )
      celery_logger.info("Ended batch insert text embeddings to weaviate")
      # End Batch insert text embeddings
      #####################################

    if weaviate_insertion_has_failed:
      raise WeaviateInsertionError("One or more embeddings failed to insert into Weaviate.")
    
    set_last_run_time(task_start_time)

    #####################################
    # Delete all processed listing folders
    if listing_folders is None:
      # Try to load from pickle as a last resort
      listing_folders_pickle_file = img_cache_folder / 'listing_folders.pkl'
      if listing_folders_pickle_file.exists():
        try:
          listing_folders = load_from_pickle(listing_folders_pickle_file)
          celery_logger.info(f'Loaded {len(listing_folders)} listing folders from pickle for deletion')
        except Exception as e:
          celery_logger.error(f'Failed to load listing folders from pickle: {str(e)}')

    if listing_folders is not None:
      celery_logger.info(f'Deleting all processed {len(listing_folders)} listing folders')
      for listing_folder in listing_folders:
        try:
          shutil.rmtree(listing_folder)    
        except Exception as e:
          celery_logger.warning(f'Unable to remove {listing_folder}')
      celery_logger.info(f'Deleted all processed {len(listing_folders)} listing folders')
    else:
      celery_logger.warning('No listing folders deletion happens, consider investigate if there are other problems.')
    #####################################

    # Calculate total statistics
    stats["total_embeddings_inserted"] = stats["image_embeddings_inserted"] + stats["text_embeddings_inserted"]

    # Backup
    backup(
      backup_folder=img_cache_folder/'backup',
      image_embeddings_df=image_embeddings_df,
      text_embeddings_df=text_embeddings_df,
      listing_df=listing_df,
      compression='zstd',
      compression_level=3
    )

    task_status = "Completed"

  except ConnectionError as e:
    error_message = str(e)
    send_insert_failure_alert(
      embedding_stats={},
      task_id=self.request.id,
      error_message=error_message,
      embedding_type='N/A'
    )

    task_status = "Failed"

  except (Exception, WeaviateInsertionError) as e:
    error_message = str(e)
    if isinstance(e, WeaviateInsertionError):
      celery_logger.error(f"WeaviateInsertionError: {error_message}")
    else:
      celery_logger.error(f"Error: {error_message}")
    
    # if there's an error, save the image/text embeddings and listing_df
    timestamp = datetime.now().strftime("%Y%m%d%H%M")

    if image_embeddings_df is not None and not image_embeddings_df.empty:
      image_embeddings_df.to_feather(img_cache_folder / f'image_embeddings_df.{timestamp}')
    if text_embeddings_df is not None and not text_embeddings_df.empty:
      text_embeddings_df.to_feather(img_cache_folder / f'text_embeddings_df.{timestamp}')
    if listing_df is not None and not listing_df.empty:
      listing_df.to_feather(img_cache_folder / f'listing_df.{timestamp}')

    send_insert_failure_alert(
      embedding_stats=stats or {},  # ensure stats is not None
      task_id=self.request.id,
      error_message=f"General Error: {error_message} (Timestamp: {timestamp})"
    )

    task_status = "Failed"

  finally:
    # close all connections
    connection_manager._cleanup_connections()

    # allow gc to collect dataframes
    del image_embeddings_df
    del text_embeddings_df
    gc.collect()

    # empty GPU cache
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
      log_run(task_start_time, task_end_time, task_status, stats['total_embeddings_inserted'])

    if task_status == "Completed":
      # Remove the temporary files
      files_to_cleanup = [f for f in [image_embeddings_file_used, text_embeddings_file_used, listing_df_file_used, listing_folders_pickle_file
      ] if f is not None]

      for file in files_to_cleanup:
        if file.exists():
          try:
            file.unlink()
            celery_logger.info(f"Deleted {file}")
          except Exception as e:
            celery_logger.warning(f"Failed to delete {file}: {str(e)}")
      return {"status": "Completed", 
              "message": "Embedding and indexing completed successfully",              
              "stats": stats,
              "start_time": task_start_time.strftime("%Y-%m-%d %H:%M:%S"),
              "end_time": task_end_time.strftime("%Y-%m-%d %H:%M:%S")
              }
    else:
      send_insert_failure_alert(
        embedding_stats={},
        task_id=self.request.id,
        error_message=error_message or "Unknown, please check logs or other email alerts",
        embedding_type='unknown'
      )
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


from typing import Optional, Dict, Tuple, List
import os, time, gc, json
from datetime import datetime, timedelta

from celery import Celery
from celery.utils.log import get_task_logger
from celery.exceptions import SoftTimeLimitExceeded
from pathlib import Path

import realestate_core.common.class_extensions

celery = Celery('delete_inactive_app', broker='pyamqp://guest@localhost//')

# Set log level for the Celery app
celery.conf.update(
  task_acks_late=True,
  worker_cancel_long_running_tasks_on_connection_loss=True,
  task_acks_on_failure_or_timeout=True,
  task_reject_on_worker_lost=True,
  worker_log_format="%(levelname)s: %(message)s",
  worker_task_log_format="%(levelname)s: %(message)s",
  worker_redirect_stdouts_level='INFO',

  # configs for reliability
  broker_heartbeat=60,                    # More frequent heartbeats
  broker_connection_timeout=300,           # Longer connection timeout
  worker_prefetch_multiplier=1,           # Process one task at a time
  worker_concurrency=1,                   # Single worker process
  task_time_limit=300,                  # 5 min max task runtime
  task_soft_time_limit=280,             # 4 min 40 sec soft time limit
  worker_max_tasks_per_child=1,          # Restart worker after each task
  broker_pool_limit=None,                # Don't limit connection pool

  # TODO: need some testing to see if this reduces enough memory and not sacrifice too much throughput 
  # worker_prefetch_multiplier=1,     # Reduce message prefetching

  # Settings to handle clock drift
  enable_utc=True,                       # Use UTC timestamps
  timezone='UTC',                        # Set timezone to UTC
  worker_timer_precision=1,              # 1 second timer precision
  broker_transport_options={
    'retry_on_timeout': True,
    'interval_start': 0,
    'interval_step': 1,
    'interval_max': 30,
  }
)

celery_logger = get_task_logger(__name__)
celery_logger.setLevel('INFO')

LAST_RUN_FILE = 'celery_delete_inactive.last_run.log'
DELETE_LOG_FILE = 'celery_delete_inactive.run_log.csv'
SHARD_FILE_PREFIX = 'listings_to_be_deleted'   # fiels that store the list of listings to be deleted
MAX_LISTINGS_PER_SESSION = 1000

def get_last_delete_time() -> Optional[datetime]:
  try:
    with open(LAST_RUN_FILE, 'r') as f:
      last_run_str = f.read().strip()
      return datetime.fromisoformat(last_run_str)
  except FileNotFoundError:
    return None
  except ValueError:
    return None
  
def set_last_delete_time(a_datetime: datetime):
  with open(LAST_RUN_FILE, 'w') as f:
    f.write(a_datetime.isoformat())  

def log_delete_run(start_time: datetime, end_time: Optional[datetime], status: str, stats: Dict):
  import pandas as pd
  
  duration = (end_time - start_time).total_seconds()
  
  log_path = Path(DELETE_LOG_FILE)
  log_data = {
    'start_time': [start_time],
    'end_time': [end_time],
    'status': [status],
    'duration': [duration],
    'listings_deleted': [stats.get('total_listings_deleted', 0)],
    'embeddings_deleted': [stats.get('total_embeddings_deleted', 0)]
  }
  
  log_df = pd.DataFrame(log_data)
  if log_path.exists():
    existing_df = pd.read_csv(log_path)
    updated_df = pd.concat([existing_df, log_df], ignore_index=True)
  else:
    updated_df = log_df
  
  updated_df.to_csv(log_path, index=False)    

def retry_batch_delete(func, max_retries=3, *args, **kwargs) -> Tuple[int, Dict]:
  """
  Helper function to retry batch deletions.
  
  Args:
    func: The deletion function to retry
    max_retries: Maximum number of retry attempts
    *args, **kwargs: Arguments to pass to the deletion function
  
  Returns:
    Tuple[int, Dict]: (total_objects_deleted, failed_batches_info)
  """
  for attempt in range(max_retries):
    deletion_stats = func(*args, **kwargs)
    if deletion_stats.get('error_count', 0) == 0 and 'fatal_error' not in deletion_stats:
      return deletion_stats['total_objects_deleted'], {}
    
    celery_logger.warning(
      f"Batch deletion attempt {attempt + 1} had {deletion_stats.get('error_count', 0)} failures, "
      f"retrying in {2 ** attempt} seconds..."
    )
    time.sleep(2 ** attempt)  # Exponential backoff
  
  # If we get here, we've exhausted all retries
  celery_logger.error(f"Batch deletion failed after {max_retries} attempts")
  return deletion_stats.get('total_objects_deleted', 0), deletion_stats.get('failed_batches', [])

# Helper functions to handle shard files
def get_shard_files(base_folder: Path) -> List[Path]:
  """Return shard files sorted by timestamp and number"""
  pattern = f"{SHARD_FILE_PREFIX}.*"
  return sorted(base_folder.lf(pattern))

def save_shard_file(listing_ids: List[str], shard_path: Path):
  """Save listing IDs to a shard file"""
  with open(shard_path, 'w') as f:
    json.dump(listing_ids, f)

def load_shard_file(shard_path: Path) -> List[str]:
  """Load listing IDs from a shard file"""
  with open(shard_path, 'r') as f:
    return json.load(f)



@celery.task(bind=True, max_retries=3, acks_late=False, track_started=True)
def delete_inactive_listings_task(self, img_cache_folder: str, batch_size=20, sleep=0.5):
  """
  Task to delete inactive, delisted, or sold listings from Weaviate based on BigQuery data.
  
  Args:
    batch_size: Number of listings to delete in each batch
    sleep_time: Sleep time between batches in seconds
  """
  
   # Lazy imports
  import weaviate
  from weaviate.classes.init import AdditionalConfig, Timeout
  from realestate_vss.data.weaviate_datastore import WeaviateDataStore_v4 as WeaviateDataStore
  from realestate_analytics.data.bq import BigQueryDatastore
  from dotenv import load_dotenv, find_dotenv

  task_start_time = datetime.now()
  datastore, bq_datastore = None, None
  task_status = 'Failed'
  error_message = None

  stats = {
    "total_listings_deleted": 0,
    "total_embeddings_deleted": 0,
    "failed_listings": []
  }

  try:
    _ = load_dotenv(find_dotenv())
    if not (os.getenv("USE_WEAVIATE").lower() == 'true'):
      raise ValueError("USE_WEAVIATE not set to 'true' in .env")
    
    # Initialize Weaviate client
    WEAVIATE_HOST = os.getenv("WEAVIATE_HOST")
    WEAVIATE_PORT = int(os.getenv("WEAVIATE_PORT")) if os.getenv("WEAVIATE_PORT") is not None else None
    if WEAVIATE_HOST is not None and WEAVIATE_PORT is not None:
      celery_logger.info(f'Using local Weaviate: {WEAVIATE_HOST}:{WEAVIATE_PORT}')
      client = weaviate.connect_to_local(host=WEAVIATE_HOST, 
                                         port=WEAVIATE_PORT,
                                         additional_config=AdditionalConfig(timeout=Timeout(init=30, query=60, insert=120)))
    else:
      WCS_URL = os.getenv("WCS_URL")
      WCS_API_KEY = os.getenv("WCS_API_KEY")

      if WCS_URL is None or WCS_API_KEY is None:
        raise ValueError("WCS_URL and WCS_API_KEY not found in .env")

      client = weaviate.connect_to_wcs(
        additional_config=AdditionalConfig(timeout=Timeout(init=30, query=60, insert=120)),
        cluster_url=WCS_URL,
        auth_credentials=weaviate.auth.AuthApiKey(WCS_API_KEY)
      )
    
    datastore = WeaviateDataStore(client=client, image_embedder=None, text_embedder=None)
    if not datastore.ping():
      raise Exception('Weaviate not reachable')
    
    # Initialize things to track deleting listings in shard of MAX_LISTINGS_PER_SESSION (1000)
    session_delete_count = 0
    shard_folder = Path(img_cache_folder)/'delete_shards'
    shard_folder.mkdir(exist_ok=True)

    # first: Process existing shards
    while session_delete_count < MAX_LISTINGS_PER_SESSION:
      shard_files = get_shard_files(shard_folder)
      if not shard_files:
        break

      oldest_shard = shard_files[0]
      listing_ids = load_shard_file(oldest_shard)
      celery_logger.info(f'Processing shard {oldest_shard.name} with {len(listing_ids)} listings')

      embeddings_deleted, failed_batches = retry_batch_delete(
        datastore.delete_listings_by_batch,
        listing_ids=listing_ids,
        batch_size=batch_size,
        sleep_time=sleep
      )

      if failed_batches:
        # Collect failed listing IDs
        failed_listing_ids = set()
        for batch in failed_batches:
          failed_listing_ids.update(batch['listing_ids'])
        stats['failed_listings'] = list(failed_listing_ids)

        error_message = f"Failed to delete {len(failed_listing_ids)} listings after retries"
        celery_logger.error(error_message)

        # If we have task-level retries left, retry the whole task
        if self.request.retries < self.max_retries:
          raise self.retry(
            exc=Exception(error_message),
            countdown=300 * (self.request.retries + 1)
          )
        break
      else:
        session_delete_count += len(listing_ids)
        stats['total_listings_deleted'] += len(listing_ids)
        stats['total_embeddings_deleted'] += embeddings_deleted
        oldest_shard.unlink()
        celery_logger.info(f'Successfully processed and removed shard {oldest_shard.name}')

    # 2nd: Always check BQ for new deletions
    last_run = get_last_delete_time()
    bq_query_start_time = datetime.now()
    celery_logger.info("This is the first run." if last_run is None else f"Last run: {last_run}")
    celery_logger.info(f"Last run was {last_run}")

    if last_run is not None:
      # Get deleted listings from BigQuery
      bq_datastore = BigQueryDatastore()
      deleted_listings_df = bq_datastore.get_deleted_listings(start_time=last_run)

      if len(deleted_listings_df) > 0:
        deleted_listing_ids = deleted_listings_df['listingId'].unique().tolist()
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        # celery_logger.info(f'Removing {len(deleted_listing_ids)} deleted listings from Weaviate since {last_run}')
        celery_logger.info(f'Found {len(deleted_listing_ids)} new deleted listings from BQ')

        for i, chunk_start in enumerate(range(0, len(deleted_listing_ids), 1000)):
          chunk = deleted_listing_ids[chunk_start:chunk_start + 1000]
          shard_path = shard_folder / f"{SHARD_FILE_PREFIX}.{timestamp}.{i}"
          save_shard_file(chunk, shard_path)
          celery_logger.info(f'Created shard {shard_path.name} with {len(chunk)} listings')
        
      else:
        celery_logger.info('No deleted listings found since last run.')

    set_last_delete_time(bq_query_start_time)
    celery_logger.info(f'Setting last run to {bq_query_start_time}')
    task_status = 'Completed'

  except SoftTimeLimitExceeded:
    celery_logger.warning("Task approaching time limit, attempting graceful shutdown")
    error_message = "Soft time limit exceeded (55 minutes)"
    task_status = 'Failed'
    raise

  except Exception as e:
    celery_logger.error(f"Error: {str(e)}")
    error_message = str(e)
    task_status = 'Failed'
    raise

  finally:
    if datastore:
      datastore.close()
    if bq_datastore:
      bq_datastore.close()

    gc.collect()

    task_end_time = datetime.now()
    log_delete_run(task_start_time, task_end_time, task_status, stats)

    if task_status == "Completed":
      # self.request.acknowledge()
      return {
        "status": "Completed",
        "message": "Deletion of inactive listings completed successfully",
        "stats": stats,
        "start_time": task_start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": task_end_time.strftime("%Y-%m-%d %H:%M:%S")
      }
    else:
      return {
        "status": "Failed",
        "error": error_message,
        "start_time": task_start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": task_end_time.strftime("%Y-%m-%d %H:%M:%S")
      }
    


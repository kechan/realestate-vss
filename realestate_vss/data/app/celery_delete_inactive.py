from typing import Optional, List
import os, gc
from datetime import datetime, timedelta

from celery import Celery
from celery.utils.log import get_task_logger
from celery.exceptions import SoftTimeLimitExceeded
from pathlib import Path

import weaviate
from weaviate.classes.init import AdditionalConfig, Timeout

import torch
from dotenv import load_dotenv, find_dotenv

from realestate_vss.data.weaviate_datastore import WeaviateDataStore_v4 as WeaviateDataStore
from realestate_analytics.data.bq import BigQueryDatastore

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
  broker_heartbeat=10,                    # More frequent heartbeats
  broker_connection_timeout=30,           # Longer connection timeout
  worker_prefetch_multiplier=1,           # Process one task at a time
  worker_concurrency=1,                   # Single worker process
  task_time_limit=3600,                  # 1 hour max task runtime
  task_soft_time_limit=3300,             # 55 minutes soft limit
  worker_max_tasks_per_child=1,          # Restart worker after each task
  broker_pool_limit=None,                # Don't limit connection pool

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

def get_last_delete_time():
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

def log_delete_run(start_time: datetime, end_time: Optional[datetime], status: str, stats: dict):
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

@celery.task(bind=True, max_retries=3)
def delete_inactive_listings_task(self, batch_size=20, sleep=0.5):
  """
  Task to delete inactive, delisted, or sold listings from Weaviate based on BigQuery data.
  
  Args:
    batch_size: Number of listings to delete in each batch
    sleep_time: Sleep time between batches in seconds
  """

  datastore, bq_datastore = None, None
  task_status = 'Failed'
  error_message = None
  task_start_time = datetime.now()

  stats = {
    "total_listings_deleted": 0,
    "total_embeddings_deleted": 0
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
      client = weaviate.connect_to_local(WEAVIATE_HOST, WEAVIATE_PORT)
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
    
    last_run = get_last_delete_time()
    celery_logger.info("This is the first run." if last_run is None else f"Last run: {last_run}")

    if last_run is not None:
      # Get deleted listings from BigQuery
      bq_datastore = BigQueryDatastore()
      deleted_listings_df = bq_datastore.get_deleted_listings(start_time=last_run)

      if len(deleted_listings_df) > 0:
        count_b4_del = datastore.count_all()
        deleted_listing_ids = deleted_listings_df['listingId'].unique().tolist()

        celery_logger.info(f'Removing {len(deleted_listing_ids)} deleted listings from Weaviate since {last_run}')

        datastore.delete_listings_by_batch(
          listing_ids=deleted_listing_ids,
          batch_size=batch_size,
          sleep_time=sleep
        )

        count_after_del = datastore.count_all()

        stats['total_listings_deleted'] = len(deleted_listing_ids)
        stats['total_embeddings_deleted'] = count_b4_del - count_after_del

        celery_logger.info(f'Successfully removed {stats["total_listings_deleted"]} listings '
                          f'({stats["total_embeddings_deleted"]} embeddings)')
        
      else:
        celery_logger.info('No deleted listings found since last run.')

    set_last_delete_time(task_start_time)
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

    task_end_time = datetime.now()
    log_delete_run(task_start_time, task_end_time, task_status, stats)

    if task_status == "Completed":
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
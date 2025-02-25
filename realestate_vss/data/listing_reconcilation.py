from typing import List, Set, Dict, Any, Optional, Iterator, Union
import logging, gc, json
from pathlib import Path
from datetime import datetime
import time

import weaviate
from weaviate.classes.query import Filter

from .weaviate_datastore import WeaviateDataStore_v4 as WeaviateDataStore
from .es_client import ESClient

class ListingReconciliation:
  def __init__(
    self, 
    weaviate_datastore: WeaviateDataStore,
    es_client: ESClient,
    batch_size: int = 500,
    max_batches: Optional[int] = None,
    sleep_time: float = 1.0,
    skip_deletion: bool = False,  # Option to skip actual deletion
    es_snapshot_file: Optional[Union[str, Path]] = None,  # File to store ES query results
    use_snapshot_only=False
  ):
    self.datastore = weaviate_datastore
    self.es_client = es_client
    self.batch_size = batch_size
    self.max_batches = max_batches
    self.sleep_time = sleep_time
    self.skip_deletion = skip_deletion
    self.es_snapshot_file = es_snapshot_file
    self.use_snapshot_only = use_snapshot_only
    self.logger = logging.getLogger(__name__)

  def get_all_weaviate_listing_ids(self) -> Iterator[str]:
    """
    Yield listing IDs from Weaviate in an iterator instead of keeping all in memory.
    """
    image_collection = self.datastore.client.collections.get("Listing_Image")
    for obj in image_collection.iterator():
      yield obj.properties.get('listing_id')

  def load_or_fetch_active_listings(self, listing_ids: List[str]) -> Set[str]:
    """
    Either query ES for active listings for the current batch (and update the snapshot file)
    or, if use_snapshot_only is True, simply load the active listings from the snapshot file.
    In either case, only the current batch's active listings are returned.
    """
    if self.use_snapshot_only:
      # Read the snapshot file and return its contents.
      if self.es_snapshot_file is None: 
        raise ValueError("Snapshot file path is required in snapshot-only mode!")
      try:
        with open(self.es_snapshot_file, 'r') as f:
          active_listing_ids = set(json.load(f))
        self.logger.info("Loaded active listings from snapshot file (snapshot-only mode).")
      except (FileNotFoundError, json.JSONDecodeError):
        self.logger.warning("Snapshot file not found or corrupted in snapshot-only mode!")
        active_listing_ids = set()
      return active_listing_ids

    # Otherwise, query ES for active listings for the current batch.
    active_listings = self.es_client.get_active_listings(listing_ids)
    active_listing_ids = {doc['jumpId'] for doc in active_listings}

    if self.es_snapshot_file:
      try:
        # Open the snapshot file in read/write mode to update it.
        with open(self.es_snapshot_file, 'r+') as f:
          try:
            existing_data = json.load(f)
          except (FileNotFoundError, json.JSONDecodeError):
            existing_data = []
          # Merge the previous active listing IDs with the current batch results.
          updated_data = sorted(set(existing_data) | active_listing_ids)
          f.seek(0)
          f.truncate()
          json.dump(updated_data, f)
      except FileNotFoundError:
        # If the file does not exist, create it and write the current active IDs.
        with open(self.es_snapshot_file, 'w') as f:
          json.dump(sorted(active_listing_ids), f)

    # Return only the current batch's active listings.
    return active_listing_ids


  def process_batch(self, listing_ids: List[str]) -> Dict[str, Any]:
    """
    Process a batch of listing IDs:
    1. Query ES for these IDs (or load from file)
    2. Identify which ones need deletion
    3. Delete from Weaviate if deletion is not skipped
    """
    stats = {
      "processed": len(listing_ids),
      "to_delete": 0,
      "deleted": 0,
      "errors": 0
    }

    # Get active listings (either from ES or file)
    active_listing_ids = self.load_or_fetch_active_listings(listing_ids)

    # Identify listings to delete
    to_delete = list(filter(lambda x: x not in active_listing_ids, listing_ids))
    stats["to_delete"] = len(to_delete)

    if not to_delete or self.skip_deletion:
      return stats

    try:
      # Reduce batch size dynamically
      batch_size = min(10, len(to_delete))  # Small safe batch
      deletion_stats = self.datastore.delete_listings_by_batch(
        listing_ids=to_delete,
        batch_size=batch_size,
        sleep_time=self.sleep_time
      )
      
      stats["deleted"] = deletion_stats.get("total_objects_deleted", 0)
      stats["errors"] = deletion_stats.get("error_count", 0)

    except Exception as e:
      self.logger.error(f"Error deleting listings: {str(e)}")
      stats["errors"] = len(to_delete)

    # Force garbage collection after deletion
    del to_delete
    gc.collect()

    return stats

  def reconcile(self, max_listings: Optional[int] = None) -> Dict[str, Any]:
    """
    Main reconciliation process with an optional limit on the number of batches.
    """
    start_time = datetime.now()
    total_stats = {
      "total_processed": 0,
      "total_objects_deleted": 0,
      "total_errors": 0,
      "batches_processed": 0
    }

    try:
      # Process listings as an iterator to reduce memory pressure
      listing_id_iterator = self.get_all_weaviate_listing_ids()
      
      batch = []
      for i, listing_id in enumerate(listing_id_iterator):
        if max_listings and i >= max_listings:
          break
        batch.append(listing_id)

        if len(batch) >= self.batch_size:
          batch_stats = self.process_batch(batch)
          
          # Update total stats
          total_stats["total_processed"] += batch_stats["processed"]
          total_stats["total_objects_deleted"] += batch_stats["deleted"]
          total_stats["total_errors"] += batch_stats["errors"]
          total_stats["batches_processed"] += 1

          self.logger.info(
            f"Processed batch {total_stats['batches_processed']}: "
            f"{batch_stats['deleted']} deleted, {batch_stats['errors']} errors"
          )

          batch.clear()  # Clear memory
          time.sleep(self.sleep_time)

          # Stop processing if max_batches limit is reached
          if self.max_batches and total_stats["batches_processed"] >= self.max_batches:
            break

      # Process remaining batch if within batch limit
      if batch and (not self.max_batches or total_stats["batches_processed"] < self.max_batches):
        batch_stats = self.process_batch(batch)
        total_stats["total_processed"] += batch_stats["processed"]
        total_stats["total_objects_deleted"] += batch_stats["deleted"]
        total_stats["total_errors"] += batch_stats["errors"]
        total_stats["batches_processed"] += 1

      total_stats["duration_seconds"] = (datetime.now() - start_time).total_seconds()

    except Exception as e:
      self.logger.error(f"Reconciliation failed: {str(e)}")
      total_stats["error"] = str(e)

    return total_stats

# Example usage:
# reconciler = ListingReconciliation(weaviate_datastore, es_client)
# stats = reconciler.reconcile(max_listings=10000)  # Process first 10k listings
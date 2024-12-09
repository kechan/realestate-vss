from typing import List, Set, Dict, Any, Optional
import logging
from datetime import datetime
import time

import weaviate
from weaviate.classes.query import Filter

from .weaviate_datastore import WeaviateDataStore_v4 as WeaviateDataStore
from .es_client import ESClient

class ListingReconciliation:
  def __init__(self, 
    weaviate_datastore: WeaviateDataStore,
    es_client: ESClient,
    batch_size: int = 1000,
    sleep_time: float = 0.5
  ):
    self.datastore = weaviate_datastore
    self.es_client = es_client
    self.batch_size = batch_size
    self.sleep_time = sleep_time
    self.logger = logging.getLogger(__name__)

  def get_all_weaviate_listing_ids(self) -> Set[str]:
    """
    Get all unique listing IDs from Listing_Image collection in Weaviate.
    Uses collection iterator to handle large datasets efficiently.
    
    Note: We only need to check Listing_Image since text embeddings are always
    generated based on the listings found in the image collection during the
    embed_and_index process.

    It took about 8.5 minutes to return 58941 listings.
    """
    listing_ids = set()
    image_collection = self.datastore.client.collections.get("Listing_Image")
    
    for obj in image_collection.iterator():
      listing_ids.add(obj.properties.get('listing_id'))
        
    return listing_ids

  def process_batch(self, listing_ids: List[str]) -> Dict[str, Any]:
    """
    Process a batch of listing IDs:
    1. Query ES for these IDs
    2. Identify which ones need deletion
    3. Delete from Weaviate
    """
    stats = {
      "processed": len(listing_ids),
      "to_delete": 0,
      "deleted": 0,
      "errors": 0
    }

    # Get active listings from ES
    active_listings = self.es_client.get_active_listings(listing_ids)
    active_listing_ids = {doc['jumpId'] for doc in active_listings}

    # Identify listings to delete (not in ES or not active)
    to_delete = set(listing_ids) - active_listing_ids
    stats["to_delete"] = len(to_delete)

    if not to_delete:
      return stats

    try:
      # Delete listings from Weaviate
      deletion_stats = self.datastore.delete_listings_by_batch(
        listing_ids=list(to_delete),
        batch_size=min(10, len(to_delete)),  # Smaller batches for deletion
        sleep_time=self.sleep_time
      )
      
      stats["deleted"] = deletion_stats.get("total_objects_deleted", 0)
      stats["errors"] = deletion_stats.get("error_count", 0)

    except Exception as e:
      self.logger.error(f"Error deleting listings: {str(e)}")
      stats["errors"] = len(to_delete)

    return stats

  def reconcile(self, max_listings: Optional[int] = None) -> Dict[str, Any]:
    """
    Main reconciliation process.
    
    Args:
      max_listings: Optional limit on number of listings to process
    """
    start_time = datetime.now()
    total_stats = {
      "total_listings": 0,
      "total_processed": 0,
      "total_objects_deleted": 0,
      "total_errors": 0,
      "batches_processed": 0
    }

    try:
      # Get all listing IDs from Weaviate
      all_listing_ids = list(self.get_all_weaviate_listing_ids())
      total_stats["total_listings"] = len(all_listing_ids)

      if max_listings:
        all_listing_ids = all_listing_ids[:max_listings]

      # Process in batches
      for i in range(0, len(all_listing_ids), self.batch_size):
        batch = all_listing_ids[i:i + self.batch_size]
        
        batch_stats = self.process_batch(batch)
        
        # Update total stats
        total_stats["total_processed"] += batch_stats["processed"]
        total_stats["total_objects_deleted"] += batch_stats["deleted"]
        total_stats["total_errors"] += batch_stats["errors"]
        total_stats["batches_processed"] += 1

        # Log progress
        self.logger.info(
          f"Processed batch {total_stats['batches_processed']}: "
          f"{batch_stats['deleted']} deleted, "
          f"{batch_stats['errors']} errors"
        )

        time.sleep(self.sleep_time)  # Prevent overwhelming the services

    except Exception as e:
      self.logger.error(f"Reconciliation failed: {str(e)}")
      total_stats["error"] = str(e)

    total_stats["duration_seconds"] = (datetime.now() - start_time).total_seconds()
    
    return total_stats

# Example usage:
# reconciler = ListingReconciliation(weaviate_datastore, es_client)
# stats = reconciler.reconcile(max_listings=10000)  # Process first 10k listings
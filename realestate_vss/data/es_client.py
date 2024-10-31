from typing import List, Set, Union

from elasticsearch import Elasticsearch
from elasticsearch.exceptions import NotFoundError
from elasticsearch.helpers import scan

class ESClient:
  def __init__(self, host: str, port: int, index_name: str, fields: List[str] = None):
    self.es = Elasticsearch([f'http://{host}:{port}/'])
    self.index_name = index_name
    self.fields = fields    # desired top level key/value in json to return

  def ping(self) -> bool:
    return self.es.ping()

  def get_listing(self, listing_id: str):
    try:
      return self.es.get(index=self.index_name, id=listing_id)['_source']
    except NotFoundError:
      return
    
  def get_active_listings(self, listingIds: List[str]) -> List[dict]:
    if len(listingIds) == 0: return []
      
    query = {
      "query": {
          "bool": {
              "must": [
                  {"terms": {"_id": listingIds}},
                  {"match": {"listingStatus": "ACTIVE"}}
              ]
          }
      },
      "_source": self.fields if self.fields else ['jumpId']
    }
    
    listing_docs = scan(self.es, index=self.index_name, query=query)
    return [doc['_source'] for doc in listing_docs]

  def search_listings(self, query: dict):
    return self.es.search(index=self.index_name, body=query)


  def get_inactive_or_absent(self, listing_ids: Union[Set[str], List[str]], batch_size: int = 1000) -> Set[str]:
    """
    Efficiently get IDs of listings that are either inactive or absent from the index.
    
    Optimized for:
    - Minimal data transfer (only fetches IDs, no source data)
    - Single query to Elasticsearch
    - Efficient memory usage with sets
    - Batched processing for large result sets
    
    Args:
        listing_ids: Set or List of listing IDs to check
        batch_size: Number of results to process per batch
        
    Returns:
        Set of listing IDs that are either inactive or absent
    """
    # Convert to set if input is list, use directly if already a set
    # This is efficient as Python's set constructor will use the iterator protocol
    inactive_or_absent = set(listing_ids)
    
    if not inactive_or_absent:
      return inactive_or_absent
    
    # Query to find only ACTIVE listings from our input set
    query = {
      "query": {
        "bool": {
          "must": [
            {
              "terms": {
                "_id": list(inactive_or_absent)  # ES requires list for terms query
              }
            },
            {
              "term": {
                "listingStatus.keyword": "ACTIVE"
              }
            }
          ]
        }
      },
      "_source": False  # Don't fetch source data, we only need IDs
    }

    try:
      # Use scan for efficient pagination of large result sets
      # Only fetching IDs of active listings
      for hit in scan(
        client=self.es,
        query=query,
        index=self.index_name,
        size=batch_size,
        _source=False
      ):
        if '_id' in hit:
          inactive_or_absent.discard(hit['_id'])

    except Exception as e:
      raise Exception(f"Error checking listing statuses: {str(e)}")

    return inactive_or_absent
  
  def close(self):
    self.es.close()
  
  # def get_listing_ids(self):
  #   try:
  #     for hit in scan(self.es, index=self.index_name, query={"query": {"match_all": {}}}):
  #       yield hit['_id']
  #   except NotFoundError:
  #     return

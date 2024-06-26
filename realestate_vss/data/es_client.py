from typing import List

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

  # def get_listing_ids(self):
  #   try:
  #     for hit in scan(self.es, index=self.index_name, query={"query": {"match_all": {}}}):
  #       yield hit['_id']
  #   except NotFoundError:
  #     return

from typing import Any, Tuple, List, Dict, Optional, Union, Iterable
import weaviate, uuid, math, gc
from PIL import Image
from datetime import datetime
from dateutil import parser

import numpy as np
import pandas as pd

from tqdm.auto import tqdm

from realestate_core.common.utils import join_df
from realestate_vision.common.utils import get_listingId_from_image_name
from ..utils.obsolete import WeaviateDataStore
from ..data.index import FaissIndex

class WeaviateDataStore_v3(WeaviateDataStore):
  def __init__(self, 
               client: weaviate.client.Client,
               image_embedder,
               text_embedder,
               score_aggregation_method = 'max'
               ):
    super().__init__(image_embedder, text_embedder, score_aggregation_method)
    self.client = client

    try:
      self.create_schema()
    except weaviate.exceptions.UnexpectedStatusCodeException as e:
      if e.status_code == 422:
        print('Schema already exists')
      else:
        raise e

  def create_schema(self):
    # Object corresponding to a vector from an image of a listing
    listing_image_schema = {
      "class": "Listing_Image",
      "vectorizer": "none",
      "properties": [
        {"name": "listing_id", "dataType": ["string"], "indexInverted": True},
        {"name": "city", "dataType": ["string"], "indexInverted": True},
        {"name": "provState", "dataType": ["string"], "indexInverted": True},
        {"name": "postalCode", "dataType": ["string"]},
        {"name": "lat", "dataType": ["number"]},
        {"name": "lng", "dataType": ["number"]},
        {"name": "streetName", "dataType": ["string"]},
        {"name": "beds", "dataType": ["string"]},
        {"name": "bedsInt", "dataType": ["number"], "indexInverted": True},
        {"name": "baths", "dataType": ["string"]},
        {"name": "bathsInt", "dataType": ["number"], "indexInverted": True},
        {"name": "sizeInterior", "dataType": ["string"]},
        {"name": "sizeInteriorUOM", "dataType": ["string"]},
        {"name": "lotSize", "dataType": ["string"]},
        {"name": "lotUOM", "dataType": ["string"]},
        {"name": "propertyFeatures", "dataType": ["string"]},
        {"name": "propertyType", "dataType": ["string"]},
        {"name": "transactionType", "dataType": ["string"]},
        {"name": "carriageTrade", "dataType": ["boolean"]},
        {"name": "price", "dataType": ["number"]},
        {"name": "leasePrice", "dataType": ["number"]},
        {"name": "pool", "dataType": ["boolean"]},
        {"name": "garage", "dataType": ["boolean"]},
        {"name": "waterFront", "dataType": ["boolean"]},
        {"name": "fireplace", "dataType": ["boolean"]},
        {"name": "ac", "dataType": ["boolean"]},
        {"name": "remarks", "dataType": ["string"]},
        {"name": "photo", "dataType": ["string"]},
        {"name": "listingDate", "dataType": ["date"]},
        {"name": "lastUpdate", "dataType": ["date"]},
        {"name": "lastPhotoUpdate", "dataType": ["date"]},
        {"name": "image_name", "dataType": ["string"]}
      ],
      "vectorIndexType": "hnsw",
      "vectorIndexConfig": {
          "skip": False,
          "cleanupIntervalSeconds": 300,
          "pq": {"enabled": False},
          "maxConnections": 64,
          "efConstruction": 128,
          "ef": -1,
          "dynamicEfMin": 100,
          "dynamicEfMax": 500,
          "dynamicEfFactor": 8,
          "vectorCacheMaxObjects": 2000000,
          "flatSearchCutoff": 40000,
          "distance": "cosine"
      },
      "invertedIndexConfig": {
        "stopwords": {"additions": [], "preset": "none", "removals": []}
      }
    }

    # Object corresponding to a text embedding of a listing, reuse same properties except replace image_name with remark_chunk_id
    listing_text_schema = {
      "class": "Listing_Text",
      "vectorizer": "none",
      "properties": [
        prop if prop['name'] != 'image_name' else {'name': 'remark_chunk_id', 'dataType': ['string']} for prop in listing_image_schema['properties']
      ],
      "vectorIndexType": "hnsw",
      "vectorIndexConfig": {
          "skip": False,
          "cleanupIntervalSeconds": 300,
          "pq": {"enabled": False},
          "maxConnections": 64,
          "efConstruction": 128,
          "ef": -1,
          "dynamicEfMin": 100,
          "dynamicEfMax": 500,
          "dynamicEfFactor": 8,
          "vectorCacheMaxObjects": 2000000,
          "flatSearchCutoff": 40000,
          "distance": "cosine"
      },
      "invertedIndexConfig": listing_image_schema['invertedIndexConfig']
    }

    # Create the schema for both classes
    self.client.schema.create_class(listing_image_schema)
    self.client.schema.create_class(listing_text_schema)

 
  def get_schema(self):
    return self.client.schema.get()

  def delete_all(self):
    self.client.schema.delete_all()

  def delete_listing(self, listing_id: str):
    """
    Delete all objects related to a listing_id.
    """
    listing_images = self.get(listing_id, embedding_type='I')  # image embeddings
    listing_texts = self.get(listing_id, embedding_type='T')   # text embeddings

    for listing in listing_images:
      self._delete_object_by_uuid(listing['uuid'], 'Listing_Image')

    for listing in listing_texts:
      self._delete_object_by_uuid(listing['uuid'], 'Listing_Text')

  def count_all(self) -> int:
    # count both Listing_Image and Listing_Text
    count_listing_image = self._count_by_class_name("Listing_Image")
    count_listing_text = self._count_by_class_name("Listing_Text")
    return count_listing_image + count_listing_text
 
  def get(self, listing_id: Optional[str] = None, embedding_type: str = 'I'):
    """
    Retrieve items from Weaviate related to listing_id and embedding_type.
    If listing_id is None, retrieve all items.
    embedding_type: 'I' for image, 'T' for text.
    """

    limit = 1000
    offset = 0

    class_name = "Listing_Image" if embedding_type == 'I' else "Listing_Text"
    extra_properties = ['image_name'] if embedding_type == 'I' else ['remark_chunk_id']
    additionals = ["id", "vector"]
    
    results = []
    while True:
      query = self.client.query.get(
        class_name=class_name,
        properties=self.common_properties + extra_properties
      ).with_additional(additionals).with_limit(limit).with_offset(offset)
      
      # Add the where filter only if a listing_id is provided
      if listing_id is not None:
        query = query.with_where({
          "operator": "Equal",
          "path": ["listing_id"],
          "valueText": listing_id
        })
      
      _results = query.do()

      if not _results['data']['Get'][class_name]:
        break  # Exit loop if no more records are returned
      
      for result in _results['data']['Get'][class_name]:
        result['uuid'] = result['_additional']['id']
        result['embedding'] = result['_additional']['vector']

        del result['_additional']

        result = self._postprocess_listing_json(result)
        results.append(result)

      offset += limit

    return results
    
 

  def insert(self, listing_json: Dict, embedding_type: str = 'I'):
    '''
    Insert a listing into the Weaviate database
    '''
    class_name = "Listing_Image" if embedding_type == 'I' else "Listing_Text"

    listing_json = self._preprocess_listing_json(listing_json, embedding_type=embedding_type)
    key = self._create_key(listing_json, embedding_type)

    if not 'embedding' in listing_json or listing_json['embedding'] is None or len(listing_json['embedding']) == 0:
      raise ValueError("The listing_json must contain an 'embedding' field with a non-empty vector.")

    vector = listing_json.pop('embedding')

    try:
      uuid = self.client.data_object.create(
          data_object=listing_json,
          class_name=class_name,
          vector=vector,
          uuid=key
      )
      return uuid
    except ObjectAlreadyExistsException as e:
      # TODO: log this instead
      print(f"Object with UUID {key} already exists for listing_id {listing_json['listing_id']}.")
      return None
    
  def upsert(self, listing_json: Dict, embedding_type: str = 'I'):
    class_name = "Listing_Image" if embedding_type == 'I' else "Listing_Text"

    listing_json = self._preprocess_listing_json(listing_json, embedding_type=embedding_type)
    key = self._create_key(listing_json, embedding_type)

    if not 'embedding' in listing_json or listing_json['embedding'] is None or len(listing_json['embedding']) == 0:
        raise ValueError("The listing_json must contain an 'embedding' field with a non-empty vector.")    
  
    vector = listing_json.pop('embedding')
    try:
      existing_object = self.client.data_object.get(uuid=key)
      if existing_object:
        self.client.data_object.update(
            data_object=listing_json,
            class_name=class_name,
            uuid=key,
            vector=vector
        )
      else:
        self.client.data_object.create(
            data_object=listing_json,
            class_name=class_name,
            vector=vector,
            uuid=key
        )
      return key
    except Exception as e:
      print(f"Error upserting object: {e}")
      return None
  
  
  def batch_insert(self, listings: Iterable[Dict], batch_size=100, embedding_type: str = 'I'):
    class_name = "Listing_Image" if embedding_type == 'I' else "Listing_Text"

    with self.client.batch.configure(batch_size=batch_size) as batch:
      for listing_json in tqdm(listings):
        listing_json = self._preprocess_listing_json(listing_json, embedding_type=embedding_type)
        key = self._create_key(listing_json, embedding_type)

        vector = listing_json.pop('embedding')

        try:
          batch.add_data_object(
              data_object=listing_json,
              class_name=class_name,
              vector=vector,
              uuid=key
          )
        except ObjectAlreadyExistsException as e:
          print(f"Object with UUID already exists: {e}")
        except Exception as e:
          print(listing_json)
          print(f"Error inserting object: {e}")

  def batch_upsert(self, listings: Iterable[Dict], batch_size=1000, embedding_type: str = 'I'):
    """
    Note this can be less inefficient due to the need to perform update in non batch manner
    # TODO: need to investigate this
    """
    class_name = "Listing_Image" if embedding_type == 'I' else "Listing_Text"

    with self.client.batch.configure(batch_size=batch_size) as batch:
      for listing_json in tqdm(listings):
        listing_json = self._preprocess_listing_json(listing_json, embedding_type=embedding_type)
        key = self._create_key(listing_json, embedding_type)
        vector = listing_json.pop('embedding')

        try:
          # check if the object already exists
          existing_object = self.client.data_object.get(uuid=key)
          if existing_object:
            self.client.data_object.update(
                data_object=listing_json,
                class_name=class_name,
                uuid=key,
                vector=vector
            )
          else:
            batch.add_data_object(
                data_object=listing_json,
                class_name=class_name,
                vector=vector,
                uuid=key
            )
        except Exception as e:
          print(f"Error upserting object: {e}")

    

  def import_from_faiss_index(self, faiss_index: FaissIndex, listing_df: pd.DataFrame, embedding_type: str = 'I', offset: int = None, length: int = None):
    """
    Import data from a FaissIndex object into Weaviate. The listing_df must be 
    there to provide the necessary metadata for each listing.
    """
    if offset is None and length is None:
      _embeddings = faiss_index.index.reconstruct_n(0, faiss_index.index.ntotal)
      _df = join_df(faiss_index.aux_info, listing_df, left_on='listing_id', right_on='jumpId', how='left')
    else:
      _embeddings = faiss_index.index.reconstruct_n(offset, length)
      _df = join_df(faiss_index.aux_info.iloc[offset:offset+length], listing_df, left_on='listing_id', right_on='jumpId', how='left')
    
    _df.drop(columns=['jumpId'], inplace=True)
    _df['embedding'] = [_embeddings[i] for i in range(_embeddings.shape[0])]

    listing_jsons = _df.to_dict(orient='records')

    self.batch_insert(listing_jsons, embedding_type=embedding_type)

    del _embeddings
    del _df
    gc.collect();

 
  def _delete_object_by_uuid(self, uuid, class_name):
    try:
      # Delete the object based on its UUID and class name
      self.client.data_object.delete(
        uuid=uuid,
        class_name=class_name
      )
      print(f"Object with UUID {uuid} successfully deleted.")
    except Exception as e:
      print(f"Error deleting object with UUID {uuid}: {e}")

  def _count_by_class_name(self, class_name) -> int:
    try:
      # Count the number of objects in the class
      count = self.client.query.aggregate(
        class_name=class_name
      ).with_meta_count().do()
      return count['data']['Aggregate'][class_name][0]['meta']['count']
    except Exception as e:
      print(f"Error counting objects in class {class_name}: {e}")
      return None

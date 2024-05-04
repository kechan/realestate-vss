from typing import Any, Tuple, List, Dict, Optional, Union
import redis, os, arrow, re
import json, pytz, math
from enum import Enum, auto
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from PIL import Image
from tqdm import tqdm

from realestate_core.common.utils import join_df
from realestate_vision.common.utils import get_listingId_from_image_name
from ..data.index import FaissIndex

from redis.commands.json.path import Path
from redis.commands.search.query import Query as RediSearchQuery
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.field import (
    TagField,
    TextField,
    NumericField,
    VectorField
)

INDEX_NAME = "listing_index"
DOC_PREFIX = "listing"
REDIS_INDEX_TYPE = os.environ.get("REDIS_INDEX_TYPE", "FLAT")
REDIS_DISTANCE_METRIC = os.environ.get("REDIS_DISTANCE_METRIC", "COSINE")
EMBEDDING_DIM = os.environ.get("EMBEDDING_VECTOR_DIMENSION", 768)

REDIS_DEFAULT_ESCAPED_CHARS = re.compile(r"[,.<>{}\[\]\\\"\':;!@#$%^&()\-+=~\/ ]")

# class ScoreAggregationMethod(Enum):
#   MAX = auto()
#   MEAN = auto()

# Helper functions
def to_unix_timestamp(date_str: str) -> int:
  """
  Convert a date string to a unix timestamp (seconds since epoch).

  Args:
      date_str: The date string to convert.

  Returns:
      The unix timestamp corresponding to the date string.

  If the date string cannot be parsed as a valid date format, returns the current unix timestamp and prints a warning.
  """
  # Try to parse the date string using arrow, which supports many common date formats
  try:
    date_obj = arrow.get(date_str)
    return int(date_obj.timestamp())
  except arrow.parser.ParserError:
    # If the parsing fails, return the current unix timestamp and print a warning
    # logger.info(f"Invalid date format: {date_str}")
    return int(arrow.now().timestamp())

def unpack_schema(d: dict):
  for v in d.values():
    if isinstance(v, dict):
      yield from unpack_schema(v)
    else:
      yield v


class RedisDataStore:
  def __init__(self, 
               client: redis.Redis, 
               image_embedder, 
               text_embedder,
               score_aggregation_method = 'max'
               ):
    self.client = client
    assert self.client.ping(), "Redis client is not working"
    self.image_embedder = image_embedder
    self.text_embedder = text_embedder
    self.score_aggregation_method = score_aggregation_method

    self.image_prefix = f"{DOC_PREFIX}:I"
    self.text_prefix = f"{DOC_PREFIX}:T"
    self.image_index_name = f"{INDEX_NAME}_I"
    self.text_index_name = f"{INDEX_NAME}_T"
    
    self.schema = self.create_schema()
    self.create_index()

  def get(self, listing_id: Optional[str] = None, obj_id: Optional[str] = None, embedding_type='I') -> Optional[Union[Dict, List[Dict]]]:
    """
    Retrieve a specific item from Redis based on the listing_id, obj_id and embedding_type.
    If listing_id is None, retrieve all items.
    If obj_id is None, retrieve all items for the given listing_id.
    embedding_type: 'I' for image, 'T' for text.
    """
    doc_prefix = self.image_prefix if embedding_type == 'I' else self.text_prefix
    try:
      if listing_id is None and obj_id is None:   # get all 
        redis_keys = self.client.keys(f"{doc_prefix}:*")
        listings_data = [self._postprocess_listing_json(self.client.json().get(redis_key)) for redis_key in redis_keys]
        return listings_data

      if obj_id is not None:
        redis_key = f"{doc_prefix}:{listing_id}:{obj_id}"
        listing_data = self.client.json().get(redis_key)
        if not listing_data: return None
        listing_data = self._postprocess_listing_json(listing_data)
        return listing_data
      else:
        redis_key_pattern = f"{doc_prefix}:{listing_id}:*"
        redis_keys = self.client.keys(redis_key_pattern)
        listings_data = [self._postprocess_listing_json(self.client.json().get(redis_key)) for redis_key in redis_keys]
        return listings_data
      
    except Exception as e:
      print(f"Error retrieving document: {e}")
      return None
    
  def get_unique_listing_ids(self) -> List[str]:
    try:
      keys = self.client.keys(f"{DOC_PREFIX}:*")
      listing_ids = set()
      for key in keys:
        key_str = key.decode("utf-8")
        listing_id = key_str.split(":")[2]
        listing_ids.add(listing_id)
      return list(listing_ids)
    except Exception as e:
      print(f"Error retrieving unique listing IDs: {e}")
      return []
    
  def _create_key(self, listing_json: Dict, embedding_type: str = 'I') -> str:
    """
    Create a Redis key for a listing document based on the listing_json.
    """
    if 'image_name' in listing_json.keys() and 'remark_chunk_id' in listing_json.keys():
      raise ValueError("Both 'image_name' and 'remark_chunk_id' should not be present at the same time since the vector is either an image embedding or text embedding.")

    if embedding_type == 'I' and 'image_name' in listing_json.keys() and listing_json['image_name'] is not None:
      redis_key = f"{self.image_prefix}:{listing_json['listing_id']}:{listing_json['image_name']}"
    elif embedding_type == 'T' and 'remark_chunk_id' in listing_json.keys() and listing_json['remark_chunk_id'] is not None:
      redis_key = f"{self.text_prefix}:{listing_json['listing_id']}:{listing_json['remark_chunk_id']}"
    else:
      # Fallback Redis key if neither is provided
      redis_key = f"{DOC_PREFIX}:{listing_json['listing_id']}"

    return redis_key

  def insert(self, listing_json: Dict, embedding_type: str = 'I'):
    """
    Perform necessary preprocessing on listing_doc and insert a listing into Redis.
    """
    # check to ensure either 'image_name' or 'remark_chunk_id' is provided but not both
    redis_key = self._create_key(listing_json, embedding_type=embedding_type)

    listing_json = self._preprocess_listing_json(listing_json)

    # for key, value in listing_json.items():
    #   print(f"Checking {key}: Type of value is {type(value)}")

    try:
      self.client.json().set(redis_key, Path.root_path(), listing_json)
      print(f"{redis_key} inserted into Redis.")

      # self.client.save()   # TODO: does this need optimization, maybe we don't need to save every time?
    except Exception as e:
      print(f"Error inserting data into Redis: {e}")

  def batch_insert(self, listings: List[Dict], batch_size=10000, embedding_type: str = 'I'):
    for i in tqdm(range(0, len(listings), batch_size), desc="Batch Insert Progress"):
      with self.client.pipeline(transaction=False) as pipe:
        for listing_json in listings[i:i+batch_size]:
          redis_key = self._create_key(listing_json, embedding_type=embedding_type)

          listing_json = self._preprocess_listing_json(listing_json)

          pipe.json().set(redis_key, Path.root_path(), listing_json)

        pipe.execute()

    # Optionally, call BGSAVE here if you want to ensure data is saved after the batch operation
    if self.client.info('persistence')['rdb_bgsave_in_progress'] == 0:      
      self.client.bgsave()
    else:
      print("Background save already in progress. Skipping bgsave.")      
  
  def _search_image_2_image(self, image: Image.Image = None, embedding: bytes = None, topk=5, group_by_listingId=False, include_all_fields=False, **filters):
    if embedding is None:
      image_features = self.image_embedder.embed_from_single_image(image).flatten().tolist()   # list[float]
      embedding = np.array(image_features, dtype=np.float64).tobytes()

    # image -> image 
    _query_filters = filters.copy()
    # _query_filters['embeddingType'] = 'I'

    redis_query: RediSearchQuery = self._get_redis_query(topk, **_query_filters)

    query_response = self.client.ft(self.image_index_name).search(
      redis_query, {"embedding": embedding}
    )

    top_scores, top_image_names = [], []
    # listing_remarks = {}
    all_json_fields = {}
    for doc in query_response.docs:
      doc_json = json.loads(doc.json)
      score = 1.-float(doc.score)    # such that higher score is better match
      top_scores.append(score)
      top_image_names.append(doc_json['image_name'])

      # add remarks (for dev, and demo, not needed for production, so remove later)
      remarks = doc_json['remarks']
      # listing_remarks[doc_json['listing_id']] = remarks

      if include_all_fields:
        all_json_fields[doc_json['listing_id']] = {k: v for k, v in doc_json.items() if k != 'embedding'}
      else:
        # only store remarks
        all_json_fields[doc_json['listing_id']] = {'remarks': remarks}

    if group_by_listingId:
      return self._image_search_groupby_listing(top_image_names, top_scores, include_all_fields, all_json_fields)

    return top_image_names, top_scores
  
  def _search_image_2_text(self, image: Image.Image = None, embedding: bytes = None, topk=5, group_by_listingId=False, include_all_fields=False, **filters):
    if embedding is None:
      image_features = self.image_embedder.embed_from_single_image(image).flatten().tolist()   # list[float]
      embedding = np.array(image_features, dtype=np.float64).tobytes()

    # image -> text
    _query_filters = filters.copy()
    # _query_filters['embeddingType'] = 'T'

    redis_query: RediSearchQuery = self._get_redis_query(topk, **_query_filters)

    query_response = self.client.ft(self.text_index_name).search(
      redis_query, {"embedding": embedding}
    )

    top_scores, top_remark_chunk_ids = [], []
    # listing_remarks = {}
    all_json_fields = {}
    for doc in query_response.docs:
      doc_json = json.loads(doc.json)
      score = 1.-float(doc.score)    # such that higher score is better match
      top_scores.append(score)
      top_remark_chunk_ids.append(doc_json['remark_chunk_id'])

      # add remarks (for dev, and demo, not needed for production, so remove later)
      remarks = doc_json['remarks']
      # listing_remarks[doc_json['listing_id']] = remarks
      if include_all_fields:
        all_json_fields[doc_json['listing_id']] = {k: v for k, v in doc_json.items() if k != 'embedding'}
      else:
        # only store remarks
        all_json_fields[doc_json['listing_id']] = {'remarks': remarks}

    if group_by_listingId:
      return self._text_search_groupby_listing(top_remark_chunk_ids, top_scores, include_all_fields, all_json_fields)
    
    return top_remark_chunk_ids, top_scores
  
  def _search_text_2_image(self, phrase: str = None, embedding: bytes = None, topk=5, group_by_listingId=False, include_all_fields=False, **filters):
    if embedding is None:
      text_features = self.text_embedder.embed_from_texts([phrase], batch_size=1)[0].flatten().tolist()   # list[float]
      embedding = np.array(text_features, dtype=np.float64).tobytes()

    # text -> image
    _query_filters = filters.copy()
    # _query_filters['embeddingType'] = 'I'

    redis_query: RediSearchQuery = self._get_redis_query(topk, **_query_filters)

    query_response = self.client.ft(self.image_index_name).search(
      redis_query, {"embedding": embedding}
    )

    top_scores, top_image_names = [], []
    # listing_remarks = {}
    all_json_fields = {}
    for doc in query_response.docs:
      doc_json = json.loads(doc.json)
      score = 1.-float(doc.score)
      top_scores.append(score)
      top_image_names.append(doc_json['image_name'])

      # add remarks (for dev, and demo, not needed for production, so remove later)
      remarks = doc_json['remarks']
      # listing_remarks[doc_json['listing_id']] = remarks

      if include_all_fields:
        all_json_fields[doc_json['listing_id']] = {k: v for k, v in doc_json.items() if k != 'embedding'}
      else:
        # only store remark
        all_json_fields[doc_json['listing_id']] = {'remarks': remarks}

    if group_by_listingId:
      return self._image_search_groupby_listing(top_image_names, top_scores, include_all_fields, all_json_fields)
    
    return top_image_names, top_scores

  def _search_text_2_text(self, phrase: str = None, embedding: bytes = None, topk=5, group_by_listingId=False, include_all_fields=False, **filters):
    if embedding is None:
      text_features = self.text_embedder.embed_from_texts([phrase], batch_size=1)[0].flatten().tolist()
      embedding = np.array(text_features, dtype=np.float64).tobytes()

    # text -> text
    _query_filters = filters.copy()
    # _query_filters['embeddingType'] = 'T'

    redis_query: RediSearchQuery = self._get_redis_query(topk, **_query_filters)

    query_response = self.client.ft(self.text_index_name).search(
      redis_query, {"embedding": embedding}
    )

    top_scores, top_remark_chunk_ids = [], []
    # listing_remarks = {}
    all_json_fields = {}
    for doc in query_response.docs:
      doc_json = json.loads(doc.json)
      score = 1.-float(doc.score)
      top_scores.append(score)
      top_remark_chunk_ids.append(doc_json['remark_chunk_id'])

      # add remarks (for dev, and demo, not needed for production, so remove later)
      remarks = doc_json['remarks']
      # listing_remarks[doc_json['listing_id']] = remarks
      if include_all_fields:
        all_json_fields[doc_json['listing_id']] = {k: v for k, v in doc_json.items() if k != 'embedding'}
      else:
        # only store remark
        all_json_fields[doc_json['listing_id']] = {'remarks': remarks}

    if group_by_listingId:
      return self._text_search_groupby_listing(top_remark_chunk_ids, top_scores, include_all_fields, all_json_fields)
    
    return top_remark_chunk_ids, top_scores

  def search(self, 
             image: Image.Image = None, 
             image_embedding: bytes = None,
             phrase: str = None, 
             text_embedding: bytes = None,
             topk=5, 
             group_by_listingId=False, 
             include_all_fields=False,
             **filters):
    # check if no image or phrase is provided, than just return empty thing

    if (not image and not phrase) and (not image_embedding and not text_embedding):
      if group_by_listingId: return []
      else: return [], []

    combined_results = []
    if image or image_embedding:
      if image_embedding is None:
        image_features = self.image_embedder.embed_from_single_image(image).flatten().tolist()   # list[float]
        image_embedding = np.array(image_features, dtype=np.float64).tobytes()

      listings_image_image = self._search_image_2_image(embedding=image_embedding, topk=topk, 
                                                        group_by_listingId=group_by_listingId, 
                                                        include_all_fields=include_all_fields,
                                                        **filters)
      listings_image_text = self._search_image_2_text(embedding=image_embedding, topk=topk, 
                                                      group_by_listingId=group_by_listingId, 
                                                      include_all_fields=include_all_fields,
                                                      **filters)

      if include_all_fields:
        # strip out fields and store it separately in listing_info
        listing_info = {}
        for listing in listings_image_image + listings_image_text:
          listing_info[listing['listingId']] = {k: v for k, v in listing.items() if k not in ['listingId', 'agg_score', 'remarks', 'image_names', 'image_name', 'embeddingType']}
          for key in list(listing.keys()):
            if key not in ['listingId', 'agg_score', 'remarks', 'image_names', 'remark_chunk_ids']:
              del listing[key]

      if group_by_listingId:        
        listings_image_image = self.normalize_scores(listings_image_image, 'agg_score')
        listings_image_text = self.normalize_scores(listings_image_text, 'agg_score')
        combined_results = self.merge_results(listings_image_text, listings_image_image)
      else:
        top_item_names, top_scores = listings_image_text
        top_scores = self.normalize_score_list(top_scores)

        top_item_names_2, top_scores_2 = listings_image_image
        top_scores_2 = self.normalize_score_list(top_scores_2)

        top_item_names += top_item_names_2
        top_scores += top_scores_2
    
    combined_results_2 = []
    if phrase or text_embedding:
      if text_embedding is None:
        text_features = self.text_embedder.embed_from_texts([phrase], batch_size=1)[0].flatten().tolist()   # list[float]
        text_embedding = np.array(text_features, dtype=np.float64).tobytes()

      listings_text_image = self._search_text_2_image(embedding=text_embedding, topk=topk, 
                                                      group_by_listingId=group_by_listingId, 
                                                      include_all_fields=include_all_fields,
                                                      **filters)
      listings_text_text = self._search_text_2_text(embedding=text_embedding, topk=topk, 
                                                    group_by_listingId=group_by_listingId,
                                                    include_all_fields=include_all_fields,
                                                    **filters)

      if include_all_fields:
        # strip out fields and store it separately in listing_info
        for listing in listings_text_image + listings_text_text:
          listing_info[listing['listingId']] = {k: v for k, v in listing.items() if k not in ['listingId', 'agg_score', 'remarks', 'image_names', 'image_name', 'embeddingType']}
          for key in list(listing.keys()):
            if key not in ['listingId', 'agg_score', 'remarks', 'image_names', 'remark_chunk_ids']:
              del listing[key]

      if group_by_listingId:
        listings_text_image = self.normalize_scores(listings_text_image, 'agg_score')
        listings_text_text = self.normalize_scores(listings_text_text, 'agg_score')
        combined_results_2 += self.merge_results(listings_text_text, listings_text_image)
      else:
        top_item_names_3, top_scores_3 = listings_text_image
        top_scores_3 = self.normalize_score_list(top_scores_3)

        top_item_names_4, top_scores_4 = listings_text_text
        top_scores_4 = self.normalize_score_list(top_scores_4)

        top_item_names += top_item_names_3
        top_scores += top_scores_3
        top_item_names += top_item_names_4
        top_scores += top_scores_4

    combined_results += combined_results_2

    if group_by_listingId:
      combined_results.sort(key=lambda x: x['agg_score'], reverse=True)
      if not include_all_fields:
        return combined_results
      else:
        # for each result, add the stripped out fields
        for result in combined_results:
          result.update(listing_info[result['listingId']])
        return combined_results
    else:
      # sort tuples by score in descending order
      top_item_names, top_scores = zip(*sorted(zip(top_item_names, top_scores), key=lambda x: x[1], reverse=True))
      return top_item_names, top_scores

  def multi_image_search(self, images: List[Image.Image], phrase: str = None, topk=5, group_by_listingId=False, **filters):
    # use mean(image embeddings) for now.
    all_image_features = []
    # print(f'# of images: {len(images)}')
    for image in images:
      image_features = self.image_embedder.embed_from_single_image(image)
      # print(f'Image features shape: {image_features.shape}')
      all_image_features.append(image_features)
    mean_vector = np.mean(all_image_features, axis=0)
    # print(f'Mean vector shape: {mean_vector.shape}')

    image_features = mean_vector.flatten().tolist()   # list[float]
    image_embedding = np.array(image_features, dtype=np.float64).tobytes()

    text_embedding = None
    if phrase:
      text_features = self.text_embedder.embed_from_texts([phrase], batch_size=1)[0].flatten().tolist()
      text_embedding = np.array(text_features, dtype=np.float64).tobytes()

    return self.search(image_embedding=image_embedding, text_embedding=text_embedding, topk=topk, group_by_listingId=group_by_listingId, **filters)




  def _preprocess_listing_json(self, listing_json: Dict) -> Dict:
    """
    Perform necessary preprocessing on listing_json to conform to schema data type requirements before INSERT.
    Failing to do so, the doc won't get indexed properly.
    """
    # Transform embedding if it's a NumPy array      
    if 'embedding' in listing_json and isinstance(listing_json['embedding'], np.ndarray):
      # if a NumPy array, convert to a list of floats
      listing_json['embedding'] = listing_json['embedding'].tolist()

    # transform propertyFeatures to comma-separated string (original a list or nadarray of strings)
    if 'propertyFeatures' in listing_json and isinstance(listing_json['propertyFeatures'], (list, np.ndarray)):
      listing_json['propertyFeatures'] = ', '.join(listing_json['propertyFeatures'])

    # Convert boolean fields to integers to align with Redisearch NumericField expectations
    boolean_fields = ['carriageTrade', 'pool', 'garage', 'waterFront', 'fireplace', 'ac']  # Add other boolean fields as needed
    for field in boolean_fields:
      if field in listing_json:
        listing_json[field] = int(listing_json[field])

    if listing_json.get('listingDate') and isinstance(listing_json['listingDate'], str):
      listing_json['listingDate'] = to_unix_timestamp(listing_json['listingDate'])

    if listing_json.get('lastUpdate') and isinstance(listing_json['lastUpdate'], str):
      listing_json['lastUpdate'] = to_unix_timestamp(listing_json['lastUpdate'])
    
    if listing_json.get('lastPhotoUpdate') and isinstance(listing_json['lastPhotoUpdate'], str):
      listing_json['lastPhotoUpdate'] = to_unix_timestamp(listing_json['lastPhotoUpdate'])

    # Handle NaN values for 'lat' and 'lng'
    if 'lat' in listing_json and np.isnan(listing_json['lat']):
        listing_json['lat'] = None  # Replace NaN with None (becomes null in JSON)
        
    if 'lng' in listing_json and np.isnan(listing_json['lng']):
        listing_json['lng'] = None  # Replace NaN with None

    # Check if the embedding is an image embedding, all else is 0 for now (i.e. text embedding)
    if 'image_name' in listing_json and listing_json['image_name'] is not None:
      listing_json['embeddingType'] = 'I'
    else:
      listing_json['embeddingType'] = 'T'

    return listing_json
  
  def _postprocess_listing_json(self, listing_json: Dict) -> Dict:
    '''
    Basically reverse of _preprocess_listing_json for relevant fields.
    '''
    if 'propertyFeatures' in listing_json and isinstance(listing_json['propertyFeatures'], str):
      listing_json['propertyFeatures'] = listing_json['propertyFeatures'].split(', ')

    # Convert boolean fields to integers
    boolean_fields = ['carriageTrade', 'pool', 'garage', 'waterFront', 'fireplace', 'ac']  # Add other boolean fields as needed
    for field in boolean_fields:
      if field in listing_json:
        listing_json[field] = bool(listing_json[field])

    if listing_json.get('listingDate') and isinstance(listing_json['listingDate'], int):
      listing_json['listingDate'] = arrow.get(listing_json['listingDate']).format('YYYY/MM/DD')
    
    if listing_json.get('lastUpdate') and isinstance(listing_json['lastUpdate'], int):
      listing_json['lastUpdate'] = arrow.get(listing_json['lastUpdate']).format('YYYY/MM/DD')

    if listing_json.get('lastPhotoUpdate') and isinstance(listing_json['lastPhotoUpdate'], int):
      listing_json['lastPhotoUpdate'] = arrow.get(listing_json['lastPhotoUpdate']).format('YYYY/MM/DD')

    # if 'embeddingType' in listing_json and isinstance(listing_json['embeddingType'], str):
    #   listing_json['embeddingType'] = bool(listing_json['embeddingType'])

    return listing_json
    
  
  def _get_redis_query(self, topk: int = 50, **filters):
    query_parts = []
    for field, value in filters.items():
      query_part = self._construct_query_part(field, value)
      if query_part:
        query_parts.append(query_part)

    if len(query_parts) == 1:
      final_attrib_query = f'{query_parts[0]}'
    elif len(query_parts) > 1:
      query_parts = [f"{query_part}" for query_part in query_parts]   # surround with () for each query part
      final_attrib_query = " ".join(query_parts).strip()
    else: #elif len(query_parts) == 0:
      final_attrib_query = "*"

    # Prepare query string
    query_str = (
        f"({final_attrib_query})=>[KNN {topk} @embedding $embedding as score]"
    )
    print(f'query_str: {query_str}')
    redis_query = RediSearchQuery(query_str).sort_by("score").paging(0, topk).dialect(2)

    return redis_query
    
  def import_from_faiss_index(self, faiss_index: FaissIndex, listing_df: pd.DataFrame, embedding_type: str = 'I', sample_size: int = None):
    """
    Import data from a FaissIndex object into Redisearch. The listing_df must be 
    there to provide the necessary metadata for each listing.
    """
    _embeddings = faiss_index.index.reconstruct_n(0, faiss_index.index.ntotal)

    _df = join_df(faiss_index.aux_info, listing_df, left_on='listing_id', right_on='jumpId', how='left')
    _df.drop(columns=['jumpId'], inplace=True)
    _df['embedding'] = [_embeddings[i] for i in range(_embeddings.shape[0])]

    if sample_size is not None:
      _df = _df.sample(sample_size)
      _df.defrag_index(drop=True)

    listing_jsons = _df.to_dict(orient='records')

    self.batch_insert(listing_jsons, embedding_type=embedding_type)


  def delete_index(self):
    try:
      self.client.ft(self.image_index_name).dropindex()
      self.client.ft(self.text_index_name).dropindex()
    except Exception as e:
      print(f"Error deleting index: {e}")

  def delete_all_listings(self):
    try:
      keys = self.client.keys(f"{DOC_PREFIX}:*")
      if keys:
        self.client.delete(*keys)
        print(f"All documents with prefix {DOC_PREFIX} deleted successfully.")
      else:
        print(f"No documents found with prefix {DOC_PREFIX}.")
    except Exception as e:
      print(f"Error deleting documents: {e}")

  def delete_listing(self, listing_id: str):
    try:
      keys = self.client.keys(f"{DOC_PREFIX}:{listing_id}:*")
      if keys:
        self.client.delete(*keys)
        print(f"All documents with prefix {DOC_PREFIX}:{listing_id} deleted successfully.")
      else:
        print(f"No documents found with prefix {DOC_PREFIX}:{listing_id}.")
    except Exception as e:
      print(f"Error deleting documents: {e}")

  def delete_listings(self, listing_ids: List[str]):
    # for listing_id in listing_ids:
    #   self.delete_listing(listing_id)
    try:
      keys = []
      for listing_id in listing_ids:
        keys.extend(self.client.keys(f"{DOC_PREFIX}:{listing_id}:*"))
      if keys:
        self.client.delete(*keys)
        print(f"All {keys} documents deleted successfully.")
      else:
        print(f"No documents belonging to listing_ids found.")
    except Exception as e:
      print(f"Error deleting documents: {e}")


  def create_schema(self) -> Dict:

    return {
      "source": {
        "listing_id": TextField("$.listing_id", as_name='listing_id'),
        "city": TextField("$.city", as_name='city'),
        "provState": TagField("$.provState", as_name='provState'),
        "embeddingType": TagField("$.embeddingType", as_name='embeddingType'),
        "postalCode": TextField("$.postalCode", as_name='postalCode'),
        "lat": NumericField("$.lat", as_name='lat'),
        "lng": NumericField("$.lng", as_name='lng'),
        "streetName": TextField("$.streetName", as_name='streetName'),
        "beds": TextField("$.beds", as_name='beds'),     # e.g. '3+1'
        "bedsInt": NumericField("$.bedsInt", as_name='bedsInt'),
        "baths": TextField("$.baths", as_name='baths'),
        "bathsInt": NumericField("$.bathsInt", as_name='bathsInt'),
        "sizeInterior": TextField("$.sizeInterior", as_name='sizeInterior'),
        "sizeInteriorUOM": TagField("$.sizeInteriorUOM", as_name='sizeInteriorUOM'),
        "lotSize": TextField("$.lotSize", as_name='lotSize'),
        "lotUOM": TagField("$.lotUOM", as_name='lotUOM'),
        "propertyFeatures": TextField("$.propertyFeatures", as_name='propertyFeatures'),  # e.g. '[fireplace, ac, parking, garage]'
        "propertyType": TagField("$.propertyType", as_name='propertyType'),
        "transactionType": TagField("$.transactionType", as_name='transactionType'),
        "carriageTrade": NumericField("$.carriageTrade", as_name='carriageTrade'),  # boolean
        "price": NumericField("$.price", as_name='price'),
        "leasePrice": NumericField("$.leasePrice", as_name='leasePrice'),
        "pool": NumericField("$.pool", as_name='pool'),    # boolean
        "garage": NumericField("$.garage", as_name='garage'),  # boolean
        "waterFront": NumericField("$.waterFront", as_name='waterFront'), # boolean
        "fireplace": NumericField("$.fireplace", as_name='fireplace'),  # boolean
        "ac": NumericField("$.ac", as_name='ac'),  # boolean
        "remarks": TextField("$.remarks", as_name='remarks'),
        "photo": TextField("$.photo", as_name='photo'),
        "listingDate": NumericField("$.listingDate", as_name='listingDate'),
        "lastUpdate": NumericField("$.lastUpdate", as_name='lastUpdate'),
        "lastPhotoUpdate": NumericField("$.lastPhotoUpdate", as_name='lastPhotoUpdate')
      },
      "embedding_related": {
        "image_name": TextField("$.image_name", as_name='image_name'),
        "remark_chunk_id": TextField("$.remark_chunk_id", as_name='remark_chunk_id')
      },
      "embedding":
        VectorField("$.embedding", 
                    REDIS_INDEX_TYPE,
                    {
                      "TYPE": "FLOAT64",
                      "DIM": EMBEDDING_DIM,
                      "DISTANCE_METRIC": REDIS_DISTANCE_METRIC,
                    },
                    as_name='embedding')
    }
  
  def create_index(self):
    # we create 2 kind of indexes, one for listing with image embedding and another for text embedding
    # for image, we will adapt DOC_PREFIX:I:{listing_id}:{image_name}
    # for text, we will adapt DOC_PREFIX:T:{listing_id}:{remark_chunk_id}

    
    self.image_index_definition = IndexDefinition(prefix=[self.image_prefix], index_type=IndexType.JSON)
    self.text_index_definition = IndexDefinition(prefix=[self.text_prefix], index_type=IndexType.JSON)

    # create the index with the defined schema
    try:
      fields = list(unpack_schema(self.schema))

      self.client.ft(self.image_index_name).create_index(fields=fields, definition=self.image_index_definition)
      print(f"Index {self.image_index_name} created successfully")

      self.client.ft(self.text_index_name).create_index(fields=fields, definition=self.text_index_definition)
      print(f"Index {self.text_index_name} created successfully")
    except Exception as e:
      print(f"Error creating index {self.image_index_name}, {self.text_index_name}: {e}")

  def index_info(self):
    return [self.client.ft(self.image_index_name).info(), self.client.ft(self.text_index_name).info()]
  
  def bgsave(self):
    self.client.bgsave()

  
  def _construct_query_part(self, field: str, value: Any) -> str:
    def _escape(value: str) -> str:
      """
      Escape filter value.

      Args:
          value (str): Value to escape.

      Returns:
          str: Escaped filter value for RediSearch.
      """

      def escape_symbol(match) -> str:
        value = match.group(0)
        return f"\\{value}"

      return REDIS_DEFAULT_ESCAPED_CHARS.sub(escape_symbol, value)
    
    field_type = self.schema['source'].get(field, None)
    if not field_type:
      field_type = self.schema['embedding_related'].get(field, None)

    if isinstance(field_type, TagField):
      return f"@{field}:{{{_escape(value)}}}"
    elif isinstance(field_type, TextField):
      return f"@{field}:{value}"
    elif isinstance(field_type, NumericField):
      if isinstance(value, dict) and 'min' in value and 'max' in value:
        return f"@{field}:[{value['min']} {value['max']}]"
      elif isinstance(value, list) and len(value) == 2 and value[0] is None and value[1] is not None:
        return f"@{field}:[-inf {value[1]}]"
      elif isinstance(value, list) and len(value) == 2 and value[0] is not None and value[1] is None:
        return f"@{field}:[{value[0]} inf]"
      else:
        return f"@{field}:{value}"
    else:
      return None # Unsupported field type
  
  def _image_search_groupby_listing(self, 
                                    image_names: List[str], 
                                    scores: List[float],
                                    include_all_fields: bool = False,
                                    all_json_fields: Dict[str, Any] = None
                                    ) -> List[Dict[str, Union[str, float, List[str]]]]:
    """
    Given a list of image names and their scores, generate a list of listings with the aggregated score and image names for that listing

    # listingIds = [get_listingId_from_image_name(image_name) for image_name in image_names]
    # image names are of format {listingId}_{imageId}.jpg, we want to organize by listingId
    # such that we get a dict whose keys are listing_ids and values are list of image names
    # and another dict whose keys are listing_ids and values are list of corresponding scores
    """
    listingId_to_image_names = {}
    listingId_to_scores = {}
    for image_name, score in zip(image_names, scores):
      listingId = get_listingId_from_image_name(image_name)
      if listingId not in listingId_to_image_names:
        listingId_to_image_names[listingId] = []
        listingId_to_scores[listingId] = []

      listingId_to_image_names[listingId].append(image_name)
      listingId_to_scores[listingId].append(score)

    listings = []
    for listingId, image_names in listingId_to_scores.items():
      if self.score_aggregation_method == 'max':
        agg_score = np.max(np.array(listingId_to_scores[listingId]))
      elif self.score_aggregation_method == 'mean':
        agg_score = np.mean(np.array(listingId_to_scores[listingId]))
      else:
        raise ValueError(f'Unknown score aggregation method {self.score_aggregation_method}')

      image_names = [f"{listingId}/{image_name}" for image_name in listingId_to_image_names[listingId]]
      listings.append({
        'listingId': listingId,
        "agg_score": float(agg_score),
        "image_names": image_names,
      })

    listings = sorted(listings, key=lambda x: x['agg_score'], reverse=True)

    for listing in listings:
      listing['remarks'] = all_json_fields[listing['listingId']]['remarks']
      if include_all_fields:
        listing.update(all_json_fields[listing['listingId']])

    return listings

  def _text_search_groupby_listing(self, 
                                   remark_chunk_ids: List[str], 
                                   scores: List[float],
                                   include_all_fields: bool = False,
                                   all_json_fields: Dict[str, Any] = None
                                   ) -> List[Dict[str, Union[str, float, List[str]]]]:
      """
      Given a list of remark chunk IDs and their scores, generate a list of listings with the aggregated score and remark chunk IDs for that listing.

      The remark chunk IDs are of the format {listingId}_{chunkId}, we want to organize by listingId
      such that we get a dict whose keys are listing_ids and values are list of remark chunk IDs
      and another dict whose keys are listing_ids and values are list of corresponding scores
      """
      listingId_to_remark_chunk_ids = {}
      listingId_to_scores = {}
      for remark_chunk_id, score in zip(remark_chunk_ids, scores):
        listingId = remark_chunk_id.split('_')[0]  # Assuming the format is {listingId}_{chunkId}
        if listingId not in listingId_to_remark_chunk_ids:
          listingId_to_remark_chunk_ids[listingId] = []
          listingId_to_scores[listingId] = []

        listingId_to_remark_chunk_ids[listingId].append(remark_chunk_id)
        listingId_to_scores[listingId].append(score)

      listings = []
      for listingId, remark_chunk_ids in listingId_to_remark_chunk_ids.items():
        if self.score_aggregation_method == 'max':
          agg_score = np.max(np.array(listingId_to_scores[listingId]))
        elif self.score_aggregation_method == 'mean':
          agg_score = np.mean(np.array(listingId_to_scores[listingId]))
        else:
          raise ValueError(f'Unknown score aggregation method {self.score_aggregation_method}')

        listings.append({
          'listingId': listingId,
          "agg_score": float(agg_score),
          "remark_chunk_ids": remark_chunk_ids,
        })

      # Sort the listings by average score in descending order
      listings = sorted(listings, key=lambda x: x['agg_score'], reverse=True)

      for listing in listings:
        listing['image_names'] = []    # no image names for text search, but keep response more consistent
        listing['remarks'] = all_json_fields[listing['listingId']]['remarks']
        if include_all_fields:
          listing.update(all_json_fields[listing['listingId']])

      return listings

  def normalize_scores(self, results: List[Dict], score_key: str) -> List[Dict]:
    if len(results) == 0: return results
    scores = [result[score_key] for result in results]
    min_score = min(scores)
    max_score = max(scores)
    for result in results:
      result[score_key] = (result[score_key] - min_score) / (max_score - min_score)
    return results
  
  def normalize_score_list(self, scores: List[float]) -> List[float]:
    if len(scores) == 0: return scores
    min_score = min(scores)
    max_score = max(scores)
    return [(score - min_score) / (max_score - min_score) for score in scores]
  
  def merge_results(self, text_results: List[Dict], image_results: List[Dict]) -> List[Dict]:
    # Convert image_results to a dictionary for easy lookup
    listings_dict = {listing['listingId']: listing for listing in image_results}

    # Merge text_results into listings_dict
    for result in text_results:
      listingId = result['listingId']
      if listingId in listings_dict:
        # Add score to agg_score
        listings_dict[listingId]['agg_score'] += result['agg_score']
      else:
        # Add new listing to listings_dict
        listings_dict[listingId] = {
            'listingId': listingId,
            'agg_score': result['agg_score'],
            'remarks': result['remarks'],
            'image_names': []
        }

    # Convert listings_dict back to a list
    combined_results = list(listings_dict.values())

    return combined_results

  # these are methods that conform to class ListingSearchEngine interface
  # which is also expected by main.py the fastAPI app
  def get_listing(self, listing_id: str) -> Dict[str, Any]:
    listing_docs = self.get(listing_id=listing_id)
    if len(listing_docs) == 0: return {}

    listing_doc = listing_docs[0]

    # add jumpId
    listing_doc['jumpId'] = listing_doc['listing_id']

    # remove embedding and embeddingType
    if 'embedding' in listing_doc:
      listing_doc.pop('embedding', None)
    if 'embeddingType' in listing_doc:
      listing_doc.pop('embeddingType', None)
    if 'propertyFeatures' in listing_doc and isinstance(listing_doc['propertyFeatures'], np.ndarray):
      listing_doc['propertyFeatures'] = listing_doc['propertyFeatures'].tolist()

    return listing_doc
  
  def get_imagenames(self, listingId: str) -> List[str]:
    image_names = self.client.keys(f"{self.image_prefix}:{listingId}:*")

    image_names = [f'{image_name.decode("utf-8").split(":")[-1]}'
                   for image_name in image_names]
    return image_names
  

      
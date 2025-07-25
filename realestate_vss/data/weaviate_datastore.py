from typing import Any, Tuple, List, Dict, Optional, Union, Iterable
import weaviate, asyncio, uuid, math, time, gc
from PIL import Image
from datetime import datetime
from dateutil import parser
from pathlib import Path

import numpy as np
import pandas as pd

from tqdm.auto import tqdm

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, wait_fixed

from realestate_core.common.utils import join_df, save_to_pickle, load_from_pickle
from realestate_vision.common.utils import get_listingId_from_image_name

from weaviate.exceptions import ObjectAlreadyExistsException, UnexpectedStatusCodeException
from requests.exceptions import ConnectionError, Timeout

from weaviate.client import WeaviateAsyncClient

try:
  # for weaviate-client v4
  from weaviate.classes.config import Configure, Property, DataType
  from weaviate.classes.query import MetadataQuery, Filter
  from weaviate.exceptions import UnexpectedStatusCodeError
  from weaviate.classes.config import ConsistencyLevel
except ImportError:
  raise NotImplementedError("This class is only compatible with weaviate-client v4")

import logging

RETRY_SETTINGS = {
    "stop": stop_after_attempt(10),
    # "wait": wait_exponential(multiplier=1, min=2, max=10),
    "wait": wait_fixed(30),
    "retry": retry_if_exception_type(UnexpectedStatusCodeError),
    "reraise": True
}

class WeaviateDataStore:
  def __init__(self,
               image_embedder,
               text_embedder,
               score_aggregation_method = 'max',
               use_replication = False,
               consistency_level: str = "ONE"
               ):
    self.logger = logging.getLogger(self.__class__.__name__)

    self.image_embedder = image_embedder
    self.text_embedder = text_embedder
    self.score_aggregation_method = score_aggregation_method

    self.doc_prefix = 'listing'   # used to help generate unique UUIDs
    self.custom_uuid_namespace = uuid.uuid5(uuid.NAMESPACE_DNS, 'jumptools.com')

    self.common_properties = [
      'listing_id', # Identifiers
      'city', 'provState', 'postalCode', # Location details
      'lat', 'lng', 'streetName', # Geographical details
      'beds', 'bedsInt', 'baths', 'bathsInt', # Basic property specifications
      'sizeInterior', 'sizeInteriorUOM', # Interior size details
      'lotSize', 'lotUOM', # Lot size details
      'propertyType', 'transactionType', # Property classification
      'price', 'leasePrice', # Pricing details
      'carriageTrade', # Premium property indicator
      'propertyFeatures', 'pool', 'garage', 'waterFront', 'fireplace', 'ac', # Amenities
      'remarks', 'photo', # Additional information
      'listingDate', 'lastPhotoUpdate', 'lastUpdate' # Timestamps
    ]

    # Replication settings
    self.use_replication = use_replication
    self._consistency_levels = {
      "ONE": weaviate.classes.config.ConsistencyLevel.ONE,
      "QUORUM": weaviate.classes.config.ConsistencyLevel.QUORUM,
      "ALL": weaviate.classes.config.ConsistencyLevel.ALL 
    }
    if consistency_level not in self._consistency_levels:
      raise ValueError(f"consistency_level must be one of {list(self._consistency_levels.keys())}")
    self.consistency_level = self._consistency_levels[consistency_level]


  def _create_key(self, listing_json: Dict, embedding_type: str = 'I') -> uuid.UUID:
      """
      Create a uuid key for a listing document based on the listing_json.
      """
      if 'image_name' in listing_json.keys() and 'remark_chunk_id' in listing_json.keys():
        self.logger.error("Both 'image_name' and 'remark_chunk_id' are present in the listing_json. This is not expected.")
        raise ValueError("Both 'image_name' and 'remark_chunk_id' should not be present at the same time since the vector is either an image embedding or text embedding.")
      
      if embedding_type == 'I' and 'image_name' not in listing_json.keys():
        self.logger.error("The 'image_name' field is required for image embeddings.")
        raise ValueError("The 'image_name' field is required for image embeddings.")
      
      if embedding_type == 'T' and 'remark_chunk_id' not in listing_json.keys():
        self.logger.error("The 'remark_chunk_id' field is required for text embeddings.")
        raise ValueError("The 'remark_chunk_id' field is required for text embeddings.")

      if embedding_type == 'I' and 'image_name' in listing_json.keys() and listing_json['image_name'] is not None:
        key = f"{self.doc_prefix}:{listing_json['listing_id']}:{listing_json['image_name']}"
      elif embedding_type == 'T' and 'remark_chunk_id' in listing_json.keys() and listing_json['remark_chunk_id'] is not None:
        key = f"{self.doc_prefix}:{listing_json['listing_id']}:{listing_json['remark_chunk_id']}"
      else:
        # Fallback Redis key if neither is provided
        key = f"{self.doc_prefix}:{listing_json['listing_id']}"

      unique_uuid = uuid.uuid5(self.custom_uuid_namespace, key)

      return unique_uuid

  def _convert_datestr_to_iso(self, date_str: str) -> str:
    """
    Convert a date string to 'YYYY-MM-DDTHH:MM:SSZ'
    """
    # return datetime.strptime(date_str, '%Y/%m/%d').strftime('%Y-%m-%dT%H:%M:%SZ')
    try:
      # Parse the datetime string
      dt = parser.parse(date_str)
      # Format it into the desired ISO 8601 format
      formatted_date = dt.strftime('%Y-%m-%dT%H:%M:%SZ')
      return formatted_date
    except ValueError as e:
      try:
        return datetime.strptime(date_str, '%y-%m-%d:%H:%M:%S').strftime('%Y-%m-%dT%H:%M:%SZ')
      except ValueError as e:
        self.logger.error(f"Error parsing date: {e}")
        return None

  def _preprocess_listing_json(self, listing_json: Dict, embedding_type: str = 'I') -> Dict:
    """
    Perform necessary preprocessing on the listing json before storing it in the Weaviate database
    """
    def sanitize_listing_json(listing_json):
      # Replace NaNs or non-scalar NaNs with None
      sanitized_listing_json = {}
      for key, value in listing_json.items():
        if isinstance(value, (list, dict)):  # If the value is a list or dict, keep it as is.
          sanitized_listing_json[key] = value
        elif pd.isna(value):  # Replace NaN or NaT with None
          sanitized_listing_json[key] = None
        else:
          sanitized_listing_json[key] = value
      return sanitized_listing_json
    
    if embedding_type == 'I':
      wanted_properties = self.common_properties + ['image_name']
    else:
      wanted_properties = self.common_properties + ['remark_chunk_id', 'chunk_start', 'chunk_end']

    # keep only keys that are in self.common_properties and 'embedding'
    property_keys = [p for p in wanted_properties] + ['embedding']
    listing_json = {key: listing_json[key] for key in property_keys if key in listing_json}
    
    if 'embedding' in listing_json and isinstance(listing_json['embedding'], np.ndarray):
      listing_json['embedding'] = listing_json['embedding'].flatten().tolist()
    
    if 'propertyFeatures' in listing_json and isinstance(listing_json['propertyFeatures'], (list, np.ndarray)):
      listing_json['propertyFeatures'] = ', '.join(listing_json['propertyFeatures'])
    
    # format the date correctly for weaviate
    if 'listingDate' in listing_json and isinstance(listing_json['listingDate'], str):
      # listing_json['listingDate'] = datetime.strptime(listing_json['listingDate'], '%Y/%m/%d').strftime('%Y-%m-%dT%H:%M:%SZ')
      listing_json['listingDate'] = self._convert_datestr_to_iso(listing_json['listingDate'])
    if 'lastUpdate' in listing_json and isinstance(listing_json['lastUpdate'], str):
      # listing_json['lastUpdate'] = datetime.strptime(listing_json['lastUpdate'], '%Y/%m/%d').strftime('%Y-%m-%dT%H:%M:%SZ')
      listing_json['lastUpdate'] = self._convert_datestr_to_iso(listing_json['lastUpdate'])
    if 'lastPhotoUpdate' in listing_json and isinstance(listing_json['lastPhotoUpdate'], str):
      # listing_json['lastPhotoUpdate'] = datetime.strptime(listing_json['lastPhotoUpdate'], '%Y/%m/%d').strftime('%Y-%m-%dT%H:%M:%SZ')
      listing_json['lastPhotoUpdate'] = self._convert_datestr_to_iso(listing_json['lastPhotoUpdate'])

    if 'lat' in listing_json:
      if isinstance(listing_json['lat'], str):
        listing_json['lat'] = float(listing_json['lat'])
      if isinstance(listing_json['lat'], float) and math.isnan(listing_json['lat']):
        listing_json['lat'] = None
    
    if 'lng' in listing_json:
      if isinstance(listing_json['lng'], str):
        listing_json['lng'] = float(listing_json['lng'])
      if isinstance(listing_json['lng'], float) and math.isnan(listing_json['lng']):
        listing_json['lng'] = None

    # photos is not needed and its quite long, not needed. its also np.ndarray
    if 'photos' in listing_json:
      del listing_json['photos']

    # Replace NaNs with None
    listing_json = sanitize_listing_json(listing_json)

    return listing_json
  
  def _postprocess_listing_json(self, listing_json: Dict) -> Dict:
    '''
    Basically reverse of _preprocess_listing_json for relevant fields.
    '''

    if 'propertyFeatures' in listing_json and listing_json['propertyFeatures'] is not None:
      listing_json['propertyFeatures'] = listing_json['propertyFeatures'].split(', ')

    if 'listingDate' in listing_json and isinstance(listing_json['listingDate'], datetime):
      # listing_json['listingDate'] = datetime.strptime(listing_json['listingDate'], '%Y-%m-%dT%H:%M:%SZ').strftime('%Y/%m/%d')
      listing_json['listingDate'] = listing_json['listingDate'].strftime('%Y/%m/%d')
    if 'lastUpdate' in listing_json and isinstance(listing_json['lastUpdate'], datetime):
      # listing_json['lastUpdate'] = datetime.strptime(listing_json['lastUpdate'], '%Y-%m-%dT%H:%M:%SZ').strftime('%Y/%m/%d')
      listing_json['lastUpdate'] = listing_json['lastUpdate'].strftime('%Y/%m/%d')
    if 'lastPhotoUpdate' in listing_json and isinstance(listing_json['lastPhotoUpdate'], datetime):
      # listing_json['lastPhotoUpdate'] = datetime.strptime(listing_json['lastPhotoUpdate'], '%Y-%m-%dT%H:%M:%SZ').strftime('%Y/%m/%d')
      listing_json['lastPhotoUpdate'] = listing_json['lastPhotoUpdate'].strftime('%Y/%m/%d')

    return listing_json

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
      listing['remarks'] = all_json_fields[listing['listingId']].get('remarks', '')
      if include_all_fields:
        listing.update(all_json_fields[listing['listingId']])

    return listings

  def _text_search_groupby_listing(self, 
                                   remark_chunk_ids: List[str], 
                                   scores: List[float],
                                   positions: List[Tuple[int, int]],
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
      listingId_to_positions = {}
      for remark_chunk_id, score, pos in zip(remark_chunk_ids, scores, positions):
        listingId = remark_chunk_id.split('_')[0]  # Assuming the format is {listingId}_{chunkId}
        if listingId not in listingId_to_remark_chunk_ids:
          listingId_to_remark_chunk_ids[listingId] = []
          listingId_to_scores[listingId] = []
          listingId_to_positions[listingId] = []

        listingId_to_remark_chunk_ids[listingId].append(remark_chunk_id)
        listingId_to_scores[listingId].append(score)
        listingId_to_positions[listingId].append(pos)

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
          "remark_chunk_pos": listingId_to_positions[listingId]
        })

      # Sort the listings by average score in descending order
      listings = sorted(listings, key=lambda x: x['agg_score'], reverse=True)

      for listing in listings:
        listing['image_names'] = []    # no image names for text search, but keep response more consistent
        listing['remarks'] = all_json_fields[listing['listingId']].get('remarks', '')
        if include_all_fields:
          listing.update(all_json_fields[listing['listingId']])

      return listings
  
  def normalize_scores(self, results: List[Dict], score_key: str) -> List[Dict]:
    if len(results) == 0: return results
    scores = [result[score_key] for result in results]
    min_score = min(scores)
    max_score = max(scores)
    if min_score == max_score:
      return results
    else:
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
        # Preserve remarks from text result if it exists
        if 'remarks' in result and result['remarks']:
          listings_dict[listingId]['remarks'] = result['remarks']

        # Add score to agg_score
        listings_dict[listingId]['agg_score'] += result['agg_score']
      else:
        # Add new listing to listings_dict
        if 'remarks' in result:
          listings_dict[listingId] = {
              'listingId': listingId,
              'agg_score': result['agg_score'],
              'remarks': result['remarks'],
              'image_names': []
          }
        else:
          listings_dict[listingId] = {
              'listingId': listingId,
              'agg_score': result['agg_score'],
              'image_names': []
          }

    # Convert listings_dict back to a list
    combined_results = list(listings_dict.values())

    return combined_results

  def _get_weaviate_filters(self, **filters):
    supported_keys = ['provState', 'city', 'bedsInt', 'bathsInt', 'price', 'leasePrice', 'transactionType']

    weaviate_filters = None
    for k, v in filters.items():
      if k not in supported_keys:
        raise ValueError(f"Filter key {k} is not supported.")
      
      if isinstance(v, (list, tuple)) and k in ['bedsInt', 'bathsInt', 'price']:
        if len(v) != 2 or not all(isinstance(i, (int, float, type(None))) for i in v):
          raise ValueError(f"Filter value for {k} should be a list or tuple of two numbers or None.")
        if weaviate_filters is None:
          if v[0] is not None:
            weaviate_filters = Filter.by_property(k).greater_or_equal(v[0])
          if v[1] is not None:
            weaviate_filters = (weaviate_filters & Filter.by_property(k).less_or_equal(v[1])) if weaviate_filters else Filter.by_property(k).less_or_equal(v[1])
        else:
          if v[0] is not None:
            weaviate_filters = weaviate_filters & Filter.by_property(k).greater_or_equal(v[0])
          if v[1] is not None:
            weaviate_filters = weaviate_filters & Filter.by_property(k).less_or_equal(v[1])
      else:
        if weaviate_filters is None:
          weaviate_filters = Filter.by_property(k).equal(v)
        else:
          weaviate_filters = weaviate_filters & Filter.by_property(k).equal(v)

    return weaviate_filters

  def _ping_impl(self) -> bool:
    return self.client.is_ready()

  # TODO: verify if this works on fastAPI context, this may be complicated.
  def ping(self) -> bool:
    return self.client.is_ready()

  # Helpers
  def _get_collection(self, collection_name: str):
    """Helper to get collection with consistency level if replication is enabled"""
    collection = self.client.collections.get(collection_name)
    if self.use_replication:
      collection = collection.with_consistency_level(self.consistency_level)
    return collection

class WeaviateDataStore_v4(WeaviateDataStore):
  def __init__(self, 
               client: weaviate.client.Client,
               image_embedder,
               text_embedder,
               score_aggregation_method = 'max',
               run_create_collection = False,
               use_replication = False,
               n_replications = 3,
               consistency_level: str = "ONE"
               ):
    super().__init__(image_embedder, text_embedder, score_aggregation_method,
                     use_replication=use_replication,
                     consistency_level=consistency_level)
    self.client = client
    self.n_replications = n_replications

    # TODO: work on the specific exception to catch
    if run_create_collection:
      try:
        self.create_collection()
      except weaviate.exceptions.UnexpectedStatusCodeException as e:
        if e.status_code == 422:
          self.logger.info('Schema already exists')
        else:
          raise e

  def close(self):
    self.client.close()

  def create_collection(self):
    common_properties = [
      Property(name="listing_id", data_type=DataType.TEXT, index_inverted=True),
      Property(name="city", data_type=DataType.TEXT, index_inverted=True),
      Property(name="provState", data_type=DataType.TEXT, index_inverted=True),
      Property(name="postalCode", data_type=DataType.TEXT),
      Property(name="lat", data_type=DataType.NUMBER),
      Property(name="lng", data_type=DataType.NUMBER),
      Property(name="streetName", data_type=DataType.TEXT),
      Property(name="beds", data_type=DataType.TEXT),
      Property(name="bedsInt", data_type=DataType.NUMBER, index_inverted=True),
      Property(name="baths", data_type=DataType.TEXT),
      Property(name="bathsInt", data_type=DataType.NUMBER, index_inverted=True),
      Property(name="sizeInterior", data_type=DataType.TEXT),
      Property(name="sizeInteriorUOM", data_type=DataType.TEXT),
      Property(name="lotSize", data_type=DataType.TEXT),
      Property(name="lotUOM", data_type=DataType.TEXT),
      Property(name="propertyFeatures", data_type=DataType.TEXT),
      Property(name="propertyType", data_type=DataType.TEXT),
      Property(name="transactionType", data_type=DataType.TEXT),
      Property(name="carriageTrade", data_type=DataType.BOOL),
      Property(name="price", data_type=DataType.NUMBER),
      Property(name="leasePrice", data_type=DataType.NUMBER),
      Property(name="pool", data_type=DataType.BOOL),
      Property(name="garage", data_type=DataType.BOOL),
      Property(name="waterFront", data_type=DataType.BOOL),
      Property(name="fireplace", data_type=DataType.BOOL),
      Property(name="ac", data_type=DataType.BOOL),
      Property(name="remarks", data_type=DataType.TEXT),
      Property(name="photo", data_type=DataType.TEXT),
      Property(name="listingDate", data_type=DataType.DATE),
      Property(name="lastUpdate", data_type=DataType.DATE),
      Property(name="lastPhotoUpdate", data_type=DataType.DATE)
  ]

    listing_image_properties = common_properties + [
      Property(name="image_name", data_type=DataType.TEXT)
    ]

    image_collection_config = {
      "vectorizer_config": Configure.Vectorizer.none(),
      "vector_index_config": Configure.VectorIndex.hnsw(
        distance_metric=weaviate.classes.config.VectorDistances.COSINE,
        max_connections=64,
        ef_construction=128,
        cleanup_interval_seconds=300,
        vector_cache_max_objects=2000000
      ),
      "properties": listing_image_properties,
      "inverted_index_config": Configure.inverted_index(
        stopwords_preset=weaviate.classes.config.StopwordsPreset.NONE,
        index_null_state=True,
        index_property_length=True,
        index_timestamps=True,
      )
    }

    if self.use_replication:
      image_collection_config["replication_config"] = Configure.replication(
        factor=self.n_replications
      )
    
    self.client.collections.create(
      name="Listing_Image",
      **image_collection_config
    )

    listing_text_properties = common_properties + [
      Property(name="remark_chunk_id", data_type=DataType.TEXT),
      Property(name="chunk_start", data_type=DataType.INT),
      Property(name="chunk_end", data_type=DataType.INT)
    ]

    text_collection_config = {
      "vectorizer_config": Configure.Vectorizer.none(),
      "vector_index_config": Configure.VectorIndex.hnsw(
          distance_metric=weaviate.classes.config.VectorDistances.COSINE,
          max_connections=64,
          ef_construction=128,
          cleanup_interval_seconds=300,
          vector_cache_max_objects=2000000
      ),
      "properties": listing_text_properties,
      "inverted_index_config": Configure.inverted_index(
          stopwords_preset=weaviate.classes.config.StopwordsPreset.NONE,
          index_null_state=True,
          index_property_length=True,
          index_timestamps=True
      )
    }

    if self.use_replication:
      text_collection_config["replication_config"] = Configure.replication(
        factor=self.n_replications
      )

    self.client.collections.create(
      name="Listing_Text",
      **text_collection_config
    )

    # self.client.collections.create(
    #   name="Listing_Image",
    #   vectorizer_config=Configure.Vectorizer.none(),
    #   vector_index_config=Configure.VectorIndex.hnsw(
    #       distance_metric=weaviate.classes.config.VectorDistances.COSINE,
    #       max_connections=64,
    #       ef_construction=128,
    #       cleanup_interval_seconds=300,
    #       vector_cache_max_objects=2000000
    #   ),
    #   properties=listing_image_properties,
    #   inverted_index_config=Configure.inverted_index(
    #       stopwords_preset=weaviate.classes.config.StopwordsPreset.NONE,
    #       index_null_state=True,
    #       index_property_length=True,
    #       index_timestamps=True,
    #   )
    # )

    # self.client.collections.create(
    #   name="Listing_Text",
    #   vectorizer_config=Configure.Vectorizer.none(),
    #   vector_index_config=Configure.VectorIndex.hnsw(
    #       distance_metric=weaviate.classes.config.VectorDistances.COSINE,
    #       max_connections=64,
    #       ef_construction=128,
    #       cleanup_interval_seconds=300,
    #       vector_cache_max_objects=2000000
    #   ),
    #   properties=listing_text_properties,
    #   inverted_index_config=Configure.inverted_index(
    #       stopwords_preset=weaviate.classes.config.StopwordsPreset.NONE,
    #       index_null_state=True,
    #       index_property_length=True,
    #       index_timestamps=True
    #   )
    # )


  def get_collections_config(self):
    def format_collection_config(collection_config):
      # Basic collection details
      formatted = {
        'name': collection_config.name,
        'description': collection_config.description or "no description provided",
        'vectorizer': collection_config.vectorizer,
        'properties': {},
        'multi_tenancy_config': {
          'enabled': collection_config.multi_tenancy_config.enabled,
          'auto_tenant_creation': collection_config.multi_tenancy_config.auto_tenant_creation
        },
        'replication_config': {
          'factor': collection_config.replication_config.factor
        },
        'sharding_config': {
          'virtual_per_physical': collection_config.sharding_config.virtual_per_physical,
          'desired_count': collection_config.sharding_config.desired_count,
          'actual_count': collection_config.sharding_config.actual_count,
          'desired_virtual_count': collection_config.sharding_config.desired_virtual_count,
          'actual_virtual_count': collection_config.sharding_config.actual_virtual_count,
          'key': collection_config.sharding_config.key,
          'strategy': collection_config.sharding_config.strategy,
          'function': collection_config.sharding_config.function
        },
        'vector_index_config': {
          'distance_metric': collection_config.vector_index_config.distance_metric,
          'dynamic_ef_min': collection_config.vector_index_config.dynamic_ef_min,
          'dynamic_ef_max': collection_config.vector_index_config.dynamic_ef_max,
          'dynamic_ef_factor': collection_config.vector_index_config.dynamic_ef_factor,
          'ef': collection_config.vector_index_config.ef,
          'ef_construction': collection_config.vector_index_config.ef_construction,
          'flat_search_cutoff': collection_config.vector_index_config.flat_search_cutoff,
          'max_connections': collection_config.vector_index_config.max_connections,
          'skip': collection_config.vector_index_config.skip,
          'vector_cache_max_objects': collection_config.vector_index_config.vector_cache_max_objects
        }
      }

      # Property details
      for prop in collection_config.properties:
        formatted['properties'][prop.name] = {
          'data_type': prop.data_type.name,
          'index_filterable': prop.index_filterable,
          'index_searchable': prop.index_searchable,
          'vectorizer': prop.vectorizer,
          'tokenization': prop.tokenization.name if prop.tokenization else "none"
        }

      # Inverted index configuration details
      if collection_config.inverted_index_config:
        inverted_index = collection_config.inverted_index_config
        formatted['inverted_index_config'] = {
          'bm25': {
            'b': inverted_index.bm25.b,
            'k1': inverted_index.bm25.k1
          },
          'cleanup_interval_seconds': inverted_index.cleanup_interval_seconds,
          'index_null_state': inverted_index.index_null_state,
          'index_property_length': inverted_index.index_property_length,
          'index_timestamps': inverted_index.index_timestamps,
          'stopwords': {
            'preset': inverted_index.stopwords.preset.name,
            'additions': inverted_index.stopwords.additions,
            'removals': inverted_index.stopwords.removals
          }
        }

      return formatted

    configs = []
    for name, _ in self.client.collections.list_all().items():
      _config = self.client.collections.get(name).config.get()
      configs.append(format_collection_config(_config))

    return configs

  def delete_all(self):
    # delete everything.
    for name, _ in self.client.collections.list_all().items():
       # Delete each collection, which also removes all its data objects
      self.client.collections.delete(name)

  @retry(**RETRY_SETTINGS)
  def delete_listing(self, listing_id: str, embedding_type: str = None):
    """
    Delete all objects related to a listing_id and embedding_type
    if embedding_type is None, delete all objects related to the listing_id
    """
    if embedding_type == 'I' or embedding_type is None:
      # listing_images = self.get(listing_id, embedding_type='I')  # image embeddings
      # for doc in listing_images:
      #   self._delete_object_by_uuid(doc['uuid'], 'Listing_Image')      
      # collection = self.client.collections.get("Listing_Image")
      collection = self._get_collection("Listing_Image")
      collection.data.delete_many(
        where=Filter.by_property("listing_id").equal(listing_id)
      )

    if embedding_type == 'T' or embedding_type is None:
      # listing_texts = self.get(listing_id, embedding_type='T')  # text embeddings
      # for doc in listing_texts:
      #   self._delete_object_by_uuid(doc['uuid'], 'Listing_Text')
      # collection = self.client.collections.get("Listing_Text")
      collection = self._get_collection("Listing_Text")
      collection.data.delete_many(
        where=Filter.by_property("listing_id").equal(listing_id)
      )

  def delete_listings(self, listing_ids: List[str], embedding_type: str = None):
    """
    Delete objects related to multiple listing_ids and embedding_type.
    If embedding_type is None, delete all objects related to the listing_ids.
    """
    for listing_id in tqdm(listing_ids):
      self.delete_listing(listing_id, embedding_type=embedding_type)

  @retry(**RETRY_SETTINGS)
  def delete_listings_by_batch(self, listing_ids: List[str], embedding_type: str = None, batch_size: int = 100, sleep_time: int = 0) -> Dict[str, Any]:
    """
    Delete objects related to multiple listing_ids and embedding_type by batch of batch_size.
    If embedding_type is None, delete all objects related to the listing_ids.

    Args:
      listing_ids: List of listing IDs to delete
      embedding_type: 'I' for image, 'T' for text, None for both
      batch_size: Number of listings to process in each batch
      sleep_time: Time to sleep between batches
      
    Returns:
        Dict containing deletion statistics:
        {
            'total_objects_deleted': Number of objects deleted,
            'failed_batches': List of failed batch information,
            'error_count': Total number of errors encountered
        }
    """
    stats = {
      'total_objects_deleted': 0,
      'failed_batches': [],
      'error_count': 0
    }

    try:
      count_before = self.count_all()

      if embedding_type == 'I' or embedding_type is None:
        # collection = self.client.collections.get("Listing_Image")
        collection = self._get_collection("Listing_Image")
        for i in tqdm(range(0, len(listing_ids), batch_size), desc="Deleting image embeddings"):
          batch_ids = listing_ids[i:i+batch_size]
          try:
            result = collection.data.delete_many(
              where=Filter.by_property("listing_id").contains_any(batch_ids)
            )
            stats['error_count'] += result.failed
            time.sleep(sleep_time)
          except (UnexpectedStatusCodeError, ConnectionError, TimeoutError) as e:
            self.logger.error("UnexpectedStatusCodeError, ConnectionError, TimeoutError during batch deletion")
            error_info = {
              'batch_start_idx': i,
              'listing_ids': batch_ids,
              'error': str(e),
              'type': 'image'
            }
            stats['failed_batches'].append(error_info)
            stats['error_count'] += 1
            self.logger.error(f"Failed to delete image batch starting at index {i}: {str(e)}")
      
      if embedding_type == 'T' or embedding_type is None:
        # collection = self.client.collections.get("Listing_Text")
        collection = self._get_collection("Listing_Text")
        for i in tqdm(range(0, len(listing_ids), batch_size), desc="Deleting text embeddings"):
          batch_ids = listing_ids[i:i+batch_size]
          try:
            result = collection.data.delete_many(
              where=Filter.by_property("listing_id").contains_any(batch_ids)
            )
            stats['error_count'] += result.failed
            time.sleep(sleep_time)
          except (UnexpectedStatusCodeError, ConnectionError, TimeoutError) as e:
            self.logger.error("UnexpectedStatusCodeError, ConnectionError, TimeoutError during batch deletion")
            error_info = {
              'batch_start_idx': i,
              'listing_ids': batch_ids,
              'error': str(e),
              'type': 'text'
            }
            stats['failed_batches'].append(error_info)
            stats['error_count'] += 1
            self.logger.error(f"Failed to delete text batch starting at index {i}: {str(e)}")

      stats['total_objects_deleted'] = count_before - self.count_all()
      if stats['error_count'] > 0:
        self.logger.error(
          f"Completed with {stats['error_count']} failed batches. "
          f"Deleted {stats['total_objects_deleted']} objects. "
          "See 'failed_batches' in return value for details."
        )
      else:
        self.logger.info(f"Successfully deleted {stats['total_objects_deleted']} objects")

      return stats
    except Exception as e:
      self.logger.error(f"Fatal error in batch deletion: {str(e)}")
      stats['fatal_error'] = str(e)
      return stats



  def count_all(self) -> int:
    count = 0
    for name, _ in self.client.collections.list_all().items():
      count += self._count_by_collection_name(name)

    return count

  def get(self, listing_id: Optional[str] = None, embedding_type: str = 'T', include_vector=False):
    """
    Retrieve items from Weaviate related to listing_id and embedding_type.
    If listing_id is None, retrieve all items.
    embedding_type: 'I' for image, 'T' for text.
    """
    # this should be safe as we don't expect more than 1000 items for one listing_id
    offset, limit = 0, 1000  

    collection_name = "Listing_Image" if embedding_type == 'I' else "Listing_Text"
    extra_properties = ['image_name'] if embedding_type == 'I' else ['remark_chunk_id']

    # collection = self.client.collections.get(collection_name)
    collection = self._get_collection(collection_name)

    # Create filter if listing_id is provided
    if listing_id:
      filters = Filter.by_property("listing_id").equal(listing_id) if listing_id else None

    # Fetch objects with filters and a limit, if applicable
    response = collection.query.fetch_objects(
      filters=filters,
      include_vector=include_vector,
      limit=limit  
    )

    results = []
    for o in response.objects:    
      listing_json = self._postprocess_listing_json(o.properties)
      listing_json['uuid'] = o.uuid.__str__()

      if include_vector:
        listing_json['embedding'] = o.vector['default']

      results.append(listing_json)

    return results
  
  def get_all(self, embedding_type: str = 'I') -> List[Dict]:
    """
    Retrieve all items related to embedding_type.
    """
    collection_name = "Listing_Image" if embedding_type == 'I' else "Listing_Text"
    # collection = self.client.collections.get(collection_name)
    collection = self._get_collection(collection_name)

    all_objects = []
    for o in tqdm(collection.iterator(
        # include_vector=True  
    )):
      listing_json = self._postprocess_listing_json(o.properties)
      listing_json['uuid'] = o.uuid.__str__()
      all_objects.append(listing_json)

    return all_objects
    
  def get_unique_listing_ids(self) -> List[str]:
    listing_ids = set()
    for listing in self.get_all(embedding_type='I'):
      listing_ids.add(listing['listing_id'])
    for listing in self.get_all(embedding_type='T'):
      listing_ids.add(listing['listing_id'])

    return list(listing_ids)

  def insert(self, listing_json: Dict, embedding_type: str = 'I'):
    collection_name = "Listing_Image" if embedding_type == 'I' else "Listing_Text"
    # collection = self.client.collections.get(collection_name)
    collection = self._get_collection(collection_name)

    listing_json = self._preprocess_listing_json(listing_json, embedding_type=embedding_type)
    key = self._create_key(listing_json, embedding_type)

    if not 'embedding' in listing_json or listing_json['embedding'] is None or len(listing_json['embedding']) == 0:
      raise ValueError("The listing_json must contain an 'embedding' field with a non-empty vector.")

    vector = listing_json.pop('embedding')

    try:
      uuid = collection.data.insert(
          properties=listing_json,
          vector=vector,
          uuid=key
      )
      return uuid
    except UnexpectedStatusCodeError as e:
      self.logger.info(f"{e.message}")
      return None

  def upsert(self, listing_json: Dict, embedding_type: str = 'I'):
    collection_name = "Listing_Image" if embedding_type == 'I' else "Listing_Text"
    # collection = self.client.collections.get(collection_name
    collection = self._get_collection(collection_name)

    listing_json = self._preprocess_listing_json(listing_json, embedding_type=embedding_type)
    key = self._create_key(listing_json, embedding_type)

    if not 'embedding' in listing_json or listing_json['embedding'] is None or len(listing_json['embedding']) == 0:
      raise ValueError("The listing_json must contain an 'embedding' field with a non-empty vector.")
    
    vector = listing_json.pop('embedding')

    try:
      existing_object = collection.query.fetch_object_by_id(key)
      if existing_object:
        collection.data.update(
          uuid=key,
          properties=listing_json,
          vector=vector
        )
      else:
        collection.data.insert(
          properties=listing_json,
          vector=vector,
          uuid=key
        )
      return key
    
    except Exception as e:
      self.logger.error(f"Error upserting object: {e}")
      return None

  @retry(**RETRY_SETTINGS)
  def raw_export_all(self, embedding_type: str, output_path: Path, chunk_size=100000) -> int:
    """
    Export/dump all raw data from Weaviate without post-processing.
    
    Export all raw data from Weaviate in chunks to separate pickle files.
  
    Args:
      embedding_type: Required string indicating collection type ('I' for images, 'T' for text)
      output_path: Base path for pickle files (without extension)
      chunk_size: Number of records per pickle file
    
    Returns:
      int: Total number of records exported
    """
    if embedding_type not in ['I', 'T']:
      raise ValueError("embedding_type must be either 'I' for images or 'T' for text")

    collection_name = "Listing_Image" if embedding_type == 'I' else "Listing_Text"
    # collection = self.client.collections.get(collection_name)
    collection = self._get_collection(collection_name)

    total_count = collection.aggregate.over_all().total_count
    current_chunk = []
    chunk_num = 0
    total_processed = 0
    
    for item in tqdm(collection.iterator(include_vector=True), total=total_count, desc=f"Exporting {collection_name}"):
      raw_data = {
        **item.properties,
        'vector': item.vector['default']
      }
      current_chunk.append(raw_data)
      total_processed += 1

      # when chunk is full, save it and start a new one
      if len(current_chunk) >= chunk_size:
        chunk_file = Path(f"{output_path}_{chunk_num}.pkl")
        save_to_pickle(current_chunk, chunk_file, compressed=True)
        current_chunk = []
        chunk_num += 1
        gc.collect()

    # save the last chunk
    if current_chunk:
      chunk_file = Path(f"{output_path}_{chunk_num}.pkl")
      save_to_pickle(current_chunk, chunk_file, compressed=True)

    return total_processed
  
  @retry(**RETRY_SETTINGS)
  def raw_import_all(self, input_path: Path, embedding_type: str, batch_size=1000, sleep_time=0.5, max_chunks=None) -> int:
    """
    Import data previously exported using raw_export_all back into Weaviate.
    Handles multiple pickle files in format input_path_N.pkl.
    Preserves data exactly as exported while assigning proper UUIDs based on listing data.
    
    Args:
      input_path: Base path of the pickle files (without extension and chunk number)
      embedding_type: Required string indicating collection type ('I' for images, 'T' for text)
      batch_size: Number of items to process in each batch
      sleep_time: Sleep time between batches to prevent overload
      max_chunks: Maximum number of chunk files to process (None for no limit)
    
    Returns:
      int: Total number of records imported
    """
    if embedding_type not in ['I', 'T']:
      raise ValueError("embedding_type must be either 'I' for images or 'T' for text")
   
    collection_name = "Listing_Image" if embedding_type == 'I' else "Listing_Text"
    # collection = self.client.collections.get(collection_name)
    collection = self._get_collection(collection_name)

    def chunks(iterable, size):
      for i in range(0, len(iterable), size):
        yield iterable[i:i + size]

    total_imported = 0
    chunk_num = 0

    while True:
      if max_chunks is not None and chunk_num >= max_chunks:
        self.logger.info(f"Reached max_chunks limit ({max_chunks}). Stopping import.")
        break
      
      chunk_file = Path(f"{input_path}_{chunk_num}.pkl")
      if not chunk_file.exists():
        break

      try:
        listings = load_from_pickle(chunk_file, compressed=True)

        for batch_listings in chunks(list(listings), batch_size):
          with collection.batch.dynamic() as batch_writer:
            for listing_json in tqdm(batch_listings):
              key = self._create_key(listing_json, embedding_type)
              vector = listing_json.pop('vector')
              batch_writer.add_object(
                properties=listing_json,
                vector=vector,
                uuid=key
              )
          time.sleep(sleep_time)
        
        total_imported += len(listings)
        chunk_num += 1
        gc.collect()
      except Exception as e:
        self.logger.error(f"Error importing chunk {chunk_num}: {e}")
        raise

    # try:
    #   for batch_listings in chunks(list(listings), batch_size):
    #     with collection.batch.dynamic() as batch_writer:
    #       for listing_json in tqdm(batch_listings):
            
    #         key = self._create_key(listing_json, embedding_type)
    #         vector = listing_json.pop('vector')
            
    #         batch_writer.add_object(
    #           properties=listing_json,
    #           vector=vector,
    #           uuid=key
    #         )
    #     time.sleep(sleep_time)
    # except Exception as e:
    #   self.logger.error(f"Error batch inserting objects: {e}")
    #   raise

    return total_imported

  @retry(**RETRY_SETTINGS)
  def batch_insert(self, listings: Iterable[Dict], embedding_type: str = 'I', batch_size=1000, sleep_time=1) -> Dict[str, List[Dict]]:
    """
    Batch insert listings into Weaviate.
    
    Args:
      listings: Iterable of listing dictionaries to insert
      embedding_type: 'I' for image embeddings, 'T' for text embeddings
      batch_size: Size of each batch
      sleep_time: Time to sleep between batches
    
    Returns:
      Dict containing:
        - failed_objects: list of objects that failed to insert
    """
    collection_name = "Listing_Image" if embedding_type == 'I' else "Listing_Text"
    # collection = self.client.collections.get(collection_name)
    collection = self._get_collection(collection_name)

    def chunks(iterable, size):
      for i in range(0, len(iterable), size):
        yield iterable[i:i + size]
                       
    total_processed = 0
    total_failed = 0
    all_failed_objects = []

    try:
      for batch_listings in chunks(list(listings), batch_size):        
        with collection.batch.dynamic() as batch:
          for listing_json in tqdm(batch_listings):
            listing_json = self._preprocess_listing_json(listing_json, embedding_type=embedding_type)
            key = self._create_key(listing_json, embedding_type)
            vector = listing_json.pop('embedding')

            batch.add_object(
              properties=listing_json,
              uuid=key,
              vector=vector
            )
            total_processed += 1

        # After the batch is processed, check for failed objects
        failed_objects = collection.batch.failed_objects

        # if hasattr(batch, 'failed_objects') and batch.failed_objects:
        failed_count = len(failed_objects)
        total_failed += failed_count
        all_failed_objects.extend(failed_objects)
        
        self.logger.info(f"Batch insertion had {failed_count} failures")
        
        if failed_count > 0:
          # Log details of first few failed objects (to avoid excessive logging)
          for failed_obj in failed_objects[:3]:  # Show first 3 failures
            self.logger.error(f"Failed object example: {failed_obj.object_.properties.get('listing_id')}")
        
          if failed_count > 3:
            self.logger.error(f"... and {failed_count - 3} more failures")

        time.sleep(sleep_time)

      # Log final statistics
      success_rate = ((total_processed - total_failed) / total_processed * 100) if total_processed > 0 else 0

      self.logger.info(f"""
      Batch insertion completed:
      - Total processed: {total_processed}
      - Successfully inserted: {total_processed - total_failed}
      - Failed insertions: {total_failed}
      - Success rate: {success_rate:.2f}%
      """)

      return {"failed_objects": all_failed_objects}

    except Exception as e:
      self.logger.error(f"Error batch inserting objects: {e}")
      raise

  @retry(**RETRY_SETTINGS)
  def batch_upsert(self, listings: Iterable[Dict], embedding_type: str = 'I'):
    collection_name = "Listing_Image" if embedding_type == 'I' else "Listing_Text"
    # collection = self.client.collections.get(collection_name)
    collection = self._get_collection(collection_name)

    try:
      with collection.batch.dynamic() as batch:
        for listing_json in tqdm(listings):
          listing_json = self._preprocess_listing_json(listing_json, embedding_type=embedding_type)
          key = self._create_key(listing_json, embedding_type)
          vector = listing_json.pop('embedding')

          existing_object = collection.query.fetch_object_by_id(key)
          if existing_object:
            batch.update_object(
              uuid=key,
              properties=listing_json,
              vector=vector
            )
          else:
            batch.add_object(
              properties=listing_json,
              uuid=key,
              vector=vector
            )
    except Exception as e:
      self.logger.error(f"Error batch upserting objects: {e}")

  @retry(**RETRY_SETTINGS)
  def _search_image_2_image(self, image: Image.Image = None, embedding: List[float] = None, topk=5, group_by_listingId=False, include_all_fields=False, **filters):
    embedding_type = 'I'   # targets are images
    collection_name = "Listing_Image" if embedding_type == 'I' else "Listing_Text"
    # collection = self.client.collections.get(collection_name)
    collection = self._get_collection(collection_name)

    if embedding is None:
      embedding = self.image_embedder.embed_from_single_image(image).flatten().tolist()

    weaviate_filters = self._get_weaviate_filters(**filters)

    response = collection.query.near_vector(
      near_vector=embedding,
      limit=topk,
      return_metadata=MetadataQuery(distance=True),
      filters=weaviate_filters
    )

    top_image_names, top_scores = [], []
    all_json_fields = {}
    for o in response.objects:
      top_image_names.append(o.properties['image_name'])
      score = (1 - o.metadata.distance)   # such that higher score is better match
      top_scores.append(score)

      json_fields = self._postprocess_listing_json(o.properties)

      listing_id = o.properties['listing_id']
      if include_all_fields:
        all_json_fields[listing_id] = json_fields
      else:
        # only incl. remarks
        all_json_fields[listing_id] = {'remarks': json_fields.get('remarks', '')}

    if group_by_listingId:
      return self._image_search_groupby_listing(top_image_names, top_scores, include_all_fields, all_json_fields)

    return top_image_names, top_scores
      
  @retry(**RETRY_SETTINGS)
  def _search_image_2_text(self, image: Image.Image = None, embedding: List[float] = None, topk=5, group_by_listingId=False, include_all_fields=False, **filters):
    embedding_type = 'T'   # targets are text
    collection_name = "Listing_Image" if embedding_type == 'I' else "Listing_Text"
    # collection = self.client.collections.get(collection_name)
    collection = self._get_collection(collection_name)

    if embedding is None:
      embedding = self.image_embedder.embed_from_single_image(image).flatten().tolist()

    weaviate_filters = self._get_weaviate_filters(**filters)

    response = collection.query.near_vector(
      near_vector=embedding,
      limit=topk,
      return_metadata=MetadataQuery(distance=True),
      filters=weaviate_filters
    )

    top_remark_chunk_ids, top_scores, top_positions = [], [], []
    all_json_fields = {}

    for o in response.objects:
      top_remark_chunk_ids.append(o.properties['remark_chunk_id'])
      score = (1 - o.metadata.distance)   # such that higher score is better match
      top_scores.append(score)

      start = o.properties['chunk_start']
      end = o.properties['chunk_end']
      start = int(start) if start is not None else 0
      end = int(end) if end is not None else 0
      top_positions.append((start, end))

      json_fields = self._postprocess_listing_json(o.properties)

      listing_id = o.properties['listing_id']
      if include_all_fields:
        all_json_fields[listing_id] = json_fields
      else:
        # only incl. remarks
        all_json_fields[listing_id] = {'remarks': json_fields.get('remarks', '')}

    if group_by_listingId:
      return self._text_search_groupby_listing(top_remark_chunk_ids, top_scores, top_positions, include_all_fields, all_json_fields)
 
    return top_remark_chunk_ids, top_scores

  @retry(**RETRY_SETTINGS)
  def _search_text_2_image(self, phrase: str = None, embedding: List[float] = None, topk=5, group_by_listingId=False, include_all_fields=False, **filters):
    embedding_type = 'I'   # targets are images
    collection_name = "Listing_Image" if embedding_type == 'I' else "Listing_Text"
    # collection = self.client.collections.get(collection_name)
    collection = self._get_collection(collection_name)

    if embedding is None:
      embedding = self.text_embedder.embed_from_texts([phrase], batch_size=1)[0].flatten().tolist()

    weaviate_filters = self._get_weaviate_filters(**filters)

    response = collection.query.near_vector(
      near_vector=embedding,
      limit=topk,
      return_metadata=MetadataQuery(distance=True),
      filters=weaviate_filters
    )

    top_image_names, top_scores = [], []
    all_json_fields = {}
    for o in response.objects:
      top_image_names.append(o.properties['image_name'])
      score = (1 - o.metadata.distance)
      top_scores.append(score)

      json_fields = self._postprocess_listing_json(o.properties)

      listing_id = o.properties['listing_id']
      if include_all_fields:
        all_json_fields[listing_id] = json_fields
      else:
        # only incl. remarks
        all_json_fields[listing_id] = {'remarks': json_fields.get('remarks', '')}
    
    if group_by_listingId:
      return self._image_search_groupby_listing(top_image_names, top_scores, include_all_fields, all_json_fields)
    
    return top_image_names, top_scores

  @retry(**RETRY_SETTINGS)
  def _search_text_2_text(self, phrase: str = None, embedding: List[float] = None, topk=5, group_by_listingId=False, include_all_fields=False, **filters):
    embedding_type = 'T'   # targets are text
    collection_name = "Listing_Image" if embedding_type == 'I' else "Listing_Text"
    # collection = self.client.collections.get(collection_name)
    collection = self._get_collection(collection_name)

    if embedding is None:
      embedding = self.text_embedder.embed_from_texts([phrase], batch_size=1)[0].flatten().tolist()

    weaviate_filters = self._get_weaviate_filters(**filters)
    
    response = collection.query.near_vector(
      near_vector=embedding,
      limit=topk,
      return_metadata=MetadataQuery(distance=True),
      filters=weaviate_filters
    )

    top_remark_chunk_ids, top_scores, top_positions = [], [], []
    all_json_fields = {}

    for o in response.objects:
      top_remark_chunk_ids.append(o.properties['remark_chunk_id'])
      score = (1 - o.metadata.distance)
      top_scores.append(score)

      start = o.properties['chunk_start']
      end = o.properties['chunk_end']
      start = int(start) if start is not None else 0
      end = int(end) if end is not None else 0
      top_positions.append((start, end))

      json_fields = self._postprocess_listing_json(o.properties)

      listing_id = o.properties['listing_id']
      if include_all_fields:
        all_json_fields[listing_id] = json_fields
      else:
        # only incl. remarks
        all_json_fields[listing_id] = {'remarks': json_fields.get('remarks', '')}

    if group_by_listingId:
      return self._text_search_groupby_listing(top_remark_chunk_ids, top_scores, top_positions, include_all_fields, all_json_fields)
    
    return top_remark_chunk_ids, top_scores
      
  
  def search(self,
             image: Image.Image = None,
             image_embedding: List[float] = None,
             phrase: str = None,
             text_embedding: List[float] = None,
             topk=5,
             group_by_listingId=False,
             include_all_fields=False,
             **filters):
    if (not image and not phrase) and (not image_embedding and not text_embedding):
      if group_by_listingId: return []
      else: return [], []

    listing_info = {}
    combined_results = []
    if image or image_embedding:
      if image_embedding is None:
        image_embedding = self.image_embedder.embed_from_single_image(image).flatten().tolist()

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
        text_embedding = self.text_embedder.embed_from_texts([phrase], batch_size=1)[0].flatten().tolist()

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
    

  def multi_image_search(self, 
                         images: List[Image.Image],
                         phrase: str = None,
                         topk=5,
                         group_by_listingId=False,
                         include_all_fields=False,
                         **filters
                         ):
    # use mean(embeddings) for now 
    self.logger.info(f'Number of images: {len(images)}')
    all_image_embeddings = []
    for image in images:
      image_embedding = self.image_embedder.embed_from_single_image(image)
      all_image_embeddings.append(image_embedding)
    mean_vector = np.mean(all_image_embeddings, axis=0)

    image_embedding = mean_vector.flatten().tolist()

    text_embedding = None
    if phrase:
      text_embedding = self.text_embedder.embed_from_texts([phrase], batch_size=1)[0].flatten().tolist()

    return self.search(image_embedding=image_embedding, 
                       text_embedding=text_embedding, 
                       topk=topk, 
                       group_by_listingId=group_by_listingId, 
                       include_all_fields=include_all_fields, 
                       **filters)



  def import_from_faiss_index(self, 
                              faiss_index: Any,    # a FaissIndex object
                              listing_df: pd.DataFrame, 
                              embedding_type: str = 'I', 
                              offset: int = None, 
                              length: int = None):
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
  

  def _delete_object_by_uuid(self, uuid, collection_name):
    try:
      # Delete the object based on its UUID and collection name
      # collection = self.client.collections.get(collection_name)
      collection = self._get_collection(collection_name)
      collection.data.delete_by_id(uuid)
      self.logger.info(f"Object with UUID {uuid} successfully deleted.")

    except Exception as e:
      self.logger.error(f"Error deleting object with UUID {uuid}: {e}")


  def _count_by_collection_name(self, collection_name: str) -> int:
    try:
      # Count the number of objects in the class
      # collection = self.client.collections.get(collection_name)
      collection = self._get_collection(collection_name)
      return collection.aggregate.over_all().total_count

    except Exception as e:
      self.logger.error(f"Error counting objects in collection {collection_name}: {e}")
      return None


  # these are methods that conform to class ListingSearchEngine interface
  # which is also expected by main.py the fastAPI app
  def get_listing(self, listing_id: str) -> Dict[str, Any]:
    listing_docs = self.get(listing_id, embedding_type='T')
    if len(listing_docs) == 0: 
      listing_docs = self.get(listing_id, embedding_type='I')
      if len(listing_docs) == 0:
        return {}
      
    listing_doc = listing_docs[0]

    # add jumpId
    listing_doc['jumpId'] = listing_doc['listing_id']

    return listing_doc

  def get_imagenames(self, listingId: str) -> List[str]:
    listing_docs = self.get(listingId, embedding_type='I')
    if len(listing_docs) == 0: return []

    return [doc['image_name'] for doc in listing_docs]
      


class AsyncWeaviateDataStore_v4(WeaviateDataStore):
  def __init__(self, 
                async_client: WeaviateAsyncClient,
                image_embedder,
                text_embedder,
                score_aggregation_method = 'max',
                use_replication = False,
                consistency_level: str = "ONE"
              ):
    """
    Initialize the AsyncWeaviateDataStore_v4 with an asynchronous Weaviate client.
    
    Args:
        async_client (WeaviateAsyncClient): The asynchronous Weaviate client.
        image_embedder: The image embedder instance.
        text_embedder: The text embedder instance.
        score_aggregation_method (str): Method to aggregate scores ('max' or 'mean').
    """
    super().__init__(image_embedder, text_embedder, score_aggregation_method,
                     use_replication=use_replication,
                     consistency_level=consistency_level
                     )
    self.client = async_client


  async def ping(self) -> bool:
    return await self.client.is_ready()
  
  def is_connected(self) -> bool:
    return self.client.is_connected()


  async def close(self):
    await self.client.close()
  
  async def get(self, listing_id: Optional[str] = None, embedding_type: str = 'T', include_vector=False):
    """
    Retrieve items from Weaviate related to listing_id and embedding_type.
    If listing_id is None, retrieve all items.
    embedding_type: 'I' for image, 'T' for text.
    """
    # this should be safe as we don't expect more than 1000 items for one listing_id
    offset, limit = 0, 1000  

    collection_name = "Listing_Image" if embedding_type == 'I' else "Listing_Text"
    extra_properties = ['image_name'] if embedding_type == 'I' else ['remark_chunk_id']

    # collection = self.client.collections.get(collection_name)
    collection = self._get_collection(collection_name)

    # Create filter if listing_id is provided
    if listing_id:
      filters = Filter.by_property("listing_id").equal(listing_id) if listing_id else None

    # Fetch objects with filters and a limit, if applicable
    response = await collection.query.fetch_objects(
      filters=filters,
      include_vector=include_vector,
      limit=limit  
    )

    results = []
    for o in response.objects:    
      listing_json = self._postprocess_listing_json(o.properties)
      listing_json['uuid'] = o.uuid.__str__()

      if include_vector:
        listing_json['embedding'] = o.vector['default']

      results.append(listing_json)

    return results
  
  async def get_listing(self, listing_id: str) -> Dict[str, Any]:
    """
    Retrieve a single listing by ID, trying image embeddings first then text embeddings.
    
    Args:
        listing_id (str): The ID of the listing to retrieve
    
    Returns:
        Dict[str, Any]: Listing document with 'jumpId' added, or empty dict if not found
    """
    listing_docs = await self.get(listing_id, embedding_type='T')
    if len(listing_docs) == 0:
      listing_docs = await self.get(listing_id, embedding_type='I')
      if len(listing_docs) == 0:
        return {}

    listing_doc = listing_docs[0]
    # add jumpId
    listing_doc['jumpId'] = listing_doc['listing_id']
    return listing_doc

  async def get_imagenames(self, listingId: str) -> List[str]:
    """
    Retrieve all image names for a given listing ID.
    
    Args:
        listingId (str): The ID of the listing to get images for
    
    Returns:
        List[str]: List of image names, or empty list if no images found
    """
    listing_docs = await self.get(listingId, embedding_type='I')
    if len(listing_docs) == 0: 
      return []
    
    return [doc['image_name'] for doc in listing_docs]
  
  @retry(**RETRY_SETTINGS)
  async def _search(self, 
                    embedding_type: str,
                    image: Image.Image = None, 
                    phrase: str = None, 
                    embedding: List[float] = None, 
                    topk: int = 5, 
                    group_by_listingId: bool = False, 
                    include_all_fields: bool = False, 
                    **filters) -> Union[List[Dict[str, Union[str, float, List[str]]]], Tuple[List[str], List[float]]]:
    """
    Common search method for both image and text searches.
    
    Args:
        embedding_type (str): 'I' for image-related searches or 'T' for text-related searches.
    """
    collection_name = "Listing_Image" if embedding_type == 'I' else "Listing_Text"
    # collection = self.client.collections.get(collection_name)
    collection = self._get_collection(collection_name)

    if embedding is None:
      if image is not None:
        embedding = self.image_embedder.embed_from_single_image(image).flatten().tolist()
      elif phrase is not None:
        embedding = self.text_embedder.embed_from_texts([phrase], batch_size=1)[0].flatten().tolist()
      else:
        raise ValueError("Either image or phrase must be provided to generate an embedding.")

    weaviate_filters = self._get_weaviate_filters(**filters)

    try:
      response = await collection.query.near_vector(
        near_vector=embedding,
        limit=topk,
        return_metadata=MetadataQuery(distance=True),
        filters=weaviate_filters
      )
    except UnexpectedStatusCodeError as e:
      self.logger.error(f"Search failed: {e}")
      raise

    top_ids, top_scores = [], []
    top_positions = []     # only for text search results
    all_json_fields = {}
    property_key = 'image_name' if embedding_type == 'I' else 'remark_chunk_id'

    for o in response.objects:
      top_ids.append(o.properties[property_key])
      score = (1 - o.metadata.distance)   # such that higher score is better match
      top_scores.append(score)

      if embedding_type == 'T':
        start = o.properties['chunk_start']
        end = o.properties['chunk_end']
        start = int(start) if start is not None else 0
        end = int(end) if end is not None else 0
        top_positions.append((start, end))

      json_fields = self._postprocess_listing_json(o.properties)

      listing_id = o.properties['listing_id']
      if include_all_fields:
        all_json_fields[listing_id] = json_fields
      else:      
        all_json_fields[listing_id] = {'remarks': json_fields.get('remarks', '')}   # only incl. remarks      

    if group_by_listingId:
      if embedding_type == 'I':
        return self._image_search_groupby_listing(top_ids, top_scores, include_all_fields, all_json_fields)
      else:
        return self._text_search_groupby_listing(top_ids, top_scores, top_positions, include_all_fields, all_json_fields)

    return top_ids, top_scores


  @retry(**RETRY_SETTINGS)
  async def _search_image(self, 
                          embedding_type: str,
                          image: Image.Image = None, 
                          embedding: List[float] = None, 
                          topk: int = 5, 
                          group_by_listingId: bool = False, 
                          include_all_fields: bool = False, 
                          **filters) -> Union[List[Dict[str, Union[str, float, List[str]]]], Tuple[List[str], List[float]]]:
    return await self._search(
      embedding_type=embedding_type,
      image=image,
      embedding=embedding,
      topk=topk,
      group_by_listingId=group_by_listingId,
      include_all_fields=include_all_fields,
      **filters
    )
  

  @retry(**RETRY_SETTINGS)
  async def _search_text(self, 
                         embedding_type: str,
                         phrase: str = None, 
                         embedding: List[float] = None, 
                         topk: int = 5, 
                         group_by_listingId: bool = False, 
                         include_all_fields: bool = False, 
                         **filters) -> Union[List[Dict[str, Union[str, float, List[str]]]], Tuple[List[str], List[float]]]:
    return await self._search(
      embedding_type=embedding_type,
      phrase=phrase,
      embedding=embedding,
      topk=topk,
      group_by_listingId=group_by_listingId,
      include_all_fields=include_all_fields,
      **filters
    )
  

  @retry(**RETRY_SETTINGS)
  async def _search_image_2_image(self, 
                                  image: Image.Image = None, 
                                  embedding: List[float] = None, 
                                  topk: int = 5, 
                                  group_by_listingId: bool = False, 
                                  include_all_fields: bool = False, 
                                  **filters):
    
    if group_by_listingId:
      listings = await self._search_image(
        embedding_type='I',
        image=image,
        embedding=embedding,
        topk=topk,
        group_by_listingId=group_by_listingId,
        include_all_fields=include_all_fields,
        **filters
      )
      return listings
    else:
      top_image_names, top_scores = await self._search_image(
        embedding_type='I',
        image=image,
        embedding=embedding,
        topk=topk,
        group_by_listingId=group_by_listingId,
        include_all_fields=include_all_fields,
        **filters
      )
      return top_image_names, top_scores
    
  
  @retry(**RETRY_SETTINGS)
  async def _search_image_2_text(self, 
                                 image: Image.Image = None, 
                                 embedding: List[float] = None, 
                                 topk: int = 5, 
                                 group_by_listingId: bool = False, 
                                 include_all_fields: bool = False, 
                                 **filters):
    if group_by_listingId:
      listings = await self._search_image(
        embedding_type='T',
        image=image,
        embedding=embedding,
        topk=topk,
        group_by_listingId=group_by_listingId,
        include_all_fields=include_all_fields,
        **filters
      )
      return listings
    else:
      top_remark_chunk_ids, top_scores = await self._search_image(
        embedding_type='T',
        image=image,
        embedding=embedding,
        topk=topk,
        group_by_listingId=group_by_listingId,
        include_all_fields=include_all_fields,
        **filters
      )
      return top_remark_chunk_ids, top_scores


  @retry(**RETRY_SETTINGS)
  async def _search_text_2_image(self, 
                                 phrase: str = None, 
                                 embedding: List[float] = None, 
                                 topk: int = 5, 
                                 group_by_listingId: bool = False, 
                                 include_all_fields: bool = False, 
                                 **filters):
    if group_by_listingId:
      listings = await self._search_text(
        embedding_type='I',
        phrase=phrase,
        embedding=embedding,
        topk=topk,
        group_by_listingId=group_by_listingId,
        include_all_fields=include_all_fields,
        **filters
      )
      return listings
    else:
      top_image_names, top_scores = await self._search_text(
        embedding_type='I',
        phrase=phrase,
        embedding=embedding,
        topk=topk,
        group_by_listingId=group_by_listingId,
        include_all_fields=include_all_fields,
        **filters
      )
      return top_image_names, top_scores
  

  @retry(**RETRY_SETTINGS)
  async def _search_text_2_text(self, 
                                phrase: str = None, 
                                embedding: List[float] = None, 
                                topk: int = 5, 
                                group_by_listingId: bool = False, 
                                include_all_fields: bool = False, 
                                **filters):
    if group_by_listingId:
      listings = await self._search_text(
        embedding_type='T',
        phrase=phrase,
        embedding=embedding,
        topk=topk,
        group_by_listingId=group_by_listingId,
        include_all_fields=include_all_fields,
        **filters
      )
      return listings
    else:
      top_remark_chunk_ids, top_scores = await self._search_text(
        embedding_type='T',
        phrase=phrase,
        embedding=embedding,
        topk=topk,
        group_by_listingId=group_by_listingId,
        include_all_fields=include_all_fields,
        **filters
      )
      return top_remark_chunk_ids, top_scores
    
  @retry(**RETRY_SETTINGS)
  async def search(self,
                    image: Image.Image = None,
                    image_embedding: List[float] = None,
                    phrase: str = None,
                    text_embedding: List[float] = None,
                    topk: int = 5,
                    group_by_listingId: bool = False,
                    include_all_fields: bool = False,
                    **filters) -> Union[List[Dict[str, Union[str, float, List[str]]]], Tuple[List[str], List[float]]]:
    """
    Asynchronously perform a search based on image and/or text queries.
    
    Args:
        image (Image.Image, optional): Image to search with.
        image_embedding (List[float], optional): Precomputed image embedding.
        phrase (str, optional): Text phrase to search with.
        text_embedding (List[float], optional): Precomputed text embedding.
        topk (int, optional): Number of top results to return.
        group_by_listingId (bool, optional): Whether to group results by listing ID.
        include_all_fields (bool, optional): Whether to include all fields in the results.
        **filters: Additional filters for the search.
    
    Returns:
        Union[List[Dict], Tuple[List[str], List[float]]]: Search results.
    """
    if (not image and not phrase) and (not image_embedding and not text_embedding):
      if group_by_listingId:
        return []
      else:
        return [], []

    listing_info = {}
    combined_results = []
    if image or image_embedding:
      if image_embedding is None:
        image_embedding = self.image_embedder.embed_from_single_image(image).flatten().tolist()

      listings_image_image = await self._search_image_2_image(embedding=image_embedding, topk=topk, 
                                                      group_by_listingId=group_by_listingId, 
                                                      include_all_fields=include_all_fields, 
                                                      **filters)
    
      listings_image_text = await self._search_image_2_text(embedding=image_embedding, topk=topk,
                                                    group_by_listingId=group_by_listingId,
                                                    include_all_fields=include_all_fields,
                                                    **filters)

      if include_all_fields:
        # Strip out fields and store them separately in listing_info
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
        text_embedding = self.text_embedder.embed_from_texts([phrase], batch_size=1)[0].flatten().tolist()

      listings_text_image = await self._search_text_2_image(embedding=text_embedding, topk=topk,
                                                    group_by_listingId=group_by_listingId,
                                                    include_all_fields=include_all_fields,
                                                    **filters)
    
      listings_text_text = await self._search_text_2_text(embedding=text_embedding, topk=topk,
                                                  group_by_listingId=group_by_listingId,
                                                  include_all_fields=include_all_fields,
                                                  **filters)
    
      if include_all_fields:
        # Strip out fields and store them separately in listing_info
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
        # For each result, add the stripped out fields
        for result in combined_results:
          result.update(listing_info[result['listingId']])
        return combined_results
    else:
      if not top_item_names:
        return [], []
      # Sort tuples by score in descending order
      sorted_items = sorted(zip(top_item_names, top_scores), key=lambda x: x[1], reverse=True)
      top_item_names, top_scores = zip(*sorted_items) if sorted_items else ([], [])
      return list(top_item_names), list(top_scores)
    

  @retry(**RETRY_SETTINGS)
  async def multi_image_search(self, 
                                images: List[Image.Image] = None,
                                image_embedding: List[float] = None,
                                phrase: str = None,
                                text_embedding: List[float] = None,
                                topk: int = 5,
                                group_by_listingId: bool = False,
                                include_all_fields: bool = False,
                                **filters
                                ) -> Union[List[Dict[str, Union[str, float, List[str]]]], Tuple[List[str], List[float]]]:
    """
    Asynchronously perform a search using multiple images and an optional phrase.
    
    Args:
        images (List[Image.Image]): List of images to search with.
        phrase (str, optional): Text phrase to search with.
        topk (int, optional): Number of top results to return.
        group_by_listingId (bool, optional): Whether to group results by listing ID.
        include_all_fields (bool, optional): Whether to include all fields in the results.
        **filters: Additional filters for the search.
    
    Returns:
        Union[List[Dict], Tuple[List[str], List[float]]]: Search results.
    """
    # self.logger.info(f'Number of images: {len(images)}')
    if image_embedding is None and images is not None:
      all_image_embeddings = []
      for image in images:
        image_embedding = self.image_embedder.embed_from_single_image(image)
        all_image_embeddings.append(image_embedding)
      mean_vector = np.mean(all_image_embeddings, axis=0)

      image_embedding = mean_vector.flatten().tolist()

    
    if text_embedding is None and phrase is not None:
      text_embedding = self.text_embedder.embed_from_texts([phrase], batch_size=1)[0].flatten().tolist()

    return await self.search(image_embedding=image_embedding, 
                        text_embedding=text_embedding, 
                        topk=topk, 
                        group_by_listingId=group_by_listingId, 
                        include_all_fields=include_all_fields, 
                        **filters)
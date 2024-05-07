from typing import Any, Tuple, List, Dict, Optional, Union
import weaviate
from PIL import Image
from datetime import datetime

import numpy as np

from realestate_vision.common.utils import get_listingId_from_image_name
from ..data.index import FaissIndex

class WeaviateDataStore:
  def __init__(self, 
               client: weaviate.client.Client,
               image_embedder,
               text_embedder,
               score_aggregation_method = 'max'
               ):
    self.client = client
    self.image_embedder = image_embedder
    self.text_embedder = text_embedder
    self.score_aggregation_method = score_aggregation_method

    self.all_properties = [
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
      "invertedIndexConfig": {
        "stopwords": {"additions": [], "preset": "none", "removals": []}
      }
    }

    # Object corresponding to a text embedding of a listing
    listing_text_schema = {
      "class": "Listing_Text",
      "vectorizer": "none",
      "properties": [
        prop if prop['name'] != 'image_name' else {'name': 'remark_chunk_id', 'dataType': ['string']} for prop in listing_image_schema['properties']
      ],
      "vectorIndexType": "hnsw",
      "invertedIndexConfig": listing_image_schema['invertedIndexConfig']
    }

    # Create the schema for both classes
    self.client.schema.create_class(listing_image_schema)
    self.client.schema.create_class(listing_text_schema)

 
  def get_schema(self):
    return self.client.schema.get()

  def delete_all(self):
    self.client.schema.delete_all()
 
  def get(self, listing_id: Optional[str] = None, embedding_type: str = 'I'):
    """
    Retrieve a specific item from Weaviate based on the listing_id and embedding_type.
    If listing_id is None, retrieve all items.
    embedding_type: 'I' for image, 'T' for text.
    """

    class_name = "Listing_Image" if embedding_type == 'I' else "Listing_Text"
    extra_properties = ['image_name'] if embedding_type == 'I' else ['remark_chunk_id']

    # Initialize the query
    query = self.client.query.get(
      class_name=class_name,
      properties=self.all_properties + extra_properties
    ).with_additional(["id"])
    
    # Add the where filter only if a listing_id is provided
    if listing_id is not None:
      query = query.with_where({
        "operator": "Equal",
        "path": ["listing_id"],
        "valueText": listing_id
      })
    
    _results = query.do()

    results = []
    for result in _results['data']['Get'][class_name]:
      result['uuid'] = result['_additional']['id']
      del result['_additional']
      result = self._postprocess_listing_json(result)
      results.append(result)

    return results
    

  def insert(self, listing_json: Dict, embedding_type: str = 'I'):
    '''
    Insert a listing into the Weaviate database
    '''
    class_name = "Listing_Image" if embedding_type == 'I' else "Listing_Text"

    listing_json = self._preprocess_listing_json(listing_json)

    # TODO: for dev only, remove later
    # vector = [0.1, 0.2, 0.3, 0.4, 0.5]  
    # create a random normalized vector of dim = 5
    vector = np.random.randn(5)
    vector = vector / np.linalg.norm(vector)

    uuid = self.client.data_object.create(
        data_object=listing_json,
        class_name=class_name,
        vector=vector
    )

    return uuid
  
  def import_from_faiss_index(self, faiss_index: FaissIndex, listing_df: pd.DataFrame, embedding_type: str = 'I', sample_size: int = None):
    """
    Import data from a FaissIndex object into Weaviate. The listing_df must be 
    there to provide the necessary metadata for each listing.
    """
    _embeddings = faiss_index.index.reconstruct_n(0, faiss_index.index.ntotal)


  def _preprocess_listing_json(self, listing_json: Dict) -> Dict:
    """
    Perform necessary preprocessing on the listing json before storing it in the Weaviate database
    """
    if 'propertyFeatures' in listing_json and isinstance(listing_json['propertyFeatures'], (list, np.ndarray)):
      listing_json['propertyFeatures'] = ', '.join(listing_json['propertyFeatures'])
    
    # format the date correctly for weaviate
    if 'listingDate' in listing_json and isinstance(listing_json['listingDate'], str):
      listing_json['listingDate'] = datetime.strptime(listing_json['listingDate'], '%Y/%m/%d').strftime('%Y-%m-%dT%H:%M:%SZ')
    if 'lastUpdate' in listing_json and isinstance(listing_json['lastUpdate'], str):
      listing_json['lastUpdate'] = datetime.strptime(listing_json['lastUpdate'], '%Y/%m/%d').strftime('%Y-%m-%dT%H:%M:%SZ')
    if 'lastPhotoUpdate' in listing_json and isinstance(listing_json['lastPhotoUpdate'], str):
      listing_json['lastPhotoUpdate'] = datetime.strptime(listing_json['lastPhotoUpdate'], '%Y/%m/%d').strftime('%Y-%m-%dT%H:%M:%SZ')

    if 'lat' in listing_json:
      if np.isnan(listing_json['lat']):
        listing_json['lat'] = None
      elif isinstance(listing_json['lat'], str):
        listing_json['lat'] = float(listing_json['lat'])
    
    if 'lng' in listing_json:
      if np.isnan(listing_json['lng']):
        listing_json['lng'] = None
      elif isinstance(listing_json['lng'], str):
        listing_json['lng'] = float(listing_json['lng'])

    return listing_json
  
  def _postprocess_listing_json(self, listing_json: Dict) -> Dict:
    '''
    Basically reverse of _preprocess_listing_json for relevant fields.
    '''
    if 'propertyFeatures' in listing_json:
      listing_json['propertyFeatures'] = listing_json['propertyFeatures'].split(', ')

    if 'listingDate' in listing_json and isinstance(listing_json['listingDate'], str):
      listing_json['listingDate'] = datetime.strptime(listing_json['listingDate'], '%Y-%m-%dT%H:%M:%SZ').strftime('%Y/%m/%d')
    if 'lastUpdate' in listing_json and isinstance(listing_json['lastUpdate'], str):
      listing_json['lastUpdate'] = datetime.strptime(listing_json['lastUpdate'], '%Y-%m-%dT%H:%M:%SZ').strftime('%Y/%m/%d')
    if 'lastPhotoUpdate' in listing_json and isinstance(listing_json['lastPhotoUpdate'], str):
      listing_json['lastPhotoUpdate'] = datetime.strptime(listing_json['lastPhotoUpdate'], '%Y-%m-%dT%H:%M:%SZ').strftime('%Y/%m/%d')

    return listing_json
      








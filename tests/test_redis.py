import unittest, re
import redis
import numpy as np
from PIL import Image

from realestate_core.common.utils import save_to_pickle, load_from_pickle
from realestate_vss.models.embedding import OpenClipTextEmbeddingModel, OpenClipImageEmbeddingModel
from realestate_vss.data.redis_datastore import RedisDataStore

model_name = 'ViT-L-14'
pretrained='laion2b_s32b_b82k'

REDIS_HOST = 'localhost'
REDIS_PORT = 6380

# python -m unittest test_redis.TestRedisDataStore.test_search_image_2_text

class TestRedisDataStore(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.image_embedder = OpenClipImageEmbeddingModel(model_name=model_name, pretrained=pretrained)
    cls.text_embedder = OpenClipTextEmbeddingModel(embedding_model=cls.image_embedder)

    # Connect to Redis server on port 6380
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
    cls.datastore = RedisDataStore(client=redis_client, image_embedder=cls.image_embedder, text_embedder=cls.text_embedder)

  @classmethod
  def tearDownClass(cls):
    del cls.datastore

  def test_insert_and_get_delete(self):
    listing_json = {
      "listing_id": "1234",
      "image_name": "1234_1.jpg",
      # "embedding": np.random.rand(768).tolist(),
      "embedding": np.random.rand(1, 768),
      "remarks": "Nice house"
    }
    self.datastore.insert(listing_json, embedding_type='I')
    
    result = self.datastore.get(listing_id="1234", obj_id="1234_1.jpg", embedding_type='I')
    self.assertIsNotNone(result)
    self.assertEqual(result['listing_id'], "1234")
    self.assertEqual(result['image_name'], "1234_1.jpg")
    self.assertEqual(result['remarks'], "Nice house")

    self.datastore.delete_listing(listing_id="1234")   # cleanup

    # check to see if its still there
    result = self.datastore.get(listing_id="1234", obj_id="1234_1.jpg", embedding_type='I')
    self.assertIsNone(result)

  def test_batch_insert(self):
    listing_jsons = load_from_pickle('data/sample_image_listing_jsons.pkl')

    self.datastore.batch_insert(listing_jsons, embedding_type='I')

    results = self.datastore.get(listing_id="1234", embedding_type='I')
    self.assertEqual(len(results), 1)

    results = self.datastore.get(listing_id="1235", embedding_type='I')
    self.assertEqual(len(results), 1)

    results = self.datastore.get(listing_id="1236", embedding_type='I')
    self.assertEqual(len(results), 1)

    # cleanup
    self.datastore.delete_listing(listing_id="1234")
    self.datastore.delete_listing(listing_id="1235")
    self.datastore.delete_listing(listing_id="1236")

    # check to see if its still there
    results = self.datastore.get(listing_id="1234", embedding_type='I')
    self.assertEqual(len(results), 0)
    results = self.datastore.get(listing_id="1235", embedding_type='I')
    self.assertEqual(len(results), 0)
    results = self.datastore.get(listing_id="1236", embedding_type='I')
    self.assertEqual(len(results), 0)

  def test_search_image_2_image(self):
    image = Image.open('data/cn_tower.jpg')

    # filters = {'provState': 'ON'}
    filters = {'city': 'Toronto'}
    top_image_names, top_scores = self.datastore._search_image_2_image(image=image, topk=50, group_by_listingId=False, **filters)

    self.assertGreater(len(top_image_names), 1)
    self.assertGreater(len(top_scores), 1)
    self.assertIsInstance(top_scores[0], float)

    listings = self.datastore._search_image_2_image(image=image, topk=50, group_by_listingId=True, include_all_fields=True, **filters)
    '''
    {'listingId': '21373418',
      'agg_score': 0.888317386023,
      'image_names': ['21373418/21373418_7.jpg'],
      'listing_id': '21373418',
      'image_name': '21373418_7.jpg',
      'city': 'Toronto',
      'provState': 'ON',
      'postalCode': 'M5J 2Y5',
      'lat': 43.6400909,
      'lng': -79.3815002,
      'streetName': '#1513 -208 QUEENS QUAY W',
      'beds': '1',
      'bedsInt': 1,
      'baths': '1',
      'bathsInt': 1,
      'sizeInterior': '0',
      'sizeInteriorUOM': 'METERSQ',
      'lotSize': None,
      'lotUOM': None,
      'propertyFeatures': ['ac'],
      'propertyType': 'SINGLE_FAMILY',
      'transactionType': 'LEASE',
      'carriageTrade': False,
      'price': 0.0,
      'leasePrice': 2300.0,
      'pool': False,
      'garage': False,
      'waterFront': False,
      'fireplace': False,
      'ac': True,
      'photo': '//rlp.jumplisting.com/photos/21/37/34/18/21373418_6',
      'listingDate': '2024/02/09',
      'lastPhotoUpdate': '2024/04/12',
      'lastUpdate': '2024/04/12',
      'embeddingType': 'I'}
    '''
    self.assertIsInstance(listings, list)

    # listings[0] has key agg_score, image_names, embeddingType
    self.assertIn('agg_score', listings[0])
    self.assertIn('image_names', listings[0])
    self.assertIn('embeddingType', listings[0])

    self.assertIsInstance(listings[0]['agg_score'], float)
    self.assertIsInstance(listings[0]['image_names'], list)
    # an elem of image_names is a string in this format '1234/1234_1.jpg'
    if len(listings[0]['image_names']) > 0:
      image_name = listings[0]['image_names'][0]
      self.assertIsInstance(image_name, str)

      # Check if the string matches the required format
      pattern = re.compile(r'^\d+/\d+_\d+\.jpg$')
      match = pattern.match(image_name)
      self.assertIsNotNone(match, f"{image_name} doesn't match the required format")

    self.assertEqual(listings[0]['embeddingType'], 'I')

  def test_search_image_2_text(self):
    image = Image.open('data/cn_tower.jpg')

    # filters = {'provState': 'ON'}
    filters = {'city': 'Toronto'}
    top_remark_chunk_ids, top_scores = self.datastore._search_image_2_text(image=image, topk=50, group_by_listingId=False, **filters)

    self.assertGreater(len(top_remark_chunk_ids), 1)
    self.assertGreater(len(top_scores), 1)
    self.assertIsInstance(top_scores[0], float)

    listings = self.datastore._search_image_2_text(image=image, topk=50, group_by_listingId=True, include_all_fields=True, **filters)
    '''
    {'listingId': '21810309',
    'agg_score': 0.33671508907799996,
    'remark_chunk_ids': ['21810309_14'],
    'image_names': [],
    'listing_id': '21810309',
    'remark_chunk_id': '21810309_14',
    'city': 'Toronto',
    'provState': 'ON',
    'postalCode': 'M5V 3S2',
    'lat': 43.6414261,
    'lng': -79.3911972,
    'streetName': '#1208 -81 NAVY WHARF CRT',
    'beds': '2+1',
    'bedsInt': 3,
    'baths': '2',
    'bathsInt': 2,
    'sizeInterior': '0',
    'sizeInteriorUOM': 'METERSQ',
    'lotSize': None,
    'lotUOM': None,
    'propertyFeatures': ['fireplace', 'ac', 'parking'],
    'propertyType': 'SINGLE_FAMILY',
    'transactionType': 'SALE',
    'carriageTrade': False,
    'price': 999999.0,
    'leasePrice': 0.0,
    'pool': False,
    'garage': False,
    'waterFront': False,
    'fireplace': True,
    'ac': True,
    'photo': '//rlp.jumplisting.com/photos/21/81/3/9/21810309_0',
    'listingDate': '2024/04/16',
    'lastPhotoUpdate': '2024/04/16',
    'lastUpdate': '2024/04/16',
    'embeddingType': 'T'}
    '''

    self.assertIsInstance(listings, list)

    # listings[0] has key agg_score, remark_chunk_ids, embeddingType
    self.assertIn('agg_score', listings[0])
    self.assertIn('remark_chunk_ids', listings[0])
    self.assertIn('embeddingType', listings[0])

    self.assertIsInstance(listings[0]['agg_score'], float)
    self.assertIsInstance(listings[0]['remark_chunk_ids'], list)
    # an elem of remark_chunk_ids is a string in this format '21810309_14'
    if len(listings[0]['remark_chunk_ids']) > 0:
      remark_chunk_id = listings[0]['remark_chunk_ids'][0]
      self.assertIsInstance(remark_chunk_id, str)

      # Check if the string matches the required format
      pattern = re.compile(r'^\d+_\d+$')
      match = pattern.match(remark_chunk_id)
      self.assertIsNotNone(match, f"{remark_chunk_id} doesn't match the required format")

    self.assertEqual(listings[0]['embeddingType'], 'T')
    
  def test_search_text_2_image(self):
    phrase = 'cn tower'

    filters = {'city': 'Toronto'}
    top_image_names, top_scores = self.datastore._search_text_2_image(phrase=phrase, topk=50, group_by_listingId=False, **filters)

    self.assertGreater(len(top_image_names), 1)
    self.assertGreater(len(top_scores), 1)
    self.assertIsInstance(top_scores[0], float)

    listings = self.datastore._search_text_2_image(phrase=phrase, topk=50, group_by_listingId=True, include_all_fields=True, **filters)
    '''
    {'listingId': '21776902',
    'agg_score': 0.31757729147900005,
    'image_names': ['21776902/21776902_32.jpg', '21776902/21776902_33.jpg'],
    'listing_id': '21776902',
    'image_name': '21776902_33.jpg',
    'city': 'Toronto',
    'provState': 'ON',
    'postalCode': 'M4K 1V7',
    'lat': 43.6817169,
    'lng': -79.3551102,
    'streetName': '#UPPER -42 BROWNING AVE',
    'beds': '4',
    'bedsInt': 4,
    'baths': '2',
    'bathsInt': 2,
    'sizeInterior': '0',
    'sizeInteriorUOM': 'METERSQ',
    'lotSize': None,
    'lotUOM': None,
    'propertyFeatures': ['fireplace', 'ac'],
    'propertyType': 'SINGLE_FAMILY',
    'transactionType': 'LEASE',
    'carriageTrade': False,
    'price': 0.0,
    'leasePrice': 4950.0,
    'pool': False,
    'garage': False,
    'waterFront': False,
    'fireplace': True,
    'ac': True,
    'photo': '//rlp.jumplisting.com/photos/21/77/69/2/21776902_17',
    'listingDate': '2024/04/11',
    'lastPhotoUpdate': '2024/04/13',
    'lastUpdate': '2024/04/13',
    'embeddingType': 'I'}
    '''

    self.assertIsInstance(listings, list)

    # listings[0] has key agg_score, image_names, embeddingType
    self.assertIn('agg_score', listings[0])
    self.assertIn('image_names', listings[0])
    self.assertIn('embeddingType', listings[0])

    self.assertIsInstance(listings[0]['agg_score'], float)
    self.assertIsInstance(listings[0]['image_names'], list)
    # an elem of image_names is a string in this format '1234/1234_1.jpg'
    if len(listings[0]['image_names']) > 0:
      image_name = listings[0]['image_names'][0]
      self.assertIsInstance(image_name, str)

      # Check if the string matches the required format
      pattern = re.compile(r'^\d+/\d+_\d+\.jpg$')
      match = pattern.match(image_name)
      self.assertIsNotNone(match, f"{image_name} doesn't match the required format")

    self.assertEqual(listings[0]['embeddingType'], 'I')

  def test_search_text_2_text(self):
    phrase = 'cn tower'
    filters = {'city': 'Toronto'}
    top_remark_chunk_ids, top_scores = self.datastore._search_text_2_text(phrase=phrase, topk=50, group_by_listingId=False, **filters)

    self.assertGreater(len(top_remark_chunk_ids), 1)
    self.assertGreater(len(top_scores), 1)
    self.assertIsInstance(top_scores[0], float)

    listings = self.datastore._search_text_2_text(phrase=phrase, topk=50, group_by_listingId=True, include_all_fields=True, **filters)
    '''
    {'listingId': '21546476',
      'agg_score': 0.8318567456479999,
      'remark_chunk_ids': ['21546476_2'],
      'image_names': [],
      'listing_id': '21546476',
      'remark_chunk_id': '21546476_2',
      'city': 'Toronto',
      'provState': 'ON',
      'postalCode': 'M1M 2B3',
      'lat': 43.7333603,
      'lng': -79.2296143,
      'streetName': '200 OAKRIDGE DR',
      'beds': '5+3',
      'bedsInt': 8,
      'baths': '7',
      'bathsInt': 7,
      'sizeInterior': '0',
      'sizeInteriorUOM': 'METERSQ',
      'lotSize': '52 x 150 FT ; Premium 52 Ft Lot Frontage! No Sidewalk!',
      'lotUOM': None,
      'propertyFeatures': ['ac', 'parking', 'garage'],
      'propertyType': 'SINGLE_FAMILY',
      'transactionType': 'SALE',
      'carriageTrade': False,
      'price': 2650000.0,
      'leasePrice': 0.0,
      'pool': False,
      'garage': True,
      'waterFront': False,
      'fireplace': False,
      'ac': True,
      'photo': '//rlp.jumplisting.com/photos/21/54/64/76/21546476_39',
      'listingDate': None,
      'lastPhotoUpdate': None,
      'lastUpdate': None,
      'embeddingType': 'T'}
    '''

    self.assertIsInstance(listings, list)

    # listings[0] has key agg_score, remark_chunk_ids, embeddingType
    self.assertIn('agg_score', listings[0])
    self.assertIn('remark_chunk_ids', listings[0])
    self.assertIn('embeddingType', listings[0])

    self.assertIsInstance(listings[0]['agg_score'], float)
    self.assertIsInstance(listings[0]['remark_chunk_ids'], list)
    # an elem of remark_chunk_ids is a string in this format '21546476_2'
    if len(listings[0]['remark_chunk_ids']) > 0:
      remark_chunk_id = listings[0]['remark_chunk_ids'][0]
      self.assertIsInstance(remark_chunk_id, str)

      # Check if the string matches the required format
      pattern = re.compile(r'^\d+_\d+$')
      match = pattern.match(remark_chunk_id)
      self.assertIsNotNone(match, f"{remark_chunk_id} doesn't match the required format")
  
    self.assertEqual(listings[0]['embeddingType'], 'T')

  def test_search(self):
    image = Image.open('data/cn_tower.jpg')
    phrase = 'cn tower'
    filters = {'city': 'Toronto'}

    top_item_names, top_scores = self.datastore.search(image=image, phrase=phrase, topk=50, group_by_listingId=False, **filters)
    self.assertGreater(len(top_item_names), 1)
    self.assertGreater(len(top_scores), 1)
    self.assertIsInstance(top_scores[0], float)

    listings = self.datastore.search(image=image, phrase=phrase, topk=50, group_by_listingId=True, include_all_fields=True, **filters)
    '''
    {'listingId': '21373418',
      'agg_score': 1.0,
      'image_names': ['21373418/21373418_7.jpg'],
      'listing_id': '21373418',
      'city': 'Toronto',
      'provState': 'ON',
      'postalCode': 'M5J 2Y5',
      'lat': 43.6400909,
      'lng': -79.3815002,
      'streetName': '#1513 -208 QUEENS QUAY W',
      'beds': '1',
      'bedsInt': 1,
      'baths': '1',
      'bathsInt': 1,
      'sizeInterior': '0',
      'sizeInteriorUOM': 'METERSQ',
      'lotSize': None,
      'lotUOM': None,
      'propertyFeatures': 'ac',
      'propertyType': 'SINGLE_FAMILY',
      'transactionType': 'LEASE',
      'carriageTrade': 0,
      'price': 0.0,
      'leasePrice': 2300.0,
      'pool': 0,
      'garage': 0,
      'waterFront': 0,
      'fireplace': 0,
      'ac': 1,
      'photo': '//rlp.jumplisting.com/photos/21/37/34/18/21373418_6',
      'listingDate': 1707436800,
      'lastPhotoUpdate': 1715011163,
      'lastUpdate': 1715011163}
    '''

    self.assertIsInstance(listings, list)

    # listings[0] has key agg_score, item_names
    self.assertIn('agg_score', listings[0])
    self.assertIn('image_names', listings[0])

    self.assertIsInstance(listings[0]['agg_score'], float)
    self.assertIsInstance(listings[0]['image_names'], list)

  def test_multi_image_search(self):
    images = [
      Image.open('data/cn_tower.jpg'),
      Image.open('data/ugly_kitchen.jpg')
    ]
    phrase = 'cn tower'
    filters = {'city': 'Toronto'}

    top_item_names, top_scores = self.datastore.multi_image_search(images=images, 
                                                                    phrase=phrase, 
                                                                    topk=50, 
                                                                    group_by_listingId=False, 
                                                                    **filters)
    
    self.assertGreater(len(top_item_names), 1)
    self.assertGreater(len(top_scores), 1)
    self.assertIsInstance(top_scores[0], float)

    listings = self.datastore.multi_image_search(images=images,
                                                 phrase=phrase,
                                                 topk=50,
                                                 group_by_listingId=True,
                                                 include_all_fields=True,
                                                 **filters
                                                 )
    '''
    {'listingId': '21850113',
      'agg_score': 1.0,
      'image_names': [],
      'remark_chunk_ids': ['21850113_5'],
      'listing_id': '21850113',
      'remark_chunk_id': '21850113_5',
      'city': 'Toronto',
      'provState': 'ON',
      'postalCode': 'M5J 0C3',
      'lat': 43.6418152,
      'lng': -79.3796768,
      'streetName': '#4304 -88 HARBOUR ST',
      'beds': '2',
      'bedsInt': 2,
      'baths': '2',
      'bathsInt': 2,
      'sizeInterior': '0',
      'sizeInteriorUOM': 'METERSQ',
      'lotSize': None,
      'lotUOM': None,
      'propertyFeatures': 'ac',
      'propertyType': 'SINGLE_FAMILY',
      'transactionType': 'SALE',
      'carriageTrade': 0,
      'price': 928000.0,
      'leasePrice': 0.0,
      'pool': 0,
      'garage': 0,
      'waterFront': 0,
      'fireplace': 0,
      'ac': 1,
      'photo': '//rlp.jumplisting.com/photos/21/85/1/13/21850113_0',
      'listingDate': 1713484800,
      'lastPhotoUpdate': 1713603330,
      'lastUpdate': 1713619398}
    '''

    self.assertIsInstance(listings, list)
    # listings[0] has key agg_score, item_names
    self.assertIn('agg_score', listings[0])
    self.assertIn('image_names', listings[0])

    self.assertIsInstance(listings[0]['agg_score'], float)
    self.assertIsInstance(listings[0]['image_names'], list)
    



  def test_get_listing_and_image_names(self):
    ''' Sample 
    {
      "jumpId": "21526459",
      "city": "Toronto",
      "provState": "ON",
      "postalCode": "M3C 2E9",
      "lat": 43.7327614,
      "lng": -79.3456573,
      "streetName": "#1208 -75 THE DONWAY  W",
      "beds": "1",
      "bedsInt": 1,
      "baths": "1",
      "bathsInt": 1,
      "sizeInterior": "0",
      "sizeInteriorUOM": "METERSQ",
      "lotSize": null,
      "lotUOM": null,
      "propertyFeatures": [
        "ac",
        "parking"
      ],
      "propertyType": "SINGLE_FAMILY",
      "transactionType": "LEASE",
      "carriageTrade": false,
      "price": 0,
      "leasePrice": 2400,
      "pool": false,
      "garage": false,
      "waterFront": false,
      "fireplace": false,
      "ac": true,
      "remarks": "Wow - Located In The Shops Of Don Mills & W/ An Unobstructed View Of A Green Landscape + The Downtown Core (Cn Tower) This Unit Is Perfect From Inside & Out. The Freshly Painted & Sun Filled Unit Has An Open Concept Layout, Flr To Ceiling Windows, Soaring 10 Ft Ceilings & A Oversized Wide Balcony. The Modern White Kitchen Includes Backsplash, Stone Countertop & Stainless Steel Appliances. Unit Includes Parking, Locker, Amazing Amenities And More... **** EXTRAS **** 1 Parking Spot And 1 Locker. Balcony With Stunning & Unobstructed View. Great Amenities: Concierge, Gym, Rooftop Patio W/ Hot Tub & Bbq + Shops Of Don Mills. (id:27)",
      "listing_id": "21526459",
      "photo": "//rlp.jumplisting.com/photos/21/52/64/59/21526459_24",
      "listingDate": "2024/03/06"
    }
    '''

    listing_id = '21618852'   # TODO, should dynamically figure the listing_id such that it should exist

    listing_json = self.datastore.get_listing(listing_id)  
    self.assertIsNotNone(listing_json)

    self.assertEqual(listing_json['jumpId'], listing_id)
    self.assertEqual(listing_json['city'], 'Toronto')

    # photo, remark exist
    self.assertIn('photo', listing_json)
    self.assertIn('remarks', listing_json)

    image_names = self.datastore.get_imagenames(listing_id)

    self.assertIsInstance(image_names, list)
    self.assertGreater(len(image_names), 0)
    # check image format
    for image_name in image_names:
      pattern = re.compile(r'^\d+_\d+\.jpg$')
      match = pattern.match(image_name)
      self.assertIsNotNone(match, f"{image_name} doesn't match the required format")

  @unittest.skip("Skipping this test as it takes a long time to run")
  def test_delete_all(self):
    self.datastore.delete_index()
    self.datastore.delete_all_listings()

    # check to see if its still there
    listing_json = self.datastore.get_listing('21526459')
    self.assertIsNone(listing_json)

    image_names = self.datastore.get_imagenames('21526459')
    self.assertEqual(len(image_names), 0)
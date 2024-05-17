import unittest, re

import weaviate
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path


from realestate_core.common.utils import save_to_pickle, load_from_pickle
from realestate_vss.models.embedding import OpenClipTextEmbeddingModel, OpenClipImageEmbeddingModel
from realestate_vss.data.index import FaissIndex
from realestate_vss.data.weaviate_datastore import WeaviateDataStore_v4 as WeaviateDataStore

model_name = 'ViT-L-14'
pretrained='laion2b_s32b_b82k'

WEAVIATE_HOST = 'localhost'
WEAVIATE_PORT = 8080

# python -m unittest test_weaviate.TestWeaviateDataStore.test_search_image_2_image

class TestWeaviateDataStore(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.image_embedder = OpenClipImageEmbeddingModel(model_name=model_name, pretrained=pretrained)
    cls.text_embedder = OpenClipTextEmbeddingModel(embedding_model=cls.image_embedder)

    client = weaviate.connect_to_local(WEAVIATE_HOST, WEAVIATE_PORT)
    cls.datastore = WeaviateDataStore(client=client, 
                                      image_embedder=cls.image_embedder,
                                      text_embedder=cls.text_embedder,
                                      run_create_collection=True
                                    )
    
    # ensure there's no count
    total_count = cls.datastore.count_all()

    # if total_count != 0:
    #   raise Exception("Weaviate datastore is not empty, ensure the weaviate instance is for testing.")
    
    # # load data from faiss index (so actually testing this)
    # faiss_image_index = FaissIndex(filepath=Path("./data/faiss_image_index"))
    # faiss_text_index = FaissIndex(filepath=Path("./data/faiss_text_index"))
    # listing_df = pd.read_feather('./data/listing_df')

    # cls.datastore.import_from_faiss_index(faiss_index=faiss_image_index, 
    #                                        listing_df=listing_df, 
    #                                        embedding_type='I'
    #                                        )
    
    # cls.datastore.import_from_faiss_index(faiss_index=faiss_text_index,
    #                                        listing_df=listing_df,
    #                                        embedding_type='T'
    #                                        )
    
    # # confirm count
    # total_count = cls.datastore.count_all()
    # if total_count == 0:
    #   raise Exception("Weaviate datastore is empty, debug before proceeding to testings.")

  
  def test_insert_and_get_delete(self):
    listing_json = {
      "listing_id": "1234",
      "image_name": "1234_1.jpg",
      # "embedding": np.random.rand(768).tolist(),
      "embedding": np.random.rand(1, 768),
      "remarks": "Nice house"
    }
    self.datastore.insert(listing_json, embedding_type='I')
    
    result = self.datastore.get(listing_id="1234", embedding_type='I')[0]
    self.assertIsNotNone(result)
    self.assertEqual(result['listing_id'], "1234")
    self.assertEqual(result['image_name'], "1234_1.jpg")
    self.assertEqual(result['remarks'], "Nice house")

    self.datastore.delete_listing(listing_id="1234")   # cleanup

    # check to see if its still there
    result = self.datastore.get(listing_id="1234", embedding_type='I')
    self.assertEqual(result, [])

  def test_batch_insert(self):
    # already tested during setUpClass via import_from_faiss_index
    pass

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
    {'listingId': '21540195',
      'agg_score': 0.6006518602371216,
      'image_names': ['21540195/21540195_39.jpg',
        '21540195/21540195_38.jpg',
        '21540195/21540195_44.jpg',
        '21540195/21540195_68.jpg',
        '21540195/21540195_50.jpg',
        '21540195/21540195_57.jpg',
        '21540195/21540195_40.jpg',
        '21540195/21540195_58.jpg'],
      'remarks': "1,986 sf of Luxury Resort style living with 270 degree waterfront views!  Welcome to Suite 2006 at 1 Palace Pier, bright and spacious Suite with 2 Bedrooms,  3 Baths, 2 Solariums with Unbeatable Views and World Class Amenities. Owner-upgraded with Smooth Ceilings, Modern Cabinetry, refreshed Dark Wood flooring, Contemporary Chef's Kitchen, Japanese Smart Bidet System, and Much More! Enjoy life with all possible amenities including Complimentary Valet Parking, Private Shuttle Service, and Exclusive World Class Les Clefs D'or Concierge Staff - the same service as the Four Seasons Hotel. Standard Amenities including Indoor Pool, Tennis Courts, Gymnasium, and Guest suites are also present. Palace Place is Toronto's Most Luxurious Waterfront Condominium Residence with Unbeatable Panoramic Southern Views of Lake Ontario, City of Toronto Skyline, and Humber Bay Shores.**** EXTRAS **** Newly Renovated hallways, 3 TOTO Smart Bidet Systems, Parking in Building.  06 Units Have The Best Sun Exposure Available in This Building. Suite can be sold turnkey with all furnitures. (id:27)",
      'fireplace': False,
      'lastUpdate': None,
      'streetName': '#2006 -1 PALACE PIER CRT',
      'city': 'Toronto',
      'postalCode': 'M8V 3W9',
      'propertyFeatures': ['ac', 'pool', 'parking'],
      'bathsInt': 3.0,
      'provState': 'ON',
      'pool': True,
      'carriageTrade': False,
      'sizeInterior': '0',
      'transactionType': 'SALE',
      'lat': 43.6314812,
      'listingDate': None,
      'beds': '2',
      'listing_id': '21540195',
      'ac': True,
      'waterFront': False,
      'price': 1899000.0,
      'lotUOM': None,
      'propertyType': 'SINGLE_FAMILY',
      'lastPhotoUpdate': None,
      'baths': '3',
      'garage': False,
      'bedsInt': 2.0,
      'image_name': '21540195_58.jpg',
      'lotSize': None,
      'photo': '//rlp.jumplisting.com/photos/21/54/1/95/21540195_37',
      'lng': -79.472641,
      'sizeInteriorUOM': 'METERSQ',
      'leasePrice': 0.0}
    '''
    self.assertIsInstance(listings, list)

    # listings[0] has key agg_score, image_names
    self.assertIn('agg_score', listings[0])
    self.assertIn('image_names', listings[0])
    # self.assertIn('embeddingType', listings[0])

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
    {'listingId': '21580369',
      'agg_score': 0.29875802993774414,
      'remark_chunk_ids': ['21580369_6', '21580369_5', '21580369_0'],
      'image_names': [],
      'remarks': "A beautiful 1 Bed Suite In Chic Cinema Towers At Tiff Lounge Surrounded By Art & Culture!. 9 Ft Ceilings. Floor-To-Ceiling Windows. Private Balcony. Upscale Building W/ Excellent Amenities Including Full Basketball Court & 35 Seat Screening Room. Located In The Heart Of Toronto's World Class Entertainment District. Steps From Fine Dining, Theatres, Ttc, Lake Ont, Cn Tower, Rogers Centre & Shops!",
      'fireplace': False,
      'lastUpdate': None,
      'streetName': '21 Widmer St',
      'propertyFeatures': ['ac', 'garage'],
      'postalCode': 'M5V 2E8',
      'city': 'Toronto',
      'bathsInt': 1.0,
      'provState': 'ON',
      'pool': False,
      'carriageTrade': False,
      'sizeInterior': None,
      'transactionType': 'LEASE',
      'lat': 43.64718287399732,
      'listingDate': None,
      'beds': '1',
      'listing_id': '21580369',
      'remark_chunk_id': '21580369_0',
      'ac': True,
      'waterFront': False,
      'price': 0.0,
      'lotUOM': None,
      'propertyType': None,
      'lastPhotoUpdate': None,
      'baths': '1',
      'garage': True,
      'bedsInt': 1.0,
      'lotSize': None,
      'photo': '//rlp.jumplisting.com/photos/21/58/3/69/21580369_1',
      'lng': -79.39137717511123,
      'sizeInteriorUOM': None,
      'leasePrice': 2500.0}
    '''

    self.assertIsInstance(listings, list)

    # listings[0] has key agg_score, remark_chunk_ids
    self.assertIn('agg_score', listings[0])
    self.assertIn('remark_chunk_ids', listings[0])


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

  def test_search_text_2_image(self):
    phrase = 'cn tower'

    filters = {'city': 'Toronto'}
    top_image_names, top_scores = self.datastore._search_text_2_image(phrase=phrase, topk=50, group_by_listingId=False, **filters)

    self.assertGreater(len(top_image_names), 1)
    self.assertGreater(len(top_scores), 1)
    self.assertIsInstance(top_scores[0], float)

    listings = self.datastore._search_text_2_image(phrase=phrase, topk=50, group_by_listingId=True, include_all_fields=True, **filters)
    '''
    {'listingId': '21542578',
      'agg_score': 0.28556346893310547,
      'image_names': ['21542578/21542578_98.jpg',
        '21542578/21542578_99.jpg',
        '21542578/21542578_97.jpg',
        '21542578/21542578_96.jpg'],
      'remarks': 'Welcome to this distinguished & timeless masterpiece nestled in the heart of the coveted Sugarloaf neighbourhood rich with executive century homes. 28 Catharine Street offers a rare opportunity to own a piece of Port Colborne’s heritage while enjoying contemporary luxury living. Boasting a classical Georgian style & 4 finished levels, this 4-5 bedroom, 4.5 bath home stands majestically on an elevated lot and is just around the corner from Sugarloaf Marina on Lake Erie and the historic charm of West Street along the Welland Canal. A grand entrance hall greets you at the front entrance, showcasing the classic centre hall plan that defines the architectural allure of this time. Impeccable craftsmanship and attention to detail are evident throughout, with gleaming wood floors, intricate trim, crown molding, a butler’s pantry and 2nd staircase to the kitchen. The main floor shines with a gracious living room adorned with a cozy fireplace, a formal dining room & powder room. The modernized kitchen and adjacent breakfast nook provide a seamless fusion of style and convenience, while a sunroom bathed in natural light offers a quiet retreat & opens to a flagstone patio and gazebo. The elegant main staircase leads to the 2nd floor revealing 4 bedrooms and two bathrooms, including a luxurious primary ensuite with heated floors. Ascend further to the third floor, where a fifth bedroom or study awaits, complete with its own private four-piece bathroom, offering versatility and privacy. The finished basement presents an inviting entertainment space, illuminated by pot lights, and offering ample room for guests or the possibility of a future family suite. Outside, a paved driveway provides ample parking and leads to a double garage. Thoughtfully updated over the past seven years, this home boasts numerous improvements, including many new windows, updated bathrooms, heating, wiring, plumbing, and more, ensuring peace of mind and modern comfort for years to come.',
      'fireplace': True,
      'lastUpdate': None,
      'streetName': 'Catharine',
      'propertyFeatures': ['fireplace', 'parking', 'garage'],
      'postalCode': 'L3K 4J7',
      'city': 'Port Colborne',
      'bathsInt': 5.0,
      'provState': 'ON',
      'pool': False,
      'carriageTrade': False,
      'sizeInterior': '3238',
      'transactionType': 'SALE',
      'lat': 42.880524,
      'listingDate': None,
      'listing_id': '21542578',
      'beds': '5',
      'ac': False,
      'waterFront': False,
      'price': 1095000.0,
      'lotUOM': None,
      'propertyType': 'SINGLE_FAMILY',
      'lastPhotoUpdate': None,
      'baths': '5',
      'garage': True,
      'bedsInt': 5.0,
      'image_name': '21542578_96.jpg',
      'lotSize': None,
      'lng': -79.252532,
      'photo': '//rlp.jumplisting.com/photos/feed/28/114/4/52/2774/21542578_50',
      'sizeInteriorUOM': 'FEETSQ',
      'leasePrice': 0.0}
    '''

    self.assertIsInstance(listings, list)

    # listings[0] has key agg_score, image_names
    self.assertIn('agg_score', listings[0])
    self.assertIn('image_names', listings[0])

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

  def test_search_text_2_text(self):
    phrase = 'cn tower'
    filters = {'city': 'Toronto'}
    top_remark_chunk_ids, top_scores = self.datastore._search_text_2_text(phrase=phrase, topk=50, group_by_listingId=False, **filters)

    self.assertGreater(len(top_remark_chunk_ids), 1)
    self.assertGreater(len(top_scores), 1)
    self.assertIsInstance(top_scores[0], float)

    listings = self.datastore._search_text_2_text(phrase=phrase, topk=50, group_by_listingId=True, include_all_fields=True, **filters)
    '''
    {'listingId': '21576838',
      'agg_score': 0.7048524618148804,
      'remark_chunk_ids': ['21576838_12'],
      'image_names': [],
      'remarks': 'BUILD YOUR DREAM HOME ON THE HILL-TOP IN HIGHLY SOUGHT AFTER VISTA-VILLAGE NEIGHBORHOOD IN STREETSVILLE, MISSISSAUGA. Existing: 3 Bedroom Open Concept Bungalow With Large Living Area, Flagstone Fireplace, Upgraded Kitchen With S/S Appliances, pot-lights & Skylight. Walkout To Back Yard. Fully Insulated Garage. Beautifully Landscaped Backyard With Stone Patio, Shed, Fire Pit & Gas Outlet for BBQ. Desirable Street With Mature Trees. Centrally Located To All Amenities. Close to Erin Mills Town Centre and Streetsville Downtown. Proposed: Building Permit in process (Minor Variance Committee - was approved on April 27, 2023). Building permit to build a beautiful over 3000 SF 2 story home is expected anytime. Check out the rendering and floor plans of this proposed mansion. Plans are available upon request. The Building Permit is expected anytime.**** EXTRAS **** S/S Fridge, Gas Stove, Dishwasher. Washer & Dryer (As Is) Water Softner. Potlights Throughout. Easy showing with L/B, vacant property. Leave your business card. (id:27)',
      'fireplace': True,
      'lastUpdate': None,
      'streetName': '4 WAREHAM DR',
      'city': 'Mississauga',
      'bathsInt': 1.0,
      'postalCode': 'L5M 1B6',
      'propertyFeatures': ['fireplace', 'ac', 'parking', 'garage'],
      'provState': 'ON',
      'pool': False,
      'carriageTrade': False,
      'sizeInterior': '0',
      'transactionType': 'SALE',
      'lat': 43.5737801,
      'listingDate': None,
      'remark_chunk_id': '21576838_12',
      'beds': '3',
      'listing_id': '21576838',
      'ac': True,
      'waterFront': False,
      'price': 999999.0,
      'lotUOM': None,
      'propertyType': 'SINGLE_FAMILY',
      'lastPhotoUpdate': None,
      'baths': '1',
      'garage': True,
      'bedsInt': 3.0,
      'lotSize': '80 x 85.5 FT',
      'lng': -79.7227783,
      'photo': '//rlp.jumplisting.com/photos/21/57/68/38/21576838_1',
      'sizeInteriorUOM': 'METERSQ',
      'leasePrice': 0.0}
    '''

    self.assertIsInstance(listings, list)

    # listings[0] has key agg_score, remark_chunk_ids
    self.assertIn('agg_score', listings[0])

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

    
  @classmethod
  def tearDownClass(cls):
    # cls.datastore.delete_all()
    del cls.datastore
    

    
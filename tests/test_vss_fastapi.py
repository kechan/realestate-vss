import unittest
import requests
import json

class TestVSSFastAPI(unittest.TestCase):
  BASE_URL = 'http://localhost:8000'
  

  def test_listing(self):
    listingId = '20755797'
    response = requests.get(f'{self.BASE_URL}/listing/{listingId}')
    self.assertEqual(response.status_code, 200)

    data = response.json()
    self.assertIsInstance(data, dict)

    self.assertEqual(data['jumpId'], listingId)


  def test_search_by_image(self):
    with open('./data/niagara_fall.jpeg', 'rb') as f:
      response = requests.post(f'{self.BASE_URL}/search-by-image/', files={'file': f})
    self.assertEqual(response.status_code, 200)

    # Parse the JSON response
    data = response.json()
    
    self.assertIsInstance(data, list)
    self.assertGreater(len(data), 0)

    # check keys 
    self.assertIn('listingId', data[0])
    self.assertIn('avg_score', data[0])
    self.assertIn('image_names', data[0])

    # check image_names is a list of jpg filenames
    self.assertIsInstance(data[0]['image_names'], list)
    self.assertGreater(len(data[0]['image_names']), 0)
    self.assertTrue(all([name.endswith('.jpg') for name in data[0]['image_names']]))


  def test_images(self):
    listingId = '21075684'
    response = requests.get(f'{self.BASE_URL}/images/{listingId}/{listingId}_1.jpg')
    self.assertEqual(response.status_code, 200)

    self.assertEqual(response.headers['Content-Type'], 'image/jpeg')
    self.assertGreater(int(response.headers['content-length']), 0)

  def test_search_by_text(self):
    search_mode = 'SOFT_MATCH_AND_VSS'
    lambda_val = 0.8
    alpha_val = 0.5
    data = {
      'phrase': 'big kitchen',
      'provState': 'MB'
    }
    response = requests.post(f'{self.BASE_URL}/search-by-text/?mode={search_mode}&lambda_val={lambda_val}&alpha_val={alpha_val}', data=json.dumps(data))
    self.assertEqual(response.status_code, 200)

    data = response.json()
    
    self.assertIsInstance(data, list)
    self.assertGreater(len(data), 0)

    # check keys
    self.assertIn('listing_id', data[0])
    self.assertIn('score', data[0])
    self.assertIn('provState', data[0])
    self.assertEqual(data[0]['provState'], 'MB')

  def test_text_to_image_search(self):
    data = {
      'phrase': 'chandelier'
    }
    response = requests.post(f'{self.BASE_URL}/text-to-image-search/', data=json.dumps(data))
    self.assertEqual(response.status_code, 200)

if __name__ == '__main__':
  unittest.main()
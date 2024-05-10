import unittest
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
import os

from PIL import Image

import torch

import realestate_core.common.class_extensions
from realestate_vss.data.index import FaissIndex
from realestate_vss.models.embedding import OpenClipTextEmbeddingModel, OpenClipImageEmbeddingModel

# python -m unittest test_index.py

_ = load_dotenv(find_dotenv())


                              
class TestFaissIndex(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.faiss_image_index = FaissIndex(filepath=Path("./data/faiss_image_index"))
    cls.faiss_text_index = FaissIndex(filepath=Path("./data/faiss_text_index"))
    cls.listing_df = pd.read_feather('./data/listing_df')
    cls.embedding_dim_image = cls.faiss_image_index.index.d
    cls.embedding_dim_text = cls.faiss_text_index.index.d

    model_name = 'ViT-L-14'
    pretrained='laion2b_s32b_b82k'

    device = torch.device('cpu')

    cls.image_embedder = OpenClipImageEmbeddingModel(model_name=model_name, pretrained=pretrained, device=device)
    cls.text_embedder = OpenClipTextEmbeddingModel(embedding_model=cls.image_embedder)

  def test_image_index_loaded(self):
    self.assertGreater(self.faiss_image_index.index.ntotal, 0)
    self.assertIsInstance(self.faiss_image_index.aux_info, pd.DataFrame)

  def test_text_index_loaded(self):
    self.assertGreater(self.faiss_text_index.index.ntotal, 0)
    self.assertIsInstance(self.faiss_text_index.aux_info, pd.DataFrame)

  def test_listing_df_loaded(self):
    self.assertIsInstance(self.listing_df, pd.DataFrame)
    self.assertGreater(len(self.listing_df), 0)

  def test_add(self):
    initial_count = self.faiss_image_index.index.ntotal
    initial_embeddings = self.faiss_image_index.get_embeddings().copy()
    initial_aux_info = self.faiss_image_index.aux_info.copy()

    # select a few random entries to monitor
    sample_indices = np.random.choice(initial_count, 3, replace=False)
    sample_embeddings_before = initial_embeddings[sample_indices]
    sample_image_names_before = initial_aux_info.iloc[sample_indices]['image_name'].values 

    # new stuff to add
    new_embeddings = np.random.rand(3, self.embedding_dim_image).astype(np.float32)
    new_aux_info = pd.DataFrame({
      "listing_id": ["a", "b", "c"],
      "image_name": ["a_4.jpg", "b_1.jpg", "c_2.jpg"]
    })

    # perform add op
    self.faiss_image_index.add(new_embeddings, new_aux_info)

    # check count
    self.assertEqual(self.faiss_image_index.index.ntotal, initial_count + 3)

    # Fetch the updated embeddings and aux_info
    updated_embeddings = self.faiss_image_index.get_embeddings()
    updated_aux_info = self.faiss_image_index.aux_info

    # Ensure the embeddings for the monitored entries are unchanged
    for i, image_name in enumerate(sample_image_names_before):
      # Find the index of the image_name after the addition
      new_index = updated_aux_info[updated_aux_info['image_name'] == image_name].index[0]
      # Fetch the embedding from the new index
      embedding_after = updated_embeddings[new_index]
      # Assert that the embedding has not changed
      np.testing.assert_array_almost_equal(embedding_after, sample_embeddings_before[i], decimal=6)


  def test_remove_method(self):
    # Identify the listing_id to remove and count its occurrences
    listing_id_to_remove = "21563838"
    count_to_remove = len(self.faiss_image_index.aux_info.q("listing_id == @listing_id_to_remove"))
    initial_count = self.faiss_image_index.index.ntotal
    initial_embeddings = self.faiss_image_index.get_embeddings().copy()
    initial_aux_info = self.faiss_image_index.aux_info.copy()

    # Select a few random entries to monitor
    sample_indices = np.random.choice(initial_count, 3, replace=False)
    sample_embeddings_before = initial_embeddings[sample_indices]
    sample_image_names_before = initial_aux_info.iloc[sample_indices]['image_name'].values

    # Remove entries with this listing_id
    self.faiss_image_index.remove([listing_id_to_remove])

    # Fetch the current count and check for the presence of removed items
    remaining_count = self.faiss_image_index.index.ntotal
    remaining_ids = set(self.faiss_image_index.aux_info['listing_id'])
    updated_embeddings = self.faiss_image_index.get_embeddings()
    updated_aux_info = self.faiss_image_index.aux_info

    # Verify the number of items removed and their absence
    self.assertEqual(remaining_count, initial_count - count_to_remove)
    self.assertNotIn(listing_id_to_remove, remaining_ids)

    for i, image_name in enumerate(sample_image_names_before):
      if image_name in updated_aux_info['image_name'].values:
        # Find the index of the image_name after removal
        new_index = updated_aux_info[updated_aux_info['image_name'] == image_name].index[0]
        # Fetch the embedding from the new index
        embedding_after = updated_embeddings[new_index]
        # Assert that the embedding has not changed
        np.testing.assert_array_almost_equal(embedding_after, sample_embeddings_before[i], decimal=6)



  def test_update(self):
    # Pick a random listing_id to update from the existing index
    listing_id_to_update = np.random.choice(self.faiss_image_index.aux_info['listing_id'].unique())

    # Record the initial total count of embeddings
    initial_count = self.faiss_image_index.index.ntotal
    initial_embeddings = self.faiss_image_index.get_embeddings().copy()
    initial_aux_info = self.faiss_image_index.aux_info.copy()

    # Select a few random entries to monitor before the update
    sample_indices = np.random.choice(initial_count, 3, replace=False)
    sample_embeddings_before = initial_embeddings[sample_indices]
    sample_image_names_before = initial_aux_info.iloc[sample_indices]['image_name'].values

    # Decide on a new number of records to update for the picked listing_id (could be more or less)
    new_record_count = np.random.randint(1, 5)  # Randomly choose between 1 and 4 new records
    new_embeddings = np.random.rand(new_record_count, self.embedding_dim_image).astype(np.float32)
    new_aux_info = pd.DataFrame({
      "listing_id": [listing_id_to_update] * new_record_count,
      "image_name": [f"{listing_id_to_update}_{i + 1}.jpg" for i in range(new_record_count)]
    })

    # Perform the update operation
    self.faiss_image_index.update(new_embeddings, new_aux_info)

    # Verify if the total count of embeddings is correct after update
    expected_count = initial_count - len(initial_aux_info.query(f"listing_id == '{listing_id_to_update}'")) + new_record_count
    self.assertEqual(self.faiss_image_index.index.ntotal, expected_count)

    # Fetch the updated embeddings and aux_info
    updated_embeddings = self.faiss_image_index.get_embeddings()
    updated_aux_info = self.faiss_image_index.aux_info

    # Ensure that the embeddings for the monitored entries are unchanged
    for i, image_name in enumerate(sample_image_names_before):
      if image_name in updated_aux_info['image_name'].values:
        # Find the index of the image_name after the update
        new_index = updated_aux_info[updated_aux_info['image_name'] == image_name].index[0]
        # Fetch the embedding from the new index
        embedding_after = updated_embeddings[new_index]
        # Assert that the embedding has not changed
        np.testing.assert_array_almost_equal(embedding_after, sample_embeddings_before[i], decimal=6)

     # Check that each new embedding matches its corresponding image name
    for idx, image_name in enumerate(new_aux_info['image_name']):
      new_index = updated_aux_info[updated_aux_info['image_name'] == image_name].index[0]
      embedding_after = updated_embeddings[new_index]
      # Assert that the new embedding is set correctly
      np.testing.assert_array_almost_equal(embedding_after, new_embeddings[idx], decimal=6)


  def test_search(self):
    # Load image
    image_path = "./data/niagara_fall.jpeg"
    image = Image.open(image_path)

    # Generate embedding from the image
    query_vec = self.image_embedder.embed_from_single_image(image=image)

    # Define the number of results to return
    topK = 5

    # Search using the generated query vector
    tops, scores = self.faiss_image_index.search(query_vectors=query_vec, topK=topK)

    # Validate the results
    self.assertEqual(len(tops), topK)
    self.assertEqual(len(scores), topK)

    # Ensure `tops` contains valid image names
    for display_key in tops:
      self.assertIsInstance(display_key, str)
      self.assertIn(display_key, self.faiss_image_index.aux_info[self.faiss_image_index.display_key].values)

    # Ensure `scores` are all floats and sorted in descending order
    for score in scores:
      self.assertIsInstance(score, float)
    self.assertTrue(all(earlier >= later for earlier, later in zip(scores, scores[1:])))


    # Ensure the scores are sorted in descending order
    self.assertTrue(all(earlier >= later for earlier, later in zip(scores, scores[1:])))


if __name__ == "__main__":
  unittest.main()
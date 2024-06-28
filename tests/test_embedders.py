import unittest
# from unittest.mock import patch, MagicMock
from pathlib import Path
import torch
from optimum.quanto import qfloat8, qint4, qint8

from PIL import Image

import numpy as np
import pandas as pd
from realestate_vss.models.embedding import OpenClipTextEmbeddingModel, OpenClipImageEmbeddingModel
from realestate_vss.models.embedding import HFCLIPImageEmbeddingModel, HFCliptTextEmbeddingModel

# python -m unittest test_embedders.TestEmbeddingModels.test_single_image_quantized

class TestEmbeddingModels(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.listing_df = pd.read_feather('./data/listing_df')

    model_name = 'ViT-L-14'
    pretrained='laion2b_s32b_b82k'
    model_id = 'laion/CLIP-ViT-L-14-laion2B-s32B-b82K'    # for huggingface model format

    device = torch.device('cpu')

    cls.openclip_image_embedder = OpenClipImageEmbeddingModel(model_name=model_name, pretrained=pretrained, device=device)
    cls.openclip_text_embedder = OpenClipTextEmbeddingModel(embedding_model=cls.openclip_image_embedder)

    cls.hfclip_image_embedder = HFCLIPImageEmbeddingModel(model_id=model_id, device=device)
    cls.hfclip_text_embedder = HFCliptTextEmbeddingModel(embedding_model=cls.hfclip_image_embedder)

    cls.q_openclip_image_embedder = OpenClipImageEmbeddingModel(model_name=model_name, 
                                                                pretrained=pretrained, 
                                                                device=device,
                                                                state_dict_path='./models/quantized_open_clip.pth',
                                                                weight_quant_type=qint8, 
                                                                activation_quant_type=None
                                                                )
    
    cls.q_openclip_text_embedder = OpenClipTextEmbeddingModel(embedding_model=cls.q_openclip_image_embedder)
    
    cls.q_hfclip_image_embedder = HFCLIPImageEmbeddingModel(model_id=model_id,
                                                            device=device,
                                                            state_dict_path='./models/quantized_hfclip.pth',
                                                            weight_quant_type=qint8, 
                                                            activation_quant_type=None
                                                            )
    
    cls.q_hfclip_text_embedder = HFCliptTextEmbeddingModel(embedding_model=cls.q_hfclip_image_embedder)

  def test_single_image(self):
    # Load image
    image_path = "./data/niagara_fall.jpeg"
    image = Image.open(image_path)

    openclip_img_embeddings = self.openclip_image_embedder.embed_from_single_image(image=image)
    hfclip_img_embeddings = self.hfclip_image_embedder.embed_from_single_image(image=image)

    # consistency of embedding between open clip and huggingface clipmodel
    are_close = np.allclose(openclip_img_embeddings, hfclip_img_embeddings, atol=1e-4)
    self.assertTrue(are_close, "Image Embeddings from OpenClip and HuggingFace are not consistent.")

  def test_multiple_images(self):
    image_paths = ['cn_tower.jpg','niagara_fall.jpeg','ugly_kitchen.jpg']
    images = [Image.open(f"./data/{img}") for img in image_paths]

    openclip_img_embeddings = self.openclip_image_embedder.embed_from_images(images=images, return_df=False)
    hfclip_img_embeddings = self.hfclip_image_embedder.embed_from_images(images=images, return_df=False)

    # consistency of embedding between open clip and huggingface clipmodel
    are_close = np.allclose(openclip_img_embeddings, hfclip_img_embeddings, atol=1e-4)
    self.assertTrue(are_close, "Embeddings from OpenClip and HuggingFace are not consistent.")

  def test_single_image_quantized(self):
    # Load image
    image_path = "./data/niagara_fall.jpeg"
    # image_path = "./data/ugly_kitchen.jpg"
    image = Image.open(image_path)

    openclip_img_embeddings = self.openclip_image_embedder.embed_from_single_image(image=image)
    hfclip_img_embeddings = self.hfclip_image_embedder.embed_from_single_image(image=image)

    q_openclip_img_embeddings = self.q_openclip_image_embedder.embed_from_single_image(image=image)
    q_hfclip_img_embeddings = self.q_hfclip_image_embedder.embed_from_single_image(image=image)

    # consistency of embedding between open clip and quantized open clip
    are_close = np.allclose(openclip_img_embeddings, q_openclip_img_embeddings, atol=1e-2)
    self.assertTrue(are_close, "Embeddings from OpenClip and Quantized OpenClip are not consistent.")

    # consistency of embedding between huggingface clipmodel and quantized huggingface clipmodel
    are_close = np.allclose(hfclip_img_embeddings, q_hfclip_img_embeddings, atol=1e-2)
    self.assertTrue(are_close, "Embeddings from HuggingFace and Quantized HuggingFace are not consistent.")

  def test_many_images_quantized(self):
    image_paths = ['cn_tower.jpg','niagara_fall.jpeg','ugly_kitchen.jpg']
    images = [Image.open(f"./data/{img}") for img in image_paths]

    openclip_img_embeddings = self.openclip_image_embedder.embed_from_images(images=images, return_df=False)
    hfclip_img_embeddings = self.hfclip_image_embedder.embed_from_images(images=images, return_df=False)

    q_openclip_img_embeddings = self.q_openclip_image_embedder.embed_from_images(images=images, return_df=False)
    q_hfclip_img_embeddings = self.q_hfclip_image_embedder.embed_from_images(images=images, return_df=False)

    # consistency of embedding between open clip and quantized open clip
    are_close = np.allclose(openclip_img_embeddings, q_openclip_img_embeddings, atol=1e-2)
    self.assertTrue(are_close, "Embeddings from OpenClip and Quantized OpenClip are not consistent.")

    # consistency of embedding between huggingface clipmodel and quantized huggingface clipmodel
    are_close = np.allclose(hfclip_img_embeddings, q_hfclip_img_embeddings, atol=1e-2)
    self.assertTrue(are_close, "Embeddings from HuggingFace and Quantized HuggingFace are not consistent.")

  def test_embeddings_df_output(self):
    image_paths = [Path('./data')/f for f in ['cn_tower.jpg','niagara_fall.jpeg','ugly_kitchen.jpg']]

    hfclip_img_embeddings_df = self.openclip_image_embedder.embed(image_paths=image_paths, return_df=True)
    openclip_img_embeddings_df = self.hfclip_image_embedder.embed(image_paths=image_paths, return_df=True)

    # Step 1: Assert DataFrame Shape
    expected_shape = (3, 3)  # 3 rows, 3 columns
    assert hfclip_img_embeddings_df.shape == expected_shape, "hfclip_img_embeddings_df shape mismatch"
    assert openclip_img_embeddings_df.shape == expected_shape, "openclip_img_embeddings_df shape mismatch"

    # Step 2: Assert Column Names
    expected_columns = ['listing_id', 'image_name', 'embedding']
    assert list(hfclip_img_embeddings_df.columns) == expected_columns, "hfclip_img_embeddings_df columns mismatch"
    assert list(openclip_img_embeddings_df.columns) == expected_columns, "openclip_img_embeddings_df columns mismatch"

    # Step 3: Assert 'listing_id' and 'image_name' Columns
    np.testing.assert_array_equal(hfclip_img_embeddings_df['listing_id'], openclip_img_embeddings_df['listing_id'], "listing_id mismatch")
    np.testing.assert_array_equal(hfclip_img_embeddings_df['image_name'], openclip_img_embeddings_df['image_name'], "image_name mismatch")

    # Step 4: Assert 'embedding' Column
    # Assuming a tolerance for floating point comparison
    tolerance = 1e-6
    for i in range(len(hfclip_img_embeddings_df)):
        np.testing.assert_allclose(hfclip_img_embeddings_df['embedding'].iloc[i], openclip_img_embeddings_df['embedding'].iloc[i], atol=tolerance, err_msg=f"Embedding mismatch at row {i}")

  def test_text_embedding(self):
    texts = ["cn tower", "niagara fall"]
    openclip_text_embeddings = self.openclip_text_embedder.embed_from_texts(texts=texts)
    hfclip_text_embeddings = self.hfclip_text_embedder.embed_from_texts(texts=texts)

    # consistency of embedding between open clip and huggingface clipmodel
    are_close = np.allclose(openclip_text_embeddings, hfclip_text_embeddings, atol=1e-4)
    self.assertTrue(are_close, "Text Embeddings from OpenClip and HuggingFace are not consistent.")

  def test_text_embedding_quantized(self):
    texts = ["cn tower", "niagara fall", "fixer upper ugly kitchen"]
    openclip_text_embeddings = self.openclip_text_embedder.embed_from_texts(texts=texts)
    hfclip_text_embeddings = self.hfclip_text_embedder.embed_from_texts(texts=texts)

    q_openclip_text_embeddings = self.q_openclip_text_embedder.embed_from_texts(texts=texts)
    q_hfclip_text_embeddings = self.q_hfclip_text_embedder.embed_from_texts(texts=texts)

    # consistency of embedding between open clip and quantized open clip
    for i in range(len(texts)):
      are_close = np.allclose(openclip_text_embeddings[i], q_openclip_text_embeddings[i], atol=1e-2)
      self.assertTrue(are_close, f"Embeddings from OpenClip and Quantized OpenClip are not consistent at {i}.")

    # consistency of embedding between huggingface clipmodel and quantized huggingface clipmodel
    for i in range(len(texts)):
      are_close = np.allclose(hfclip_text_embeddings[i], q_hfclip_text_embeddings[i], atol=1e-2)
      self.assertTrue(are_close, f"Embeddings from HuggingFace and Quantized HuggingFace are not consistent at {i}.")




# Run the tests
if __name__ == '__main__':
    unittest.main()
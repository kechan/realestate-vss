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
                                                                state_dict_path=Path.home()/'tmp'/'quantized_open_clip.pth',
                                                                weight_quant_type=qint8, 
                                                                activation_quant_type=None
                                                                )
    
    cls.q_hfclip_image_embedder = HFCLIPImageEmbeddingModel(model_id=model_id,
                                                            device=device,
                                                            state_dict_path=Path.home()/'tmp'/'quantized_hfclip.pth',
                                                            weight_quant_type=qint8, 
                                                            activation_quant_type=None
                                                            )

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

  def test_text_embedding(self):
    texts = ["cn tower", "niagara fall"]
    openclip_text_embeddings = self.openclip_text_embedder.embed_from_texts(texts=texts)
    hfclip_text_embeddings = self.hfclip_text_embedder.embed_from_texts(texts=texts)

    # consistency of embedding between open clip and huggingface clipmodel
    are_close = np.allclose(openclip_text_embeddings, hfclip_text_embeddings, atol=1e-4)
    self.assertTrue(are_close, "Text Embeddings from OpenClip and HuggingFace are not consistent.")



# Run the tests
if __name__ == '__main__':
    unittest.main()
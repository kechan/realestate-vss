from typing import List, Union, Optional
from pathlib import Path

import open_clip, torch
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import pandas as pd
from PIL import Image

from tqdm.auto import tqdm

from ..data.preprocess import read_and_preprocess_image
from realestate_spam.models.embedding import EmbeddingModel, InstructorEmbeddingModel
from realestate_vision.common.utils import get_listingId_from_image_name

class OpenClipEmbeddingModel:
  def __init__(self, model_name:str=None, pretrained:str=None, 
               embedding_model: Optional['OpenClipEmbeddingModel']=None,
               device=torch.device('cpu')):
    if embedding_model is None:
      self.model_name = model_name
      self.pretrained = pretrained
      model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)

      self.model = model.to(device)
      self.preprocess = preprocess
      self.model.eval()

      self.device = device
    else:
      if not isinstance(embedding_model, OpenClipEmbeddingModel):
        raise TypeError('embedding_model must be an instance of OpenClipEmbeddingModel')
      self.model_name = embedding_model.model_name
      self.pretrained = embedding_model.pretrained
      self.model = embedding_model.model
      self.preprocess = embedding_model.preprocess
      self.device = embedding_model.device

class OpenClipImageEmbeddingModel(OpenClipEmbeddingModel):
  def __init__(self, model_name: str=None, pretrained: str = None, 
                embedding_model: Optional['OpenClipEmbeddingModel']=None,
               device=torch.device('cpu')):
    super().__init__(model_name=model_name, pretrained=pretrained, embedding_model=embedding_model, device=device)
    
  def embed(self, image_paths: List[Path], return_df=True) -> pd.DataFrame:
    batch_size = 64

    image_names, embeddings = [], []

    with ProcessPoolExecutor(max_workers=12) as executor:
      for i in tqdm(range(0, len(image_paths), batch_size), desc='Processing images'):
        batch_paths = image_paths[i: i + batch_size]
        batch_images = list(executor.map(read_and_preprocess_image, batch_paths, [self.preprocess]*len(batch_paths)))

        # Stack images into a single tensor
        batch_tensor = torch.stack(batch_images, dim=0).to(self.device)

        with torch.no_grad():
          image_features = self.model.encode_image(batch_tensor, normalize=True).cpu().numpy()

        image_names.extend([p.name for p in batch_paths])
        embeddings.extend(list(image_features))

    if return_df:
      listingIds = [get_listingId_from_image_name(image_name) for image_name in image_names]
      return pd.DataFrame(data={'listing_id': listingIds, 'image_name': image_names, 'embedding': embeddings})
    else:
      return image_names, embeddings

  def embed_from_single_image(self, image: Image, return_df=False) -> pd.DataFrame:
    '''
    This method is for single PIL.Image and its likely used during VSS & point inference
    '''
    image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
    with torch.no_grad():
      image_features = self.model.encode_image(image_tensor, normalize=True).cpu().numpy()

    if return_df:
      return pd.DataFrame(data={'embedding': image_features})
    else:
      return image_features
  

class OpenClipTextEmbeddingModel(OpenClipEmbeddingModel):
  def __init__(self, model_name: str=None, pretrained: str = None, 
                embedding_model: Optional['OpenClipEmbeddingModel']=None,
               device=torch.device('cpu')):
    super().__init__(model_name=model_name, pretrained=pretrained, embedding_model=embedding_model, device=device)
    self.tokenizer = open_clip.get_tokenizer(self.model_name)

  def embed(self, df: pd.DataFrame, batch_size=128, return_df=True) -> pd.DataFrame:
    assert 'jumpId' in df.columns, 'df must have jumpId columns'
    assert 'remarks' in df.columns, 'df must have remarks columns'

    jumpIds, text_embeddings = [], []
    for i in tqdm(range(0, len(df), batch_size), desc='Processing remarks'):
      _jumpIds = df.iloc[i: i + batch_size].jumpId.values
      batch_remarks = df.iloc[i: i + batch_size].remarks.values

      with torch.no_grad():
        text_features = self.model.encode_text(self.tokenizer(batch_remarks).to(self.device), normalize=True).cpu().numpy()

      jumpIds.extend(list(_jumpIds))
      text_embeddings.extend(list(text_features))

    if return_df:
      return pd.DataFrame(data={'listing_id': jumpIds, 'embedding': text_embeddings})
    else:
      return jumpIds, text_embeddings

  def embed_from_texts(self, texts: List[str], batch_size=128):

    text_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc='Processing texts'):
      batch_texts = texts[i: i + batch_size]
      with torch.no_grad():
        text_features = self.model.encode_text(self.tokenizer(batch_texts).to(self.device), normalize=True).cpu().numpy()

      text_embeddings.extend(list(text_features))
    
    return text_embeddings
    

class TextEmbeddingModel:
  def __init__(self, model: Union[EmbeddingModel, InstructorEmbeddingModel], device=torch.device('cpu')):
    self.model = model   #.to(device)
    self.device = device

  def embed(self, df: pd.DataFrame, batch_size=128, return_df=True, verbose=False) -> pd.DataFrame:
    assert 'jumpId' in df.columns, 'df must have jumpId columns'
    assert 'remarks' in df.columns, 'df must have remarks columns'

    jumpIds = df.jumpId.values.tolist()
    text_embeddings = self.model.embed(df.remarks.values, batch_size=batch_size, html_unescape=False, verbose=verbose)

    if return_df:
      return pd.DataFrame(data={'listing_id': jumpIds, 'embedding': [v for v in text_embeddings]})
    else:
      return jumpIds, text_embeddings
    

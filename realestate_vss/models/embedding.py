from typing import List, Union, Optional
from pathlib import Path
from itertools import chain

import open_clip, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
# from torchvision import transforms
import torch.multiprocessing as mp
from torch.utils.data._utils.collate import default_collate

from transformers import CLIPProcessor, CLIPModel
from optimum.quanto import Calibration, freeze, qfloat8, qint4, qint8, quantize

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import pandas as pd
import numpy as np
from PIL import Image
import pillow_heif
pillow_heif.register_heif_opener()  

import spacy

from tqdm.auto import tqdm

from ..data.preprocess import read_and_preprocess_image, read_and_process_image
from realestate_spam.models.embedding import EmbeddingModel, InstructorEmbeddingModel
from realestate_vision.common.utils import get_listingId_from_image_name

NON_BLOCKING = False

class ListingImageDataset(Dataset):
  def __init__(self, image_paths: List[Path], transform):
    self.image_paths = image_paths
    self.transform = transform

  def __len__(self):
    return len(self.image_paths)

  def __getitem__(self, idx):
    path = self.image_paths[idx]
    try:
      img = Image.open(path).convert('RGB')
      img_tensor = self.transform(img)
      return img_tensor
    except Exception as e:
      print(f"Error processing image {path}: {e}")
      return torch.zeros((3, 224, 224))  # Return a blank tensor instead of None
  
class ListingTextDatasetOpenClip(Dataset):
  def __init__(self, df, tokenize_sentences=False):
    self.df = df
    self.tokenize_sentences = tokenize_sentences

  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx):
    row = self.df.iloc[idx]
    jumpId = row['jumpId']
    remark = row['remarks'] if pd.notna(row['remarks']) else ''
    return jumpId, remark
  
def custom_collate_fn(batch):
  """
  Custom collate function to handle batches when tokenize_sentences=True.
  It flattens the list of lists into a single list of sentences.
  """
  if isinstance(batch[0], list):
    # Flatten the list of lists
    flat_batch = [item for sublist in batch for item in sublist]
  else:
    flat_batch = batch
  
  jumpIds = []
  remark_chunk_ids = []
  sentences = []
  
  for item in flat_batch:
    jumpIds.append(item['jumpId'])
    if 'remark_chunk_id' in item:
      remark_chunk_ids.append(item['remark_chunk_id'])
    sentences.append(item['sentence'])
  
  if 'remark_chunk_id' in flat_batch[0]:
    return jumpIds, remark_chunk_ids, sentences
  else:
    return jumpIds, sentences

class ListingTextDataset(Dataset):
  def __init__(self, df, tokenize_sentences=False, nlp=None):
    self.df = df
    self.tokenize_sentences = tokenize_sentences
    self.nlp = nlp
    if self.tokenize_sentences and self.nlp is None:
      self.nlp = spacy.load('en_core_web_sm')

  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx):
    row = self.df.iloc[idx]
    jumpId = row['jumpId']
    remark = row['remarks'] if pd.notna(row['remarks']) else ''
    
    if self.tokenize_sentences:
      if isinstance(remark, str) and remark.strip():
        doc = self.nlp(remark)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        if not sentences:
          sentences = [remark.strip()]
      else:
        sentences = ['']
      
      # Generate chunk IDs
      data = []
      for sent_idx, sentence in enumerate(sentences):
        remark_chunk_id = f"{jumpId}_{sent_idx}"
        data.append({
          'jumpId': jumpId,
          'remark_chunk_id': remark_chunk_id,
          'sentence': sentence
        })
      return data  # List of dicts
    else:
      return {
        'jumpId': jumpId,
        'sentence': remark
      }


class OpenClipEmbeddingModel:
  def __init__(self, model_name:str=None, pretrained:str=None, 
               embedding_model: Optional['OpenClipEmbeddingModel']=None,
               device=torch.device('cpu'),
               state_dict_path: Optional[Path]=None,
               weight_quant_type = qint8,
               activation_quant_type = None):
    def build_attention_mask(num_pos):
      mask = torch.empty(num_pos, num_pos)
      mask.fill_(float("-inf"))
      mask.triu_(1)  # zero out the lower diagonal
      return mask
    
    if embedding_model is None:
      self.device = device      
      self.model_name = model_name
      self.pretrained = pretrained
      if state_dict_path is not None:
        with torch.device("meta"):
          model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=torch.device("meta"))
        if weight_quant_type is not None:
          quantize(model, weights=weight_quant_type, activations=activation_quant_type)
        model.to_empty(device=torch.device("cpu"))
        state_dict = torch.load(state_dict_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=True, assign=True)

        # attn_mask are not recovered from load_state_dict, hack it here
        model.register_buffer('attn_mask', build_attention_mask(model.context_length), persistent=False)
        freeze(model)
      else:
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)

      self.model = model.to(self.device)
      self.preprocess = preprocess
      self.model.eval()
      
    else:
      if not isinstance(embedding_model, OpenClipEmbeddingModel):
        raise TypeError('embedding_model must be an instance of OpenClipEmbeddingModel')
      self.model_name = embedding_model.model_name
      self.pretrained = embedding_model.pretrained
      self.model = embedding_model.model
      self.preprocess = embedding_model.preprocess
      self.device = embedding_model.device

class OpenClipImageEmbeddingModel(OpenClipEmbeddingModel):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    
  def embed(self, image_paths: List[Path], return_df=True, use_process_pool=False, num_workers=4, batch_size=64) -> pd.DataFrame:
    """
    Embed images into feature vectors using a pre-trained model.

    Parameters:
    -----------
    image_paths : List[Path]
        List of paths to the images to be embedded.
    return_df : bool, optional
        If True, returns a pandas DataFrame containing the embeddings. If False, returns a tuple of image names and embeddings. Default is True.
    use_process_pool : bool, optional
        If True, uses a process pool for parallel image processing. Default is False.
    num_workers : int, optional
        Number of worker threads to use for data loading. Default is 4.
    batch_size : int, optional
        Number of images to process in each batch. Default is 64.

    Returns:
    --------
    pd.DataFrame or Tuple[List[str], List[np.ndarray]]
        If return_df is True, returns a DataFrame with columns 'listing_id', 'image_name', and 'embedding'.
        If return_df is False, returns a tuple containing a list of image names and a list of embeddings.

    Notes:
    ------
    - If use_process_pool is True, images are processed in parallel using a process pool.
    - If use_process_pool is False, images are processed sequentially or using a DataLoader.
    - The function handles image reading, preprocessing, and embedding extraction.
    - The embeddings are generated using a pre-trained model and are returned as numpy arrays.
    """
    image_names, embeddings = [], []

    if use_process_pool:
      with ProcessPoolExecutor(max_workers=12) as executor:
        for i in tqdm(range(0, len(image_paths), batch_size), desc='Processing images'):
          batch_paths = image_paths[i: i + batch_size]
          batch_images = list(executor.map(read_and_preprocess_image, batch_paths, [self.preprocess]*len(batch_paths)))

          # Stack images into a single tensor
          batch_tensor = torch.stack(batch_images, dim=0).to(self.device, non_blocking=NON_BLOCKING)

          with torch.no_grad():
            image_features = self.model.encode_image(batch_tensor, normalize=True).cpu().numpy()

          image_names.extend([p.name for p in batch_paths])
          embeddings.extend(list(image_features))
    else:
      '''
      for i in tqdm(range(0, len(image_paths), batch_size), desc='Processing images'):
        batch_paths = image_paths[i: i + batch_size]
        
        # Process images in batches
        batch_images = []
        for path in batch_paths:
          try:
            img = Image.open(path).convert('RGB')
            img_tensor = self.preprocess(img)
            batch_images.append(img_tensor)
          except Exception as e:
            print(f"Error processing image {path}: {e}")
            continue

        if not batch_images:
          continue

        # Stack images into a single tensor
        batch_tensor = torch.stack(batch_images, dim=0).to(self.device)

        with torch.no_grad():
          image_features = self.model.encode_image(batch_tensor, normalize=True).cpu().numpy()

        image_names.extend([p.name for p in batch_paths])
        embeddings.extend(list(image_features))
      # '''

      # '''
      dataset = ListingImageDataset(image_paths, self.preprocess)
      dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

      image_names = [path.name for path in image_paths]

      for batch_tensors in tqdm(dataloader, desc='Processing images'):
        batch_tensor = batch_tensors.to(self.device, non_blocking=NON_BLOCKING)

        with torch.no_grad():
          image_features = self.model.encode_image(batch_tensor, normalize=True).cpu().numpy()

        embeddings.extend(list(image_features))
      # '''

    if return_df:
      listingIds = [get_listingId_from_image_name(image_name) for image_name in image_names]
      return pd.DataFrame(data={'listing_id': listingIds, 'image_name': image_names, 'embedding': embeddings})
    else:
      return image_names, embeddings

  def embed_from_single_image(self, image: Image.Image, return_df=False) -> pd.DataFrame:
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
    
  def embed_from_images(self, images: List[Image.Image], batch_size=64, return_df=False) -> pd.DataFrame:
    '''
    This method is for multiple PIL.Images and its likely used during VSS & point inference
    '''
    embeddings = []

    for i in tqdm(range(0, len(images), batch_size), desc='Processing images'):
      batch_images = images[i: i + batch_size]
      image_tensors = [self.preprocess(image).unsqueeze(0).to(self.device, non_blocking=NON_BLOCKING) for image in batch_images]
      image_tensors = torch.cat(image_tensors, dim=0)  # Concatenate along the batch dimension

      with torch.no_grad():
        image_features = self.model.encode_image(image_tensors, normalize=True).cpu().numpy()
        embeddings.extend(list(image_features))

    if return_df:
      return pd.DataFrame(data={'embedding': embeddings})
    else:
      return np.array(embeddings)
  

class OpenClipTextEmbeddingModel(OpenClipEmbeddingModel):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)    
    self.tokenizer = open_clip.get_tokenizer(self.model_name)
    self.nlp = None

  def embed(self, df: pd.DataFrame, batch_size=128, return_df=True, tokenize_sentences=False, use_dataloader=False, num_workers=4) -> pd.DataFrame:
    assert 'jumpId' in df.columns, 'df must have jumpId columns'
    assert 'remarks' in df.columns, 'df must have remarks columns'
    if tokenize_sentences and self.nlp is None:
      self.nlp = spacy.load('en_core_web_sm')

    jumpIds, text_embeddings, remark_chunk_ids, tokenized_sentences = [], [], [], []

    if not use_dataloader:

      if not tokenize_sentences:
        # sentence_batch, jumpId_batch, chunk_id_batch = [], [], []

        for i in tqdm(range(0, len(df), batch_size), desc='Processing remarks'):
          _jumpIds = df.iloc[i: i + batch_size].jumpId.values
          batch_remarks = df.iloc[i: i + batch_size].remarks.values

          with torch.no_grad():
            text_features = self.model.encode_text(self.tokenizer(batch_remarks).to(self.device, non_blocking=NON_BLOCKING), normalize=True).cpu().numpy()
            jumpIds.extend(list(_jumpIds))
            text_embeddings.extend(list(text_features))
      else:
        # Initialize lists to store the sentences, jumpIds, and chunk_ids for each batch
        sentence_batch, jumpId_batch, chunk_id_batch = [], [], []

        for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing remarks"):
          jump_id = row['jumpId']
          remark = row['remarks']

          # Tokenize the remark into sentences if required
          if tokenize_sentences:
              if isinstance(remark, str) and remark.strip():
                  doc = self.nlp(remark)
                  sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
                  if not sentences:
                      # Handle cases where SpaCy fails to tokenize into sentences
                      sentences = [remark.strip()]
              else:
                  # Handle empty or non-string remarks
                  sentences = ['']
          else:
              sentences = [remark.strip()] if isinstance(remark, str) else ['']

          # Iterate through each sentence and prepare batch lists
          for sent_idx, sentence in enumerate(sentences):
              remark_chunk_id = f"{jump_id}_{sent_idx}"

              sentence_batch.append(sentence)
              jumpId_batch.append(jump_id)
              chunk_id_batch.append(remark_chunk_id)

              # If the batch size is reached, encode and reset the batch lists
              if len(sentence_batch) >= batch_size:
                  # Tokenize and encode the batch
                  tokens = self.tokenizer(sentence_batch).to(self.device, non_blocking=NON_BLOCKING)
                  with torch.no_grad():
                      embeddings = self.model.encode_text(tokens, normalize=True).cpu().numpy()

                  # Append the results to the output lists
                  text_embeddings.extend(embeddings)
                  jumpIds.extend(jumpId_batch)
                  remark_chunk_ids.extend(chunk_id_batch)
                  tokenized_sentences.extend(sentence_batch)

                  # Clear the batch lists for the next batch
                  sentence_batch, jumpId_batch, chunk_id_batch = [], [], []

        # After processing all rows, handle any remaining sentences in the batch
        if sentence_batch:
            tokens = self.tokenizer(sentence_batch).to(self.device, non_blocking=NON_BLOCKING)
            with torch.no_grad():
                embeddings = self.model.encode_text(tokens, normalize=True).cpu().numpy()

            text_embeddings.extend(embeddings)
            jumpIds.extend(jumpId_batch)
            remark_chunk_ids.extend(chunk_id_batch)
            tokenized_sentences.extend(sentence_batch)

    else:          
      dataset = ListingTextDatasetOpenClip(df, tokenize_sentences)
      dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

      with torch.no_grad():
        for batch in tqdm(dataloader, desc='Processing remarks'):
          batch_jumpIds, batch_remarks = batch
          
          if tokenize_sentences:
            sentence_batch, jumpId_batch, chunk_id_batch = [], [], []
            for idx, remark in enumerate(batch_remarks):
              if remark:
                doc = self.nlp(remark)
              else:
                doc = self.nlp(' ')
              for sent_idx, sent in enumerate(doc.sents):
                sentence_batch.append(sent.text)
                jumpId_batch.append(batch_jumpIds[idx])
                chunk_id_batch.append(f"{batch_jumpIds[idx]}_{sent_idx}")
            
            if sentence_batch:  # Process sentences if any
              text_inputs = self.tokenizer(sentence_batch).to(self.device, non_blocking=NON_BLOCKING)
              text_features = self.model.encode_text(text_inputs, normalize=True).cpu().numpy()
              text_embeddings.extend(list(text_features))
              jumpIds.extend(jumpId_batch)
              remark_chunk_ids.extend(chunk_id_batch)
          else:
            text_inputs = self.tokenizer(batch_remarks).to(self.device, non_blocking=NON_BLOCKING)
            text_features = self.model.encode_text(text_inputs, normalize=True).cpu().numpy()
            jumpIds.extend(batch_jumpIds)
            text_embeddings.extend(list(text_features))

    if return_df:
      if tokenize_sentences:
        return pd.DataFrame(data={'listing_id': jumpIds, 'remark_chunk_id': remark_chunk_ids, 'sentence': tokenized_sentences, 'embedding': text_embeddings})
      else:
        return pd.DataFrame(data={'listing_id': jumpIds, 'embedding': text_embeddings})
    else:
      if tokenize_sentences:
        return jumpIds, remark_chunk_ids, text_embeddings
      else:
        return jumpIds, text_embeddings

  def embed_from_texts(self, texts: List[str], batch_size=128):

    text_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc='Processing texts'):
      batch_texts = texts[i: i + batch_size]
      with torch.no_grad():
        text_features = self.model.encode_text(self.tokenizer(batch_texts).to(self.device, non_blocking=NON_BLOCKING), normalize=True).cpu().numpy()

      text_embeddings.extend(list(text_features))
    
    return text_embeddings



class HFCLIPModel:
  """
  Same as OpenClipEmbeddingModel but uses HuggingFace's transformers library
  """
  def __init__(self, model_id: str = None, 
               embedding_model: Optional['HFCLIPModel']=None,
               device=torch.device('cpu'),
               state_dict_path: Optional[Path]=None,
               weight_quant_type = qint8,
               activation_quant_type = None):
    if embedding_model is None:
      self.model_id = model_id
      self.device = device

      
      if state_dict_path is not None:
        with torch.device("meta"):
          model = CLIPModel.from_pretrained(model_id)
        if weight_quant_type is not None:
          quantize(model, weights=weight_quant_type, activations=activation_quant_type)
        model.to_empty(device=torch.device("cpu"))
        state_dict = torch.load(state_dict_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=True, assign=True)

        # positions_ids are not recovered from load_state_dict
        # extra hack to make it work
        num_positions = model.vision_model.embeddings.num_positions
        model.vision_model.embeddings.register_buffer('position_ids', torch.arange(num_positions).expand((1, -1)), persistent=False)

        max_position_embeddings = model.text_model.config.max_position_embeddings
        model.text_model.embeddings.register_buffer('position_ids', torch.arange(max_position_embeddings).expand((1, -1)), persistent=False)

        freeze(model)
      else:
        model = CLIPModel.from_pretrained(model_id)
      
      self.model = model.to(self.device)

      self.processor = CLIPProcessor.from_pretrained(model_id)
      self.model.eval()
      
    else:
      if not isinstance(embedding_model, HFCLIPModel):
        raise TypeError('embedding_model must be an instance of HFCLIPModel')
      self.model_id = embedding_model.model_id
      self.model = embedding_model.model
      self.processor = embedding_model.processor
      self.device = embedding_model.device

    if self.device.type == 'mps':
      self._setup_multiprocessing_for_mps()

  @staticmethod
  def _setup_multiprocessing_for_mps():
    if torch.backends.mps.is_available():
      try:
        # Set the 'fork' method only for MPS
        mp.set_start_method('fork', force=True)
        print("Using 'folk' start method for mps")
      except RuntimeError:
        print("Multiprocessing start method is already set to 'fork'")

class HFCLIPImageEmbeddingModel(HFCLIPModel):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
  
  def transform(self, img):
    return self.processor(images=img, return_tensors="pt")['pixel_values'].squeeze(0)

  def embed(self, image_paths: List[Path], return_df=True, num_workers=4, batch_size=64) -> pd.DataFrame:    
    dataset = ListingImageDataset(image_paths, self.transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    """
    print("no dataloader")
    image_names, embeddings = [], []
    for i in tqdm(range(0, len(image_paths), batch_size), desc='Processing images'):
      batch_paths = image_paths[i: i + batch_size]
      # batch_images = list(executor.map(read_and_process_image, batch_paths, [self.processor]*len(batch_paths)))
      images = [Image.open(img_path).convert("RGB") for img_path in batch_paths]
      batch_tensor = self.processor(images=images, return_tensors="pt")['pixel_values'].to(self.device)

      with torch.no_grad():
        image_features = self.model.get_image_features(pixel_values=batch_tensor)
        image_features = F.normalize(image_features, p=2, dim=1)
        image_features = image_features.cpu().numpy()

      image_names.extend([p.name for p in batch_paths])
      embeddings.extend(list(image_features))
    # """
    # """
    image_names = [path.name for path in image_paths]
    embeddings = []
    with torch.no_grad():
      for batch_tensor in tqdm(dataloader, desc='Processing images'):
        batch_tensor = batch_tensor.to(self.device)
        image_features = self.model.get_image_features(pixel_values=batch_tensor)
        image_features = F.normalize(image_features, p=2, dim=1)
        embeddings.extend(image_features.cpu().numpy())
    # """

    if return_df:
      listingIds = [get_listingId_from_image_name(image_name) for image_name in image_names]
      return pd.DataFrame(data={'listing_id': listingIds, 'image_name': image_names, 'embedding': embeddings})
    else:
      return image_names, embeddings

  def embed_from_single_image(self, image: Image.Image, return_df=False) -> pd.DataFrame:
    '''
    This method is for single PIL.Image and its likely used during VSS & point inference
    '''
    image_tensor = self.processor(images=image, return_tensors='pt')['pixel_values'].to(self.device)
    with torch.no_grad():
      image_features = self.model.get_image_features(image_tensor)
      image_features = F.normalize(image_features, p=2, dim=1)
      image_features = image_features.cpu().numpy()

    if return_df:
      return pd.DataFrame(data={'embedding': image_features.tolist()})
    else:
      return image_features
    
  def embed_from_images(self, images: List[Image.Image], batch_size=64, return_df=False) -> pd.DataFrame:
    embeddings = []
    for i in tqdm(range(0, len(images), batch_size), desc='Processing images'):
      batch_images = images[i: i + batch_size]
      image_tensor = self.processor(images=batch_images, return_tensors='pt')['pixel_values'].to(self.device)

      with torch.no_grad():
        image_features = self.model.get_image_features(image_tensor)
        image_features = F.normalize(image_features, p=2, dim=1)
        image_features = image_features.cpu().numpy()
        embeddings.extend(list(image_features))

    if return_df:
      return pd.DataFrame(data={'embedding': embeddings})
    else:
      return np.array(embeddings)
    
class HFClipTextEmbeddingModel(HFCLIPModel):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.nlp = None

  def embed(self, df: pd.DataFrame, batch_size=128, return_df=True, tokenize_sentences=False, use_dataloader=False, num_workers=4) -> pd.DataFrame:
    assert 'jumpId' in df.columns, 'df must have jumpId columns'
    assert 'remarks' in df.columns, 'df must have remarks columns'
    if tokenize_sentences and self.nlp is None:
      self.nlp = spacy.load('en_core_web_sm')

    jumpIds, text_embeddings, remark_chunk_ids = [], [], []

    if not use_dataloader:
      if not tokenize_sentences:
        for i in tqdm(range(0, len(df), batch_size), desc='Processing remarks'):
          _jumpIds = df.iloc[i: i + batch_size].jumpId.values
          batch_remarks = df.iloc[i: i + batch_size].remarks.values

          with torch.no_grad():
            text_inputs = self.processor(text=batch_remarks, padding=True, truncation=True, return_tensors="pt")
            text_inputs = {name: tensor.to(self.device) for name, tensor in text_inputs.items()}
            text_features = self.model.get_text_features(**text_inputs)
            text_features = F.normalize(text_features, p=2, dim=1).cpu().numpy()
            
            jumpIds.extend(list(_jumpIds))
            text_embeddings.extend(list(text_features))
      else:
        sentence_batch, jumpId_batch, chunk_id_batch = [], [], []

        for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing remarks"):
          jump_id = row['jumpId']
          remark = row['remarks']

          if isinstance(remark, str) and remark.strip():
            doc = self.nlp(remark)
            sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
            if not sentences:
              sentences = [remark.strip()]
          else:
            sentences = ['']

          for sent_idx, sentence in enumerate(sentences):
            remark_chunk_id = f"{jump_id}_{sent_idx}"

            sentence_batch.append(sentence)
            jumpId_batch.append(jump_id)
            chunk_id_batch.append(remark_chunk_id)

            # If the batch size is reached, encode and reset the batch lists            
            if len(sentence_batch) >= batch_size:
              text_inputs = self.processor(text=sentence_batch, padding=True, truncation=True, return_tensors="pt")
              text_inputs = {name: tensor.to(self.device) for name, tensor in text_inputs.items()}

              with torch.no_grad():                
                embeddings = self.model.get_text_features(**text_inputs)
                embeddings = F.normalize(embeddings, p=2, dim=1).cpu().numpy()

              text_embeddings.extend(list(embeddings))
              jumpIds.extend(jumpId_batch)
              remark_chunk_ids.extend(chunk_id_batch)

              # Clear the batch lists for the next batch
              sentence_batch, jumpId_batch, chunk_id_batch = [], [], []

        # After processing all rows, handle any remaining sentences in the batch
        if sentence_batch:
          text_inputs = self.processor(text=sentence_batch, padding=True, truncation=True, return_tensors="pt")
          text_inputs = {name: tensor.to(self.device) for name, tensor in text_inputs.items()}

          with torch.no_grad():
            embeddings = self.model.get_text_features(**text_inputs)
            embeddings = F.normalize(embeddings, p=2, dim=1).cpu().numpy()

          text_embeddings.extend(list(embeddings))
          jumpIds.extend(jumpId_batch)
          remark_chunk_ids.extend(chunk_id_batch)

    else:
      dataset = ListingTextDataset(df, tokenize_sentences, self.nlp)
      dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        shuffle=False,
        collate_fn=custom_collate_fn
      )

      with torch.no_grad():
        for batch in tqdm(dataloader, desc='Processing remarks'):
          if tokenize_sentences:
            batch_jumpIds, batch_chunk_ids, batch_sentences = batch
            # Filter out empty sentences if necessary
            non_empty = [s.strip() != '' for s in batch_sentences]
            filtered_sentences = [s for s, keep in zip(batch_sentences, non_empty) if keep]
            filtered_jumpIds = [jid for jid, keep in zip(batch_jumpIds, non_empty) if keep]
            filtered_chunk_ids = [cid for cid, keep in zip(batch_chunk_ids, non_empty) if keep]
            
            if not filtered_sentences:
              continue
            
            text_inputs = self.processor(
              text=filtered_sentences, 
              padding=True, 
              truncation=True, 
              return_tensors="pt"
            ).to(self.device)
            # text_inputs = {name: tensor.to(self.device) for name, tensor in text_inputs.items()}
            
            embeddings = self.model.get_text_features(**text_inputs)
            embeddings = F.normalize(embeddings, p=2, dim=1).cpu().numpy()
            
            text_embeddings.extend(embeddings.tolist())
            jumpIds.extend(filtered_jumpIds)
            remark_chunk_ids.extend(filtered_chunk_ids)
          else:
            batch_jumpIds, batch_sentences = batch
            text_inputs = self.processor(
              text=batch_sentences, 
              padding=True, 
              truncation=True, 
              return_tensors="pt"
            )
            text_inputs = {name: tensor.to(self.device) for name, tensor in text_inputs.items()}
            
            embeddings = self.model.get_text_features(**text_inputs)
            embeddings = F.normalize(embeddings, p=2, dim=1).cpu().numpy()
            
            text_embeddings.extend(embeddings.tolist())
            jumpIds.extend(batch_jumpIds)

      """
      with torch.no_grad():
        for batch in tqdm(dataloader, desc='Processing remarks'):
          batch_jumpIds, batch_remarks = batch
          
          if tokenize_sentences:
            sentence_batch, jumpId_batch, chunk_id_batch = [], [], []
            for idx, remark in enumerate(batch_remarks):
              if remark:
                doc = self.nlp(remark)
              else:
                doc = self.nlp(' ')
              for sent_idx, sent in enumerate(doc.sents):
                sentence_batch.append(sent.text)
                jumpId_batch.append(batch_jumpIds[idx])
                chunk_id_batch.append(f"{batch_jumpIds[idx]}_{sent_idx}")
            
            if sentence_batch:  # Process sentences if any
              text_inputs = self.processor(text=sentence_batch, padding=True, truncation=True, return_tensors="pt")
              text_inputs = {name: tensor.to(self.device) for name, tensor in text_inputs.items()}
              text_features = self.model.get_text_features(**text_inputs)
              text_features = F.normalize(text_features, p=2, dim=1).cpu().numpy()
              text_embeddings.extend(list(text_features))
              jumpIds.extend(jumpId_batch)
              remark_chunk_ids.extend(chunk_id_batch)
          else:
            text_inputs = self.processor(text=batch_remarks, padding=True, truncation=True, return_tensors="pt")
            text_inputs = {name: tensor.to(self.device) for name, tensor in text_inputs.items()}
            text_features = self.model.get_text_features(**text_inputs)
            text_features = F.normalize(text_features, p=2, dim=1).cpu().numpy()
            jumpIds.extend(batch_jumpIds)
            text_embeddings.extend(list(text_features))
      """

    
    if return_df:
      if tokenize_sentences:
        return pd.DataFrame(data={'listing_id': jumpIds, 'remark_chunk_id': remark_chunk_ids, 'embedding': text_embeddings})
      else:
        return pd.DataFrame(data={'listing_id': jumpIds, 'embedding': text_embeddings})
    else:
      if tokenize_sentences:
        return jumpIds, remark_chunk_ids, text_embeddings
      else:
        return jumpIds, text_embeddings
      
  def embed_from_texts(self, texts: List[str], batch_size=128):

    text_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc='Processing texts'):
      batch_texts = texts[i: i + batch_size]
      with torch.no_grad():
        # text_features = self.model.encode_text(self.tokenizer(batch_texts).to(self.device), normalize=True).cpu().numpy()
        text_inputs = self.processor(text=batch_texts, padding=True, truncation=True, return_tensors="pt")
        text_inputs = {name: tensor.to(self.device) for name, tensor in text_inputs.items()}

        text_features = self.model.get_text_features(**text_inputs)
        text_features = F.normalize(text_features, p=2, dim=1).cpu().numpy()

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
    

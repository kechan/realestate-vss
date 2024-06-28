from typing import Dict, Optional, Any, List, Union, Tuple

import torch, re, copy, time
import torch.nn.functional as F
from torch.utils.data import DataLoader
from optimum.quanto import Calibration, freeze, qfloat8, qint4, qint8, quantize

from PIL import Image
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from ..data.index import FaissIndex

import pillow_heif
pillow_heif.register_heif_opener()

from realestate_vision.common.utils import get_listingId_from_image_name

class QuantizedModelEvaluator:
  def __init__(self, 
               baseline_model, 
               processor: Optional[Any] = None, 
               tokenizer: Optional[Any] = None, 
               quantization_config: Dict = {}, 
               device=torch.device('cpu')):
    self.baseline_model = baseline_model
    self.processor = processor   # model specific processor (could be for either image or text)
    self.tokenizer = tokenizer   # tokenizer usually for text tasks

    self.quantization_config = quantization_config
    self.device = device
    self.quantized_model = self.setup_quantized_model()

    # ensure baseline is on the device and for eval
    self.baseline_model.to(self.device)
    self.baseline_model.eval()

  def setup_quantized_model(self):
    """Setup the quantized model based on the configuration."""
    method = self.quantization_config['method']
    w_qtype = self._qtype(self.quantization_config.get('weight_qtype'))
    a_qtype = self._qtype(self.quantization_config.get('activation_qtype'))

    if method == "optimum.quanto":
      model = copy.deepcopy(self.baseline_model)
      if w_qtype is not None:
        quantize(model, weights=w_qtype, activations=a_qtype)  # done this in place
        freeze(model)  # TODO: if activations need quantization, we need to calibrate
    
    elif method == "torch.quantization":
      # Setup model for torch quantization
      # model = # Your method to create and quantize the model using PyTorch
      pass
    elif method == "onnx":
      # Setup model for ONNX quantization
      # model = # Your method to create and quantize the model using ONNX
      pass
    else:
      raise ValueError("Unsupported quantization configuration")
    
    model.to(self.device)
    model.eval()  # Ensure the model is in evaluation mode
    return model
  
  def compare_module_sizes(self):
    """Compare the memory footprint of the baseline and quantized model.
    Returns:
      baseline_sizes: dict, memory footprint of the baseline model in GB.
      quantized_sizes: dict, memory footprint of the quantized model in GB.
    """
    baseline_sizes = compute_module_sizes(self.baseline_model)[''] * 1e-9
    quantized_sizes = compute_module_sizes(self.quantized_model)[''] * 1e-9
    return baseline_sizes, quantized_sizes
  
  def load_image(self, image_path) -> Image.Image:
    image = Image.open(image_path).convert('RGB')
    return image
  
  def preprocess_image(self, images: List[Image.Image], device=None) -> torch.Tensor:
    """
    Preprocess the images using the model's processor.
    device is provided such that we can override it to be on CPU in the context of Dataset
    """
    image_inputs = self.processor(images=images, return_tensors="pt")

    # move to device 
    if device is None:  # the default is to use the model's device
      image_inputs = {name: tensor.to(self.device) for name, tensor in image_inputs.items()}
    else:
      image_inputs = {name: tensor.to(device) for name, tensor in image_inputs.items()}

    return image_inputs
  
  def preprocess_text(self, texts: List[str]) -> torch.Tensor:
    text_inputs = self.processor(text=texts, padding=True, truncation=True, return_tensors="pt")

    # move to device
    text_inputs = {name: tensor.to(self.device) for name, tensor in text_inputs.items()}

    return text_inputs
  
  def run_image_inference(self, model, inputs):
    """Extract features using the given model."""
    with torch.no_grad():
      features = model.get_image_features(**inputs)  # Adjust based on your model's method
      features = F.normalize(features, p=2, dim=1)
    return features
  
  def run_text_inference(self, model, inputs):
    """Extract features using the given model."""
    with torch.no_grad():
      features = model.get_text_features(**inputs)  # Adjust based on your model's method
      features = F.normalize(features, p=2, dim=1)
    return
  
  def evaluate_latency(self, function, *args):
    start_time = time.time()
    results = function(*args)
    end_time = time.time()
    return results, end_time - start_time
  
  def compare_image_latency(self, image_paths: List[Union[str, Path]]) -> Dict[str, float]:
    """Compare the latency of the baseline and quantized model for image inference."""

    images = [self.load_image(image_path) for image_path in image_paths]  # get PIL Images
    image_inputs = self.preprocess_image(images=images)

    baseline_results, baseline_latency = self.evaluate_latency(self.run_image_inference, 
                                                               self.baseline_model, 
                                                               image_inputs)
    
    quantized_results, quantized_latency = self.evaluate_latency(self.run_image_inference, 
                                                                 self.quantized_model, 
                                                                 image_inputs)

    return {
      "baseline_latency": baseline_latency,
      "quantized_latency": quantized_latency
    }
  
  def compare_text_latency(self, texts: List[str]) -> Dict[str, float]:
    """Compare the latency of the baseline and quantized model for text inference."""

    text_inputs = self.preprocess_text(texts=texts)

    baseline_results, baseline_latency = self.evaluate_latency(self.run_text_inference, 
                                                               self.baseline_model, 
                                                               text_inputs)
    quantized_results, quantized_latency = self.evaluate_latency(self.run_text_inference, 
                                                                 self.quantized_model, 
                                                                 text_inputs)

    return {
      "baseline_latency": baseline_latency,
      "quantized_latency": quantized_latency
    }

  class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, evaluator: "QuantizedModelEvaluator", image_paths):
      self.evaluator = evaluator
      self.image_paths = image_paths

    def __len__(self):
      return len(self.image_paths)

    def __getitem__(self, idx):
      image_path = self.image_paths[idx]
      image = self.evaluator.load_image(image_path)
      # image_inputs = self.evaluator.processor(images=[image], return_tensors="pt")
      image_inputs = self.evaluator.preprocess_image(images=[image])

      return {name: tensor.squeeze(0) for name, tensor in image_inputs.items()}

  # Evaluating embedding "errors" due to model quantization
  def prepare_image_baseline_resultset(self, image_paths: List[Union[str, Path]], return_df=True) -> Union[np.ndarray, pd.DataFrame]:
    batch_size = 128
    dataset = QuantizedModelEvaluator.ImageDataset(self, image_paths=image_paths)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    all_image_features = []

    for batch in tqdm(dataloader):
      image_features = self.run_image_inference(self.baseline_model, batch).cpu().numpy()
      all_image_features.append(image_features)
    
    image_features = np.vstack(all_image_features)

    if return_df:
      df = pd.DataFrame(data={
        'image': [p.name for p in image_paths], 
        'embedding': list(image_features)
        })
      return df
    else:
      return image_features
    
  def prepare_image_quantized_resultset(self, image_paths: List[Union[str, Path]], return_df=True) -> Union[np.ndarray, pd.DataFrame]:
    batch_size = 128
    dataset = QuantizedModelEvaluator.ImageDataset(self, image_paths=image_paths)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    all_image_features = []

    for batch in tqdm(dataloader):
      image_features = self.run_image_inference(self.quantized_model, batch).cpu().numpy()
      all_image_features.append(image_features)
    
    image_features = np.vstack(all_image_features)

    if return_df:
      df = pd.DataFrame(data={
        'image': [p.name for p in image_paths], 
        'embedding': list(image_features)
        })
      return df
    else:
      return image_features

  def compute_quantization_errors(self, baseline_df: pd.DataFrame, quantized_df: pd.DataFrame) -> Dict[str, float]:
    """
    the pandas dataframe should be output from prepare_image_baseline_resultset and prepare_image_quantized_resultset
    """
    baseline_embeddings = np.stack(baseline_df['embedding'].values)
    quantized_embeddings = np.stack(quantized_df['embedding'].values)

    rmse, cosine_dist = vec_dist(baseline_embeddings, quantized_embeddings)

    # return rmse, cosine_dist
    return {'mean(rmsr)': np.mean(rmse), 
            'mean(cosine_dist)': np.mean(cosine_dist)
            }


  # Evaluate search "accuracy" performance using Faiss
  def prepare_faiss_image_index(self, baseline_df: pd.DataFrame) -> FaissIndex:
    baseline_df = baseline_df.rename(columns={'image': 'image_name'})
    baseline_df['listing_id'] = baseline_df.image_name.apply(get_listingId_from_image_name)
    image_aux_info = baseline_df.drop(columns=['embedding'])
    faiss_image_index = FaissIndex(embeddings=np.stack(baseline_df.embedding.values), 
                                    aux_info=image_aux_info, 
                                    aux_key='listing_id',
                                    display_key='image_name'
                                    )
    self.faiss_image_index = faiss_image_index
    return self.faiss_image_index
  

  def evaluate_search_accuracy(self, baseline_df: pd.DataFrame, quantized_df: pd.DataFrame, topk=50) -> Dict[str, float]:

    assert np.all(np.equal(baseline_df.image.values, quantized_df.image.values)), "Image names do not match in dfs"

    accuracies = []
    for k, row in tqdm(quantized_df.iterrows()):
      quantized_vec = row.embedding.reshape(1, -1)
      vec = baseline_df.iloc[k].embedding.reshape(1, -1)
      
      baseline_results, _ = self.faiss_image_index.search(query_vectors=vec, topK=topk)
      quantized_results, _ = self.faiss_image_index.search(query_vectors=quantized_vec, topK=topk)

      accuracies.append(self.evaluate_search_results(baseline_results, quantized_results, topk=topk, groupby_listing=True))

    return {'mean_accuracy': np.mean(accuracies)}


  def evaluate_search_results(self, baseline_results: List[str], quantized_results: List[str], topk=50, groupby_listing=False):
    """
    Evaluate the accuracy of search results from a quantized model compared with baseline results.
    
    This function looks at the topK items from the baseline and quantized results, counts how many are the same, 
    and divides by the total topK. This is useful for comparing search results from a vector obtained by a 
    quantized CLIP model versus one without quantization.

    Parameters:
    baseline_results (List[str]): The search results from the baseline model.
    quantized_results (List[str]): The search results from the quantized model.
    topk (int): The number of top results to consider for the evaluation.
    groupby_listing (bool): Whether to group results by unique listings.

    Returns:
    float: The accuracy of the quantized model's search results compared to the baseline.
    """
    
    if groupby_listing:
      baseline_results = list(set([result.split('_')[0] for result in baseline_results[:topk]]))
      quantized_results = list(set([result.split('_')[0] for result in quantized_results[:topk]]))
      topk = min(len(baseline_results), len(quantized_results))

    n_intersection = len(set(baseline_results).intersection(set(quantized_results)))

    return n_intersection / topk
  

  def save_quantized_model(self, file_path: Union[str, Path]):
    # save the state dict
    torch.save(self.quantized_model.state_dict(), file_path)

  def _qtype(self, type_: str):    
    if type_ == "int8":
      return qint8
    elif type_ == "int4":
      return qint4
    elif type_ == "float8":
      return qfloat8
    elif type_ == "none" or type_ is None:
      return None
    else:
      raise ValueError(f"Unsupported quantization type: {type_}")
     



# helpers for memory footprint estimate
def named_module_tensors(module, recurse=False):
  for named_parameter in module.named_parameters(recurse=recurse):
    name, val = named_parameter
    flag = True
    if hasattr(val,"_data") or hasattr(val,"_scale"):
      if hasattr(val,"_data"):
        yield name + "._data", val._data
      if hasattr(val,"_scale"):
        yield name + "._scale", val._scale
    else:
      yield named_parameter

  for named_buffer in module.named_buffers(recurse=recurse):
    yield named_buffer

def dtype_byte_size(dtype):
  """
  Returns the size (in bytes) occupied by one parameter of type `dtype`.
  """
  import re
  if dtype == torch.bool:
      return 1 / 8
  bit_search = re.search(r"[^\d](\d+)$", str(dtype))
  if bit_search is None:
      raise ValueError(f"`dtype` is not a valid dtype: {dtype}.")
  bit_size = int(bit_search.groups()[0])
  return bit_size // 8
    
def compute_module_sizes(model):
  """
  Compute the size of each submodule of a given model.
  """
  from collections import defaultdict
  module_sizes = defaultdict(int)
  for name, tensor in named_module_tensors(model, recurse=True):
    size = tensor.numel() * dtype_byte_size(tensor.dtype)
    name_parts = name.split(".")
    for idx in range(len(name_parts) + 1):
      module_sizes[".".join(name_parts[:idx])] += size

  return module_sizes


def vec_dist(v1: np.ndarray, v2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  if v1.ndim == 1:
    v1 = v1.reshape(1, -1)
  if v2.ndim == 1:
    v2 = v2.reshape(1, -1)

  rmse = np.sqrt(np.mean((v1 - v2) ** 2, axis=-1))  # <sqrt|v1 - v2|^2>

  cosine_sim = np.einsum('ij,ij->i', v1, v2)
  cosine_dist = 1 - cosine_sim

  return rmse, cosine_dist
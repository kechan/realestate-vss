from typing import List, Union, Optional

from pathlib import Path

import pandas as pd
import numpy as np
import faiss, gc

class FaissIndex:
  # def __init__(self, embeddings: np.ndarray, aux_info: pd.DataFrame, aux_key: str = None):
  def __init__(self, 
               embeddings: Optional[np.ndarray] = None, 
               aux_info: Optional[pd.DataFrame] = None, 
               aux_key: Optional[str] = None, 
               filepath: Optional[Path] = None):
    if filepath is not None:
      self.load(filepath)
    else: 
      if embeddings is None or aux_info is None:
        raise ValueError('Both embeddings and aux_info must be provided if not loading from file.')
      
      if isinstance(aux_info, pd.DataFrame):
        if aux_key is None or aux_key not in aux_info.columns:
          raise ValueError('aux_key is required and must be in aux_info.columns')      
      
      self.embeddings = embeddings
      self.index = faiss.IndexFlatIP(embeddings.shape[1])
      self.index.add(embeddings)

      self.aux_info = aux_info
      # just be paranoid and do a defrag here, we don't want non contiguous index number
      self.aux_info.defrag_index(inplace=True)
      self.aux_key = aux_key

      # sanity check
      assert embeddings.shape[0] == aux_info.shape[0], 'embeddings and aux_info must have the same number of rows'    

      # explit delete of self.embeddings to save memory, since we don't need it anymore
      del self.embeddings
      gc.collect()

  def add(self, embeddings: np.array, aux_info: pd.DataFrame):
    assert len(embeddings) == len(aux_info), 'each vector in embeddings must have corresponding aux_info'
    self.index.add(embeddings)
    self.aux_info = pd.concat([self.aux_info, aux_info], axis=0, ignore_index=True)

  def remove(self, items: List[str]):
    ids_to_remove = list(self.aux_info.q(f"{self.aux_key}.isin(@items)").index)
    # print(f'removing {ids_to_remove}')
    self.index.remove_ids(np.array(ids_to_remove, dtype='int64'))

    # manage aux_info
    self.aux_info.drop(index=ids_to_remove, inplace=True)
    self.aux_info.defrag_index(inplace=True)

  def update(self, embeddings: np.array, aux_info: pd.DataFrame):
    items = list(aux_info[self.aux_key].values)
    self.remove(items)
    self.add(embeddings, aux_info)

  def search(self, query_vectors, topK):
    scores, I = self.index.search(query_vectors, topK)
    # print(I)

    tops = self.aux_info.iloc[I[0, :]][self.aux_key].values.tolist()

    return tops, scores[0].tolist()
  
  def save(self, filepath: Path):
    filepath = str(filepath) 
    faiss.write_index(self.index, filepath + '.index')
    self.aux_info.to_feather(filepath + '.aux_info_df')
    # save aux_key
    with open(filepath + '.aux_key', 'w') as f:
      f.write(self.aux_key)
        
  def load(self, filepath: Path):
    # Load the FAISS index from disk
    filepath = str(filepath)
    self.index = faiss.read_index(filepath + '.index')
    self.aux_info = pd.read_feather(filepath + '.aux_info_df')
    # load aux_key
    with open(filepath + '.aux_key', 'r') as f:
      self.aux_key = f.read()

  def rebuild_index(self, listing_df: pd.DataFrame, **filters):
    """
    rebuild index based on filters on the given listing_df

    Args:
    listing_df: pd.DataFrame, dataframe containing the latest listing detail info. This should be maintained by another system
    filters: dict, key-value pairs to filter the listing_df, as in **argv 
    """

    # construct query
    query_str = ' & '.join([f"{k} == '{v}'" if isinstance(v, str) else f"{k} == {v}" for k, v in filters.items()])    
    print(query_str)

    try:
      wanted_listingIds = set(listing_df.q(query_str).jumpId.values)
      unwanted_items = self.aux_info.q("~listing_id.isin(@wanted_listingIds)")[self.aux_key].values.tolist()
      self.remove(unwanted_items)
    except Exception as e:
      print(e)
      raise e


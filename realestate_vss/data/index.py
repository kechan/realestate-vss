from typing import List, Union, Optional

from pathlib import Path

import pandas as pd
import numpy as np
import faiss, copy, pickle, gc

class FaissIndex:
  
  def __init__(self, 
               embeddings: Optional[np.ndarray] = None, 
               aux_info: Optional[pd.DataFrame] = None, 
               aux_key: Optional[str] = None, 
               filepath: Optional[Path] = None,
               index_type: Optional[str] = 'FlatIP',
               **index_params):
    self.filepath = filepath
    self.index_type = index_type
    self.index_params = index_params
    self.quantizer_type = index_params.get('quantizer_type', 'FlatIP')

    if self.filepath is not None:
      self.load(self.filepath)  # would get its own saved quantizer_type
    else: 
      if embeddings is None or aux_info is None:
        raise ValueError('Both embeddings and aux_info must be provided if not loading from file.')
      
      if isinstance(aux_info, pd.DataFrame):
        if aux_key is None or aux_key not in aux_info.columns:
          raise ValueError('aux_key is required and must be in aux_info.columns')      
      
      self.embeddings = embeddings
      d = embeddings.shape[1]

      if index_type == 'FlatIP':
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
      elif index_type == 'HNSWFlat':
        M = index_params.get('M', 32)
        efConstruction = index_params.get('efConstruction', None)

        self.index = faiss.IndexHNSWFlat(d, M, faiss.METRIC_INNER_PRODUCT)   # TODO: deal with L2 version of all these later
        if efConstruction is None:
          self.index.hnsw.efConstruction = 40

      elif index_type == 'HNSWSQ':
        M = index_params.get('M', 16)
        
        efConstruction = index_params.get('efConstruction', None)
        nbits = index_params.get('nbits', 8)

        if nbits == 8:
          sq_quantizer = faiss.ScalarQuantizer.QT_8bit
        elif nbits == 16:
          sq_quantizer = faiss.ScalarQuantizer.QT_fp16
        # TODO: add more bit type
        else:
          raise ValueError(f'nbits {nbits} not supported')

        self.index = faiss.IndexHNSWSQ(d, sq_quantizer, M, faiss.METRIC_INNER_PRODUCT)  # TODO: deal with L2 version of all these later
        # training for the scalar quantizer
        self.index.train(embeddings)

        if efConstruction is None:
          self.index.hnsw.efConstruction = 40

      elif index_type == 'IVFFlat':
        nlist = index_params.get('nlist', 16384)
        M = index_params.get('M', 32)

        if self.quantizer_type == 'FlatIP':
          quantizer = faiss.IndexFlatIP(d)
        elif self.quantizer_type == 'HNSWFlat':
          quantizer = faiss.IndexHNSWFlat(d, M)
        else:
          raise ValueError(f'quantizer {self.quantizer_type} not supported')

        self.index = faiss.IndexIVFFlat(quantizer, d, nlist)
        self.index.cp.min_points_per_centroid = 5   # quiet warning
        if self.quantizer_type == 'HNSWFlat':
          self.index.quantizer_trains_alone = 2
          # self.index.quantizer.hnsw.efConstruction = 40

        self.index.train(embeddings)
        self.index.add(embeddings)

      elif index_type == 'IVFPQ':
        nlist = index_params.get('nlist', 16384)
        m = index_params.get('m', 16)
        nbits = index_params.get('nbits', 8)

        if self.quantizer_type == 'FlatIP':
          quantizer = faiss.IndexFlatIP(d)
        else:
          raise ValueError(f'quantizer {self.quantizer_type} not yet supported')

        self.index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits)
        self.index.train(embeddings)
        self.index.add(embeddings)
      else:
        raise ValueError(f'index_type {index_type} not supported')

      self.index.add(embeddings)

      # deal with aux_info
      self.aux_info = aux_info
      # just be paranoid and do a defrag here, we don't want non contiguous index number
      self.aux_info.defrag_index(inplace=True)
      self.aux_key = aux_key

      # sanity check
      assert embeddings.shape[0] == aux_info.shape[0], 'embeddings and aux_info must have the same number of rows'    

      # explit delete of self.embeddings to save memory, since we don't need it anymore
      del self.embeddings
      gc.collect()
    
    # set query time parameters
    if self.index_type == 'HNSWSQ':
      efSearch = self.index_params.get('efSearch', 64)
      self.index.hnsw.efSearch = efSearch
    elif self.index_type == 'IVFFlat':
      nprobe = self.index_params.get('nprobe', 16)
      self.index.nprobe = nprobe
      if self.quantizer_type == 'HNSWFlat':
        self.index.quantizer.hnsw.efSearch = 64
    elif self.index_type == 'IVFPQ':
      nprobe = self.index_params.get('nprobe', 16)
      self.index.nprobe = nprobe


  @classmethod
  def from_dataframe(cls, df: pd.DataFrame, aux_key: str, embedding_column: str = 'embedding'):
    if embedding_column not in df.columns or aux_key not in df.columns:
      raise ValueError(f'{embedding_column} and {aux_key} must be columns in the dataframe')
    
    embeddings = np.stack(df[embedding_column].values)
    aux_info = df.drop(columns=[embedding_column])  # dont want to store all the embeddings in aux info
    return cls(embeddings=embeddings, aux_info=aux_info, aux_key=aux_key)


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
    # save index_type and index_params
    with open(filepath + '.index_type', 'w') as f:
      f.write(self.index_type)
    with open(filepath + '.index_params', 'wb') as f:
      pickle.dump(self.index_params, f)
        
  def load(self, filepath: Path):
    # Load the FAISS index from disk
    filepath = str(filepath)
    self.index = faiss.read_index(filepath + '.index')
    self.aux_info = pd.read_feather(filepath + '.aux_info_df')
    # load aux_key
    with open(filepath + '.aux_key', 'r') as f:
      self.aux_key = f.read()
    # load index_type and index_params
    if Path(filepath + '.index_type').exists() and Path(filepath + '.index_params').exists():
      with open(filepath + '.index_type', 'r') as f:
        self.index_type = f.read()
      with open(filepath + '.index_params', 'rb') as f:
        self.index_params = pickle.load(f)
    else:
      self.index_type = 'FlatIP'   # the default
      self.index_params = {}

    self.quantizer_type = self.index_params.get('quantizer_type', 'FlatIP')

  def rebuild_index(self, listing_df: pd.DataFrame, case_sensitive=True, **filters):
    """
    rebuild index based on filters on the given listing_df

    Args:
    listing_df: pd.DataFrame, dataframe containing the latest listing detail info. This should be maintained by another system
    case_sensitive: bool, whether to perform case-sensitive or case-insensitive matching
    filters: dict, key-value pairs to filter the listing_df, as in **argv 
    """

    # construct query
    if case_sensitive:
      query_str = ' & '.join([f"{k} == '{v}'" if isinstance(v, str) else f"{k} == {v}" for k, v in filters.items()])    
    else:
      query_str = ' & '.join([f"{k}.str.lower() == '{v.lower()}'" if isinstance(v, str) else f"{k} == {v}" for k, v in filters.items()])
    # print(f'query_str: {query_str}')

    try:
      wanted_listingIds = set(listing_df.q(query_str).jumpId.values)
      unwanted_items = self.aux_info.q("~listing_id.isin(@wanted_listingIds)")[self.aux_key].values.tolist()
      self.remove(unwanted_items)
    except Exception as e:
      print(e)
      raise e
    
  def partition(self, listing_df: pd.DataFrame, attribute: str, index_name: str = None, output_dir: Path = None):
    """
    Partitions the FAISS index based on the unique values of a specified attribute in the listing DataFrame.

    Args:
        listing_df (pd.DataFrame): The DataFrame containing the listing details.
        attribute (str): The attribute to partition the index by. This should be a column in the listing DataFrame.
        index_name (str, optional): The base name of the index files. The partition value will be appended to this to create the file name. If not provided, the stem of the current index file is used.
        output_dir (Path, optional): The directory to save the partitioned index files in. If not provided, the parent directory of the current index file is used.

    Raises:
        ValueError: If output_dir is not provided and the index was not loaded from a file.
        ValueError: If index_name is not provided and the index was not loaded from a file.
        ValueError: If attribute is not a column in the listing DataFrame.

    Side Effects:
        Saves the partitioned index files to the output directory.

    E.g.
      faiss_image_index.partition(listing_df, attribute='provState', output_dir=Path.home()/'tmp')
    """
    if output_dir is None and self.filepath is not None:
      output_dir = self.filepath.parent   # save in the same dir as this index (if it was loaded from file)      
    elif output_dir is None:
      raise ValueError('output_dir is required if index was not loaded from file')
    
    if index_name is None and self.filepath is not None:
      index_name = self.filepath.stem
    elif index_name is None:
      raise ValueError('index_name is required if index was not loaded from file')
    
    # check attribute is a col of listing_df
    if attribute not in listing_df.columns:
      raise ValueError(f'{attribute} is not a column in the listing_df')
    
    case_sensitive = True
    
    # special handling for certain attributes
    if attribute in ['provState', 'prov_code', 'province']:
      values = set(val.upper() for val in listing_df[attribute].unique() if val is not None)
      case_sensitive = False
    
    values = listing_df[attribute].unique()
    for value in values:
      index_copy = copy.deepcopy(self)
      print(f'index size before partitioning: {index_copy.index.ntotal}')
      index_copy.rebuild_index(listing_df, case_sensitive=case_sensitive, **{attribute: value})
      print(f'index size after partitioning: {index_copy.index.ntotal}')
      index_copy.save(output_dir / f'{index_name}_{value}')

      print(f'saved to {output_dir / f"{index_name}_{value}"}')

  def create(self, index_type: str, **index_params) -> "FaissIndex":
    """
    Create a new index with the same data but of different type and parameters
    """
    reconstructed_embeddings = self.index.reconstruct_n(0, self.index.ntotal)
    print(f'reconstructed_embeddings.shape: {reconstructed_embeddings.shape}')

    new_index = FaissIndex(embeddings=reconstructed_embeddings, 
                           aux_info=self.aux_info, 
                           aux_key=self.aux_key,
                           index_type=index_type,
                           **index_params)
    
    return new_index

  def estimate_memory_usage(self) -> float:
    """
    Estimate the memory usage of the index in MB
    """
    writer = faiss.VectorIOWriter()
    faiss.write_index(self.index, writer)
    bytes_size = writer.data.size()    
    print(f"Approximate memory usage of the index: {bytes_size / 1024 / 1024:.2f} MB")
    return bytes_size / 1024 / 1024


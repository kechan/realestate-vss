from typing import Dict, List, Union, Tuple, Any
from enum import Enum
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

from PIL import Image, ImageDraw, ImageFont

from realestate_core.common.utils import save_to_pickle, load_from_pickle, load_from_pickle, join_df
from realestate_vision.common.utils import get_listingId_from_image_name

class SearchMode(Enum):
  VSS_RERANK_ONLY = 1
  VSS_ONLY = 2
  SOFT_MATCH_AND_VSS = 3

class ListingSearchEngine:
  def __init__(self, image_embeddings_df, text_embeddings_df, image_embedder, text_embedder):
    '''
    image_embeddings_df: pd.DataFrame with columns ['listing_id', 'image_name', 'embedding']
    text_embeddings_df: pd.DataFrame with columns jumpId, embedding, and structured data attributes
    '''

    self.image_embeddings_df = image_embeddings_df
    # self.images = image_embeddings_df.image_name.values.tolist()
    image_embeddings = np.stack(image_embeddings_df.embedding.values)

    self.text_embeddings_df = text_embeddings_df
    self.image_embedder = image_embedder
    self.text_embedder = text_embedder

    # partition by province
    self.provStates = ['ON', 'BC', 'AB', 'SK', 'NS', 'NB', 'QC', 'NL', 'PE', 'MB', 'YT']
    self.text_embeddings_df_by_prov = {}
    for provState in self.provStates:
      self.text_embeddings_df_by_prov[provState] = text_embeddings_df.q("provState == @provState").copy()

    # use FAISS to construct index and VSS query
    # image
    self.faiss_image_index = self._build_faiss_index(image_embeddings)

    # text, partition by province
    self.faiss_text_index = {}
    for provState in self.provStates:
      df = self.text_embeddings_df_by_prov[provState]
      text_embeddings = np.stack(df.embedding.values)
      self.faiss_text_index[provState] = self._build_faiss_index(text_embeddings)


  def text_search(self, mode: SearchMode, topk=50, return_df=True, lambda_val=None, alpha_val=None, **query) -> Union[pd.DataFrame, List[Dict]]:    
    '''
    query: Dict with search attributes. Key of 'phrase' will use VSS r.p.t. to listing remarks
    lambda_val and alpha_val are used for only soft match + vss mode
    '''
    def exact_conditional_match(**query) -> List[str]:
      locals().update(query)
      query_conditions = []
      for key, value in query.items():
        if key == 'phrase': continue     # phrase is not an attribute of listing (should VSS r.p.t. to remarks)

        if isinstance(value, tuple) and len(value) == 2:
          query_conditions.append(f"{value[0]} <= {key} <= {value[1]}")    # for like range '900000 <= price <= 1000000'
        else:
          query_conditions.append(f"{key} == @{key}")                      # for like 'city == @city'

      query_string = " & ".join(query_conditions)

      df = self.text_embeddings_df_by_prov[provState].q(query_string)
      conditional_matched_listingIds = df.jumpId.values.tolist()

      return conditional_matched_listingIds

    def vss_search(phrase, topk=50) -> Tuple[List[str], List[float]]:
      text_features = self.text_embedder.embed_from_texts([phrase], batch_size=1)
      scores, top_k_indices = self._query_faiss_index(self.faiss_text_index[provState], text_features, topk=topk)

      top_k_indices = top_k_indices[0, :]
      top_listingIds = self.text_embeddings_df_by_prov[provState].iloc[top_k_indices].jumpId.values.tolist()

      return top_listingIds, scores[0].tolist()

    provState = query['provState']
    phrase = query.get('phrase')

    if mode == SearchMode.VSS_RERANK_ONLY:
      matched_listingIds = exact_conditional_match(**query)
      if phrase is None:        
        scores = np.ones(len(matched_listingIds)).tolist()    # assign all 1s to scores
      else:
        
        top_listingIds, scores = vss_search(phrase, topk=self.text_embeddings_df_by_prov[provState].shape[0])

        # sort matched_listingIds by scores
        score_map = dict(zip(top_listingIds, scores))
        scored_cond_matches = []
        for listingId in matched_listingIds:
          if listingId in score_map:
            scored_cond_matches.append((listingId, score_map[listingId]))

        sorted_scored_matches = sorted(scored_cond_matches, key=lambda x: x[1], reverse=True)

        matched_listingIds = [x[0] for x in sorted_scored_matches]
        scores = [x[1] for x in sorted_scored_matches]

    elif mode == SearchMode.VSS_ONLY:
      assert phrase is not None, 'Must provide phrase for VSS_ONLY search'

      top_listingIds, scores = vss_search(phrase, topk=topk)
      matched_listingIds = top_listingIds

    elif mode == SearchMode.SOFT_MATCH_AND_VSS:
      assert lambda_val is not None and alpha_val is not None, 'Must provide lambda_val and alpha_val for SOFT_MATCH_AND_VSS search'
      if phrase is not None:
        top_listingsIds, vss_scores = vss_search(phrase, topk=self.text_embeddings_df_by_prov[provState].shape[0])
        vss_score_df = pd.DataFrame(data={'jumpId': top_listingsIds, 'vss_score': vss_scores})
      else:
        raise ValueError('Must provide phrase for SOFT_MATCH_AND_VSS search')

      listings_df = self.text_embeddings_df_by_prov[provState]
      soft_scores = self.soft_match_score(listings_df, query, alpha_val)

      # Merge soft scores with listings_df for the calculation
      soft_score_df = pd.DataFrame(data={'jumpId': listings_df['jumpId'], 'soft_score': soft_scores})
      listings_w_soft_score_df = listings_df.merge(soft_score_df, on='jumpId', how='left')

      # Merge with VSS scores
      combined_df = listings_w_soft_score_df.merge(vss_score_df, on='jumpId', how='left')
      combined_df['final_score'] = lambda_val * combined_df['soft_score'] + (1 - lambda_val) * combined_df['vss_score']

      # Get top k results
      top_results = combined_df.nlargest(topk, 'final_score')

      matched_listingIds = top_results.jumpId.values.tolist()
      scores = top_results.final_score.values.tolist()
    else:
      raise ValueError(f'Unknown search mode {mode}')

    result_df = join_df(
        pd.DataFrame({'jumpId': matched_listingIds, 'score': scores}), 
        self.text_embeddings_df_by_prov[provState], 
        left_on='jumpId', how='inner')
    
    if return_df:
      return result_df
    else:
      # remove listing_id and embedding columns from results_df before returning the List[Dict]
      result_df = result_df.drop(columns=['jumpId', 'embedding'])
      return result_df.to_dict('records')

  def image_search(self, image: Image, topk=5) -> Tuple[List[str], List[float]]:
    '''
    Use FAISS to search for similar images

    image: PIL.Image
    topk: number of results to return

    return: top image names and scores
    '''
    image_features = self.image_embedder.embed_from_single_image(image)

    scores, top_k_indices = self._query_faiss_index(self.faiss_image_index, image_features, topk=topk)

    top_k_indices = top_k_indices[0, :]
    top_image_names = self.image_embeddings_df.iloc[top_k_indices].image_name.values.tolist()

    return top_image_names, scores[0].tolist() #, image_features, top_k_indices

  def visualize_image_search_results(self, image_names: List[str], scores: List[float], photos_dir: Path = '.') -> Image:
    resize = 350
    
    images_list = []
    for image_name, score in zip(image_names, scores):
      listingId = get_listingId_from_image_name(image_name)

      if (photos_dir/image_name).exists():
        img = Image.open(photos_dir/image_name).resize((resize, resize))
      elif (photos_dir/listingId/image_name).exists():      
        img = Image.open(photos_dir/listingId/image_name).resize((resize, resize))
      else:
        raise ValueError(f'Cannot find image {image_name} in {photos_dir}')
      
      # Draw the score on the image
      draw = ImageDraw.Draw(img)
      font_size = 20
      try:
        font = ImageFont.truetype("/Library/Fonts/Arial.ttf", font_size)
      except OSError:
        font = ImageFont.load_default()
      text = f'Score: {score:.2f}'  # Formatting the score to 2 decimal places
      text_position = (10, 10)  # You can adjust this position
      draw.text(text_position, text, fill=(0, 0, 0), font=font)  # White text, you can change color and font

      images_list.append(img)

    margin = 5
    total_width = resize * len(images_list) + margin * (len(images_list) - 1)
    max_height = resize

    new_img = Image.new('RGB', (total_width, max_height))

    # Paste each image into the new image
    x_offset = 0
    for img in images_list:
      new_img.paste(img, (x_offset, 0))
      x_offset += img.width + margin

    return new_img




  def soft_match_score(self, listings_df: pd.DataFrame, query: Dict, alpha_val: float) -> pd.Series:
    soft_scores = pd.Series(np.ones(len(listings_df)), index=listings_df.index)

    for key, value in query.items():
      if key not in listings_df.columns or key == 'phrase':
        continue

      if isinstance(value, tuple) and len(value) == 2:
        # For range conditions, use between
        condition_met = listings_df[key].between(value[0], value[1])
      else:
        # For single-value conditions, use direct comparison
        condition_met = (listings_df[key] == value)

      # Update scores where condition is not met
      soft_scores *= np.where(condition_met, 1, alpha_val)

    return soft_scores

  
  def _build_faiss_index(self, embeddings):
    # Initialize a FAISS index
    # Here we use the L2 distance metric; FAISS also supports inner product (dot product) with faiss.METRIC_INNER_PRODUCT
    # index = faiss.IndexFlatL2(embeddings.shape[1])
    index = faiss.IndexFlatIP(embeddings.shape[1])

    # Add vectors to the index
    index.add(embeddings)

    return index

  def _query_faiss_index(self, index, query_vectors, topk=5):
    query_vectors = np.atleast_2d(query_vectors).astype('float32')
    
    # Ensure the shape is (n_queries, d)
    if query_vectors.shape[1] == 1 or len(query_vectors.shape) == 1:
      query_vectors = query_vectors.reshape(1, -1)

    # Perform the search
    similarity_scores, I = index.search(query_vectors, topk)
    
    return similarity_scores, I



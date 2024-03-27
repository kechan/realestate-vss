from typing import Dict, List, Union, Tuple, Any, Optional
from enum import Enum, auto
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict

from realestate_core.common.utils import save_to_pickle, load_from_pickle, load_from_pickle, join_df
from realestate_vision.common.utils import get_listingId_from_image_name

from ..data.index import FaissIndex

class ScoreAggregationMethod(Enum):
    MAX = auto()
    MEAN = auto()

class SearchMode(str, Enum):
  VSS_RERANK_ONLY = "VSS_RERANK_ONLY"
  VSS_ONLY = "VSS_ONLY"
  SOFT_MATCH_AND_VSS = "SOFT_MATCH_AND_VSS"

class ListingSearchEngine:
  def __init__(self, 
               image_embedder, 
               text_embedder, 
               image_embeddings_df=None, 
               text_embeddings_df=None, 
               partition_by_province=True,
               faiss_image_index: FaissIndex = None,
               faiss_text_index: FaissIndex = None,
               listing_df: pd.DataFrame = None,   # carries detail info about listings
               score_aggregation_method: ScoreAggregationMethod = ScoreAggregationMethod.MEAN
               ):
    '''
    image_embeddings_df: pd.DataFrame with columns ['listing_id', 'image_name', 'embedding']
    text_embeddings_df: pd.DataFrame with columns jumpId, embedding, and structured data attributes
    '''

    # params consistency check, this may change later
    if (faiss_image_index is not None or faiss_text_index is not None) and partition_by_province:
      raise ValueError('realestate_vss.data.index.FaissIndex does not support partition_by_province at the moment')
    
    # if image FAISS index is provided, ensure text FAISS index is also provided, and vice 
    if (faiss_image_index is not None and faiss_text_index is None) or (faiss_text_index is not None and faiss_image_index is None):
      raise ValueError('Both faiss_image_index and faiss_text_index must be provided if one is provided')
    
    # ensure if we provide FAISS indexes, then 
    # 1. DO NOT provide image_embeddings_df and text_embeddings_df, to avoid confusion
    # 2. DO provide listing_df, which carries detail info about listings
    if faiss_image_index is not None and faiss_text_index is not None:
      if image_embeddings_df is not None or text_embeddings_df is not None:
        raise ValueError('If FAISS indexes are provided, image_embeddings_df and text_embeddings_df must NOT be provided')
      if listing_df is None:
        raise ValueError('If FAISS indexes are provided, listing_df must be provided, to ensure compatibility')

    if image_embeddings_df is not None:
      self.image_embeddings_df = image_embeddings_df
      image_embeddings = np.stack(image_embeddings_df.embedding.values)

      self.text_embeddings_df = text_embeddings_df

    self.image_embedder = image_embedder
    self.text_embedder = text_embedder

    self.partition_by_province = partition_by_province

    if self.partition_by_province:
      # text partition by province for df
      self.provStates = ['ON', 'BC', 'AB', 'SK', 'NS', 'NB', 'QC', 'NL', 'PE', 'MB', 'YT']
      self.text_embeddings_df_by_prov = {}
      for provState in self.provStates:
        if len(text_embeddings_df.q("provState == @provState")) > 0:
          self.text_embeddings_df_by_prov[provState] = text_embeddings_df.q("provState == @provState").copy()
        else:
          self.text_embeddings_df_by_prov[provState] = pd.DataFrame(columns=text_embeddings_df.columns)


      # text, partition by province for index
      self.faiss_text_index = {}
      for provState in self.provStates:
        df = self.text_embeddings_df_by_prov[provState]
        if len(df) > 0:
          text_embeddings = np.stack(df.embedding.values)
          self.faiss_text_index[provState] = self._build_faiss_index(text_embeddings)
        else:
          self.faiss_text_index[provState] = None  # no faiss index for this province, this None may cause err elsewhere, debug later
    else:
      # For text, no partition
      if faiss_text_index is not None:
        assert isinstance(faiss_text_index, FaissIndex), 'faiss_text_index must be an instance of realestate_vss.data.index.FaissIndex'
        self.faiss_text_index = faiss_text_index
      else:
        text_embeddings = np.stack(text_embeddings_df.embedding.values)
        self.faiss_text_index = self._build_faiss_index(text_embeddings)

    # use FAISS to construct index and VSS query
    # For image
    if faiss_image_index is not None:
      assert isinstance(faiss_image_index, FaissIndex), 'faiss_image_index must be an instance of realestate_vss.data.index.FaissIndex'
      self.faiss_image_index = faiss_image_index
    else:
      self.faiss_image_index = self._build_faiss_index(image_embeddings)

    self.listing_df = listing_df
    self.score_aggregation_method = score_aggregation_method


  def get_listing(self, listingId: str) -> Dict[str, Any]:
    """
    Get the listing data for a given listingId
    """

    if isinstance(self.faiss_text_index, FaissIndex):
      listing = self.listing_df.q("jumpId == @listingId").copy()
      if len(listing) == 0:
        return {}
      listing['listing_id'] = listing['jumpId']
    else:
      listing = self.text_embeddings_df.q("jumpId == @listingId")
      if len(listing) == 0:   # search the image set
        listing = self.image_embeddings_df.q("listing_id == @listingId")
        if len(listing) == 0:
          return {}
    
    listing_data = listing.to_dict('records')[0]

    if 'embedding' in listing_data:
      listing_data.pop('embedding')

    if 'propertyFeatures' in listing_data:
      listing_data['propertyFeatures'] = listing_data['propertyFeatures'].tolist()

    return listing_data

  def get_imagenames(self, listingId: str) -> List[str]:
    if isinstance(self.faiss_image_index, FaissIndex):
      return self.faiss_image_index.aux_info.q("listing_id == @listingId").image_name.values.tolist()
    else:
      return self.image_embeddings_df.q("listing_id == @listingId").image_name.values.tolist()

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

      if self.partition_by_province:
        df = self.text_embeddings_df_by_prov[provState].q(query_string)
      else:
        df = self.text_embeddings_df.q(query_string)
      conditional_matched_listingIds = df.jumpId.values.tolist()

      return conditional_matched_listingIds

    def vss_search(phrase, topk=50) -> Tuple[List[str], List[float]]:
      text_features = self.text_embedder.embed_from_texts([phrase], batch_size=1)
      if isinstance(self.faiss_text_index, FaissIndex):
        scores, top_listingIds = self._query_faiss_index(self.faiss_text_index, text_features, topk=topk)
        return top_listingIds, scores
      else:
        if self.partition_by_province:
          scores, top_k_indices = self._query_faiss_index(self.faiss_text_index[provState], text_features, topk=topk)
        else:
          scores, top_k_indices = self._query_faiss_index(self.faiss_text_index, text_features, topk=topk)

        top_k_indices = top_k_indices[0, :]
        if self.partition_by_province:
          top_listingIds = self.text_embeddings_df_by_prov[provState].iloc[top_k_indices].jumpId.values.tolist()
        else:
          top_listingIds = self.text_embeddings_df.iloc[top_k_indices].jumpId.values.tolist()

        return top_listingIds, scores[0].tolist()

    # provState = query['provState']
    phrase = query.get('phrase')

    if mode == SearchMode.VSS_RERANK_ONLY:
      provState = query['provState']
      matched_listingIds = exact_conditional_match(**query)
      if phrase is None:        
        scores = np.ones(len(matched_listingIds)).tolist()    # assign all 1s to scores
      else:
        # TODO returning that many topk results is not efficient
        if self.partition_by_province:  
          top_listingIds, scores = vss_search(phrase, topk=self.text_embeddings_df_by_prov[provState].shape[0])
        else:
          top_listingIds, scores = vss_search(phrase, topk=self.text_embeddings_df.shape[0])

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
      provState = query['provState']
      if phrase is not None:
        if self.partition_by_province:
          top_listingsIds, vss_scores = vss_search(phrase, topk=self.text_embeddings_df_by_prov[provState].shape[0])
        else:
          top_listingsIds, vss_scores = vss_search(phrase, topk=self.text_embeddings_df.shape[0])
        vss_score_df = pd.DataFrame(data={'jumpId': top_listingsIds, 'vss_score': vss_scores})
      else:
        raise ValueError('Must provide phrase for SOFT_MATCH_AND_VSS search')

      if self.partition_by_province:
        listings_df = self.text_embeddings_df_by_prov[provState]
      else:
        listings_df = self.text_embeddings_df.q("provState == @provState")
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

    if isinstance(self.faiss_text_index, FaissIndex):
      result_df = pd.DataFrame({'listing_id': matched_listingIds, 'score': scores})
      # the new text FaissIndex is such that stuff they index can be remark chunks, so listing_id is not unique
      # we will need to group by listing_id and take the avg score 
      if self.score_aggregation_method == ScoreAggregationMethod.MEAN:
        result_df = result_df.groupby('listing_id').score.mean().reset_index()
      elif self.score_aggregation_method == ScoreAggregationMethod.MAX:
        result_df = result_df.groupby('listing_id').score.max().reset_index()
      else:
        raise ValueError(f'Unknown score aggregation method {self.score_aggregation_method}')

      result_df = join_df(
          result_df, 
          self.listing_df, 
          left_on='listing_id', right_on='jumpId', how='inner')
      
      # sort by score in descending order
      result_df.sort_values(by='score', ascending=False, inplace=True)

    else:
      if self.partition_by_province:
        result_df = join_df(
            pd.DataFrame({'jumpId': matched_listingIds, 'score': scores}), 
            self.text_embeddings_df_by_prov[provState], 
            left_on='jumpId', how='inner')
      else:
        result_df = join_df(
            pd.DataFrame({'jumpId': matched_listingIds, 'score': scores}), 
            self.text_embeddings_df, 
            left_on='jumpId', how='inner')
    
    if return_df:
      return result_df
    else:
      # remove listing_id and embedding columns from results_df before returning the List[Dict]
      if 'jumpId' in result_df.columns:
        result_df = result_df.drop(columns=['jumpId'])
      if 'embedding' in result_df.columns:
        result_df = result_df.drop(columns=['embedding'])

      result_dict = result_df.to_dict('records')
      # Convert all values to strings
      for item in result_dict:
        for key, value in item.items():
          if key != 'score':
            item[key] = str(value)

      return result_dict

  def image_search(self, image: Image.Image, topk=5, group_by_listingId=False) -> Tuple[List[str], List[float]]:
    '''
    Use FAISS to search for similar images

    image: PIL.Image
    topk: number of results to return

    return: top image names and scores
    '''
    
    image_features = self.image_embedder.embed_from_single_image(image)
    if isinstance(self.faiss_image_index, FaissIndex):
      scores, top_image_names = self._query_faiss_index(self.faiss_image_index, image_features, topk=topk)
      if group_by_listingId:
        listings = self._gen_listings_from_image_search(top_image_names, scores)
        # Sort the listings by average score in descending order
        listings = sorted(listings, key=lambda x: x['agg_score'], reverse=True)
        return listings

      return top_image_names, scores
    else:
      scores, top_k_indices = self._query_faiss_index(self.faiss_image_index, image_features, topk=topk)

      top_k_indices = top_k_indices[0, :]
      top_image_names = self.image_embeddings_df.iloc[top_k_indices].image_name.values.tolist()

      if group_by_listingId:
        raise NotImplementedError('group_by_listingId is not yet implemented for non-FaissIndex image index')

      return top_image_names, scores[0].tolist() #, image_features, top_k_indices
  
  def text_2_image_search(self, phrase: str, topk=5, group_by_listingId=False) -> Tuple[List[str], List[float]]:
    """
    Given a phrase, use VSS to search for images that match it conceptually.
    """
    query_embedding = self.text_embedder.embed_from_texts([phrase], batch_size=1)

    if isinstance(self.faiss_image_index, FaissIndex):
      scores, top_image_names = self._query_faiss_index(self.faiss_image_index, query_embedding, topk=topk)
      if group_by_listingId:
        listings = self._gen_listings_from_image_search(top_image_names, scores)
        # Sort the listings by agg score in descending order
        listings = sorted(listings, key=lambda x: x['agg_score'], reverse=True)
        return listings
      return top_image_names, scores
    else:
      scores, top_k_indices = self._query_faiss_index(self.faiss_image_index, query_embedding, topk=topk)
      top_k_indices = top_k_indices[0, :]
      top_image_names = self.image_embeddings_df.iloc[top_k_indices].image_name.values.tolist()
      if group_by_listingId:
        raise NotImplementedError('group_by_listingId is not yet implemented for non-FaissIndex image index')
      return top_image_names, scores[0].tolist()
  
  def text_2_image_text_search(self, phrase: str, topk=5) -> List[Dict[str, Union[str, float , List[str], str]]]:
    """
    Given text, search against both image vector index and text vector index and return the combined results.
    """

    # perform text search (use VSS for now)  # TODO: could change later
    text_results = self.text_search(SearchMode.VSS_ONLY, phrase=phrase, topk=topk, return_df=False)    
    text_results = [{**x, 'listingId': x['listing_id']} for x in text_results]   # add the key listingId

    # perform text to image search
    top_image_names, top_scores = self.text_2_image_search(phrase, topk=topk)

    listings_from_image_search = self._gen_listings_from_image_search(top_image_names, top_scores)

    # add in remarks wherever available for the listings from image search
    for listing in listings_from_image_search:
      listingId = listing['listingId']
      listing_info = self.get_listing(listingId)
      remarks = listing_info.get('remarks', '')
      listing['remarks'] = remarks
    
    # Normalize scores
    text_results = self.normalize_scores(text_results, 'score')
    listings_from_image_search = self.normalize_scores(listings_from_image_search, 'agg_score')

    # keep only listingId, score and remarks
    # text_results = [{k: x[k] for k in ['listingId', 'score', 'remarks']} for x in text_results]
      
    # Merge results
    combined_results = self.merge_results(text_results, listings_from_image_search)

    combined_results.sort(key=lambda x: x['agg_score'], reverse=True)

    return combined_results
  
  def normalize_scores(self, results: List[Dict], score_key: str) -> List[Dict]:
    scores = [result[score_key] for result in results]
    min_score = min(scores)
    max_score = max(scores)
    for result in results:
      result[score_key] = (result[score_key] - min_score) / (max_score - min_score)
    return results
  
  def merge_results(self, text_results: List[Dict], image_results: List[Dict]) -> List[Dict]:
    # Convert image_results to a dictionary for easy lookup
    listings_dict = {listing['listingId']: listing for listing in image_results}

    # Merge text_results into listings_dict
    for result in text_results:
      listingId = result['listingId']
      if listingId in listings_dict:
        # Add score to agg_score
        listings_dict[listingId]['agg_score'] += result['score']
      else:
        # Add new listing to listings_dict
        listings_dict[listingId] = {
            'listingId': listingId,
            'agg_score': result['score'],
            'remarks': result['remarks'],
            'image_names': []
        }

    # Convert listings_dict back to a list
    combined_results = list(listings_dict.values())

    return combined_results

  def image_2_text_search(self, image: Image.Image, topk=5, return_df=False):
    image_embedding = self.image_embedder.embed_from_single_image(image)
    if isinstance(self.faiss_text_index, FaissIndex):
      scores, top_listingIds = self._query_faiss_index(self.faiss_text_index, image_embedding, topk=topk)
    else:
      raise NotImplementedError('image_2_text_search is not implemented for non-FaissIndex text index')
    
    if isinstance(self.faiss_text_index, FaissIndex):
      result_df = pd.DataFrame({'listing_id': top_listingIds, 'score': scores})
      # the new text FaissIndex is such that stuff they index can be remark chunks, so listing_id is not unique
      # we will need to group by listing_id and take the avg score 
      if self.score_aggregation_method == ScoreAggregationMethod.MEAN:
        result_df = result_df.groupby('listing_id').score.mean().reset_index()
      elif self.score_aggregation_method == ScoreAggregationMethod.MAX:
        result_df = result_df.groupby('listing_id').score.max().reset_index()
      else:
        raise ValueError(f'Unknown score aggregation method {self.score_aggregation_method}')

      # join with listing_df to get details
      result_df = join_df(
          result_df, 
          self.listing_df, 
          left_on='listing_id', right_on='jumpId', how='inner')
      
      # sort by score in descending order
      result_df.sort_values(by='score', ascending=False, inplace=True)

    if return_df:
      return result_df
    else:
      # remove listing_id and embedding columns from results_df before returning the List[Dict]
      if 'jumpId' in result_df.columns:
        result_df = result_df.drop(columns=['jumpId'])
      if 'embedding' in result_df.columns:
        result_df = result_df.drop(columns=['embedding'])

      result_dict = result_df.to_dict('records')
      # Convert all values to strings
      for item in result_dict:
        for key, value in item.items():
          if key != 'score':
            item[key] = str(value)

      return result_dict

  def image_2_image_text_search(self, image: Image.Image, topk=5) -> List[Dict[str, Union[str, float , List[str], str]]]:
    image_results = self.image_search(image, topk=topk, group_by_listingId=True)

    text_results = self.image_2_text_search(image, topk=topk)
    text_results = [{**x, 'listingId': x['listing_id']} for x in text_results]   # add the key listingId

    # add in remarks whenever available for the image results
    for listing in image_results:
      listingId = listing['listingId']
      listing_info = self.get_listing(listingId)
      remarks = listing_info.get('remarks', '')
      listing['remarks'] = remarks

    # Normalize scores
    image_results = self.normalize_scores(image_results, 'agg_score')
    text_results = self.normalize_scores(text_results, 'score')

    # merge results
    combined_results = self.merge_results(text_results, image_results)

    combined_results.sort(key=lambda x: x['agg_score'], reverse=True)

    return combined_results


  def many_image_search(self, images: List[Image.Image], topk=5, group_by_listingId=False):
    all_top_image_names = []
    all_scores = []
    for image in images:
      image_features = self.image_embedder.embed_from_single_image(image)
      if isinstance(self.faiss_image_index, FaissIndex):
        scores, top_image_names = self._query_faiss_index(self.faiss_image_index, image_features, topk=topk)
        all_top_image_names.extend(top_image_names)
        all_scores.extend(scores)
      else:
        raise NotImplementedError('Non ..data.index.FaissIndex image index is not supported')

    if group_by_listingId:
      # Group by listingId after collecting all the results
      listings = self._gen_listings_from_image_search(all_top_image_names, all_scores)
      # Sort the listings by agg score in descending order
      listings = sorted(listings, key=lambda x: x['agg_score'], reverse=True)
      return listings
    else:
      # Pair up image names and scores, sort by score, and unzip back into separate lists
      pairs = sorted(zip(all_top_image_names, all_scores), key=lambda pair: pair[1], reverse=True)
      all_top_image_names, all_scores = zip(*pairs)
      # Aggregate scores for duplicate image names
      score_dict = defaultdict(list)
      for name, score in zip(all_top_image_names, all_scores):
        score_dict[name].append(score)
      # Compute max score for each image name
      max_scores = {name: max(scores) for name, scores in score_dict.items()}
      # Sort image names by max score in descending order
      sorted_image_names = sorted(max_scores, key=max_scores.get, reverse=True)
      return sorted_image_names, [max_scores[name] for name in sorted_image_names]

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
    """
    Calculates a soft match score for each listing in the DataFrame based on the provided query.
    
    The query is a dictionary where each key-value pair represents a condition that the listings should meet.
    If a listing does not meet a condition, its score is reduced by a factor of alpha_val.
    
    For range conditions (specified as a tuple or list of length 2), the between function is used.
    For single-value conditions, direct comparison is used.
    
    Returns a pandas Series with the final scores for each listing. The higher the score, the better the listing matches the query.
    """
    soft_scores = pd.Series(np.ones(len(listings_df)), index=listings_df.index)

    for key, value in query.items():
      if key not in listings_df.columns or key == 'phrase':
        continue

      if (isinstance(value, tuple) or isinstance(value, list)) and len(value) == 2:
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
    index = faiss.IndexFlatIP(embeddings.shape[1])   # use dot product

    # Add vectors to the index
    index.add(embeddings)

    return index

  def _query_faiss_index(self, index, query_vectors, topk=5):
    query_vectors = np.atleast_2d(query_vectors).astype('float32')
    
    # Ensure the shape is (n_queries, d)
    if query_vectors.shape[1] == 1 or len(query_vectors.shape) == 1:
      query_vectors = query_vectors.reshape(1, -1)

    # Perform the search (depends on if index is realestate_vss.data.index.FaissIndex or faiss.IndexFlat**)
    if isinstance(index, FaissIndex):
      top_matches, similarity_scores = index.search(query_vectors=query_vectors, topK=topk)
      return similarity_scores, top_matches   # top_matches are either image_names or listing_ids depending on the index
    else:
      similarity_scores, I = index.search(query_vectors, topk)
      return similarity_scores, I


  def _gen_listings_from_image_search(self, image_names: List[str], scores: List[float]) -> List[Dict[str, Union[str, float, List[str]]]]:
    """
    Given a list of image names and their scores, generate a list of listings with the aggregated score and image names for that listing

    # listingIds = [get_listingId_from_image_name(image_name) for image_name in image_names]
    # image names are of format {listingId}_{imageId}.jpg, we want to organize by listingId
    # such that we get a dict whose keys are listing_ids and values are list of image names
    # and another dict whose keys are listing_ids and values are list of corresponding scores
    """
    listingId_to_image_names = {}
    listingId_to_scores = {}
    for image_name, score in zip(image_names, scores):
      listingId = get_listingId_from_image_name(image_name)
      if listingId not in listingId_to_image_names:
        listingId_to_image_names[listingId] = []
        listingId_to_scores[listingId] = []

      listingId_to_image_names[listingId].append(image_name)
      listingId_to_scores[listingId].append(score)

    listings = []
    for listingId, image_names in listingId_to_scores.items():
      if self.score_aggregation_method == ScoreAggregationMethod.MAX:
        agg_score = np.max(np.array(listingId_to_scores[listingId]))
      elif self.score_aggregation_method == ScoreAggregationMethod.MEAN:
        agg_score = np.mean(np.array(listingId_to_scores[listingId]))
      else:
        raise ValueError(f'Unknown score aggregation method {self.score_aggregation_method}')

      image_names = [f"{listingId}/{image_name}" for image_name in listingId_to_image_names[listingId]]
      listings.append({
        'listingId': listingId,
        "agg_score": float(agg_score),
        "image_names": image_names,
      })

    return listings
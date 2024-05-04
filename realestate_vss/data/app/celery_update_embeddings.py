from typing import Optional, List, Any

import shutil, os, re
from pathlib import Path
import pandas as pd
import numpy as np

import redis
from celery import Celery
from celery.utils.log import get_task_logger

import realestate_core.common.class_extensions
from realestate_core.common.utils import join_df
from realestate_vss.data.index import FaissIndex
from realestate_vss.data.es_client import ESClient
from realestate_vss.data.redis_datastore import RedisDataStore

from dotenv import load_dotenv, find_dotenv

celery = Celery('update_embeddings_app', broker='pyamqp://guest@localhost//')
celery_logger = get_task_logger(__name__)

# celery -A celery_update_embeddings.celery worker --loglevel=info --logfile=celery_update_embeddings.log -P solo -Q update_embed_queue --detach --hostname=update_embeddings_worker@%h

# ps aux | grep 'celeryd' | awk '{print $2}' | xargs kill -9
# ps aux | grep 'celeryd' | grep update_embeddings_worker | awk '{print $2}' | xargs kill -9

model_name = 'ViT-L-14'
pretrained='laion2b_s32b_b82k'

def upsert_embeddings_to_faiss(embeddings_df: pd.DataFrame, 
                                listingIds: List[str], 
                                faiss_index: FaissIndex, 
                                operation: str, 
                                aux_key: str) -> None:
  """
  Function to add/update embeddings to faiss index.

  Parameters:
  embeddings_df (pd.DataFrame): DataFrame containing the embeddings.
  listingIds (List[str]): List of listing IDs to be processed, which could be a subset 
  faiss_index (FaissIndex): The faiss index object where embeddings are to be added/updated.
  operation (str): The operation to be performed - 'add' or 'update'.
  aux_key (str): The column name in the DataFrame that corresponds to the auxiliary key (image name or remark chunk ID). This
                can also be thought of as the primary key to auxilliary information.

  """
  items_to_process = list(embeddings_df.q("listing_id.isin(@listingIds)")[aux_key].values)
  processed_embeddings_df = embeddings_df.q(f"{aux_key}.isin(@items_to_process)")
  aux_info = processed_embeddings_df.drop(columns=['embedding'])
  embeddings = np.stack(processed_embeddings_df.embedding.values)

  if operation == 'add':
    faiss_index.add(embeddings=embeddings, aux_info=aux_info)
  elif operation == 'update':
    faiss_index.update(embeddings=embeddings, aux_info=aux_info)
  else:
    raise ValueError('operation must be either add or update')

def process_redis_docs(embeddings_df: pd.DataFrame, 
                       listingIds: List[str], 
                       datastore: RedisDataStore, 
                       aux_key: str,
                       listing_df: pd.DataFrame,
                       embedding_type: str = 'I') -> None:
  """
  Function to process embeddings and perform operations(add/delete) on Redis.

  Parameters:
  embeddings_df (pd.DataFrame): DataFrame containing the embeddings.
  listingIds (List[Any]): List of listing IDs to be processed.
  datastore (Any): The Redis datastore object where docs are to be added/deleted.
  aux_key (str): The column name in the DataFrame that corresponds to the auxiliary key (image name or remark chunk ID).
                 This can also be thought of as the primary key to auxilliary information.  
  listing_df (pd.DataFrame): DataFrame containing the detail listing data.                 

  """
  items_to_process = list(embeddings_df.q("listing_id.isin(@listingIds)")[aux_key].values)
  processed_embeddings_df = embeddings_df.q(f"{aux_key}.isin(@items_to_process)")
  _df = join_df(processed_embeddings_df, listing_df, left_on='listing_id', right_on='jumpId', how='left').drop(columns=['jumpId'])
  listing_jsons = _df.to_dict(orient='records')
  datastore.batch_insert(listing_jsons, embedding_type=embedding_type)

@celery.task
def update_embeddings(img_cache_folder: str):
  _ = load_dotenv(find_dotenv())
  use_redis = False
  if "REDIS_HOST" in os.environ and "REDIS_PORT" in os.environ:
    redis_host = os.environ["REDIS_HOST"]
    redis_port = int(os.environ["REDIS_PORT"])
    use_redis = True
    celery_logger.info(f'redis_host: {redis_host}, redis_port: {redis_port}')
  else:
    celery_logger.info('REDIS_HOST and REDIS_PORT not found in .env, not using Redis')
  
  if use_redis:
    redis_client = redis.Redis(host=redis_host, port=redis_port, db=0)
    datastore = RedisDataStore(client=redis_client, image_embedder=None, text_embedder=None)   # no query for update

  img_cache_folder = Path(img_cache_folder)
  working_folder = img_cache_folder/f'{model_name}_{pretrained}'

  job_ids = []

  for f in working_folder.lfre('.*_image_embeddings_df'):
      match = re.search(r'(.*)_image_embeddings_df', f.name)
      if match:
          job_id = match.group(1)
          job_ids.append((os.path.getctime(f), job_id))

  # Sort the list of tuples
  job_ids.sort()

  # Extract the job ids, discarding the creation times
  job_ids = [job_id for _, job_id in job_ids]

  # print(job_ids)

  image_embeddings_df, listing_df = [], []
  text_embeddings_df = pd.DataFrame()

  for job_id in job_ids:
    # print(job_id)
    _image_embeddings_df = pd.read_feather(working_folder/f'{job_id}_image_embeddings_df')
    try:
      _text_embeddings_df = pd.read_feather(working_folder/f'{job_id}_text_embeddings_df')
      _listing_df = pd.read_feather(working_folder/f'{job_id}_listing_df')
    except FileNotFoundError:
      celery_logger.info(f'File not found: {working_folder/f"{job_id}_text_embeddings_df"} or listing_df')
      continue

    # incl. records in _image_embeddings_df whose listing_id is in _text_embeddings_df
    wanted_listingIds = _text_embeddings_df.listing_id.unique() 
    _image_embeddings_df = _image_embeddings_df.q("listing_id.isin(@wanted_listingIds)")

    # Drop rows in existing dataframe that have the same listing_id as the new dataframe
    # text_embeddings_df = text_embeddings_df[~text_embeddings_df['listing_id'].isin(_text_embeddings_df['listing_id'])]
    if not text_embeddings_df.empty:
      drop_listingIds = _text_embeddings_df.listing_id.unique()
      drop_index = text_embeddings_df.q("listing_id.isin(@drop_listingIds)").index
      text_embeddings_df.drop(index=drop_index, inplace=True)

    text_embeddings_df = pd.concat([text_embeddings_df, _text_embeddings_df], ignore_index=True)
    
    image_embeddings_df.append(_image_embeddings_df)
    listing_df.append(_listing_df)

  image_embeddings_df = pd.concat(image_embeddings_df, ignore_index=True)  
  listing_df = pd.concat(listing_df, ignore_index=True)

  # dedup
  image_embeddings_df.drop_duplicates(subset=['image_name'], keep='last', inplace=True)
  listing_df.drop_duplicates(subset=['jumpId'], keep='last', inplace=True)

  # defrag (being paranoia, cos if this isnt so, it will have bad bugs in FAISS search downstream)
  image_embeddings_df.defrag_index(inplace=True)
  text_embeddings_df.defrag_index(inplace=True)
  listing_df.defrag_index(inplace=True)

  # cast to float, the json version tends to be string
  listing_df.lat = listing_df.lat.astype(float)
  listing_df.lng = listing_df.lng.astype(float)
  listing_df.price = listing_df.price.astype(float)
  listing_df.leasePrice = listing_df.leasePrice.astype(float)

  incoming_listingIds = set(image_embeddings_df.listing_id.unique())

  # relax this requirement for now.
  # assert set(image_embeddings_df.listing_id) == set(text_embeddings_df.listing_id), 'listing_id in image_embeddings_df and text_embeddings_df must be the same'

  if not (working_folder/'faiss_image_index.index').exists() and not (working_folder/'faiss_image_index.aux_info_df').exists():
    # create a new index for the first time
    image_aux_info = image_embeddings_df.drop(columns=['embedding'])
    faiss_image_index = FaissIndex(embeddings=np.stack(image_embeddings_df.embedding.values), 
                                  aux_info=image_aux_info, 
                                  aux_key='image_name')
          
    text_aux_info = text_embeddings_df.drop(columns=['embedding'])
    faiss_text_index = FaissIndex(embeddings=np.stack(text_embeddings_df.embedding.values), 
                                  aux_info=text_aux_info, 
                                  aux_key='listing_id')
    
  else:
    # load the existing index
    faiss_image_index = FaissIndex(filepath=working_folder/'faiss_image_index')
    faiss_text_index = FaissIndex(filepath=working_folder/'faiss_text_index')

    # simple sanity check
    assert set(faiss_text_index.aux_info.listing_id.values) == set(faiss_image_index.aux_info.listing_id.values), 'listing_id in existing faiss_text_index and faiss_image_index must be the same'

    existing_listingIds = set(faiss_text_index.aux_info.listing_id.unique())

    # Update the index (similar role to runstat in db)
    # This inovolves Deletions, Additions, and Updates

    # TODO: implement Deletions
    # 1) delete
    # there's currently no good way to detect deletion of listings 
    # one possible way is to periodically query the ES to see if the listing is still active, but this could be expensive.
    # We will do this less frequently, say once a day, so this will be in a separate task

    # 2) Additions
    # detect new listing 
    new_listingIds = incoming_listingIds - existing_listingIds
    if len(new_listingIds) > 0:
      # add new image embeddings
     
      # faiss_image_index.add(embeddings=embeddings, aux_info=aux_info)
      upsert_embeddings_to_faiss(embeddings_df=image_embeddings_df, 
                                  listingIds=new_listingIds, 
                                  faiss_index=faiss_image_index, 
                                  operation='add', 
                                  aux_key='image_name')

      # add new text embeddings
      upsert_embeddings_to_faiss(embeddings_df=text_embeddings_df, 
                                  listingIds=new_listingIds, 
                                  faiss_index=faiss_text_index, 
                                  operation='add', 
                                  aux_key='remark_chunk_id')

    else:
      celery_logger.info('No new listings to add to index.')

    # 3) Update existing listings
    updated_listingIds = incoming_listingIds.intersection(existing_listingIds)
   
    if len(updated_listingIds) > 0:
      # update image embeddings
      upsert_embeddings_to_faiss(embeddings_df=image_embeddings_df,
                                  listingIds=updated_listingIds,
                                  faiss_index=faiss_image_index,
                                  operation='update',
                                  aux_key='image_name')

      # update text embeddings
      upsert_embeddings_to_faiss(embeddings_df=text_embeddings_df,
                                  listingIds=updated_listingIds,
                                  faiss_index=faiss_text_index,
                                  operation='update',
                                  aux_key='remark_chunk_id')
    else:
      celery_logger.info('No updated listings to update in index.')

  # save the index to disk
  faiss_image_index.save(working_folder/'faiss_image_index')
  faiss_text_index.save(working_folder/'faiss_text_index')

  # add new listing_df to existing listing_df (we track the latest up2date listing info on disk)
  if (working_folder/'listing_df').exists():
    existing_listing_df = pd.read_feather(working_folder/'listing_df')
    listing_df = pd.concat([existing_listing_df, listing_df], ignore_index=True)
    listing_df.drop_duplicates(subset=['jumpId'], keep='last', inplace=True)
    listing_df.defrag_index(inplace=True)

  listing_df.to_feather(working_folder/'listing_df')

  # REDIS stuff
  if use_redis:
    existing_listingIds = set(datastore.get_unique_listing_ids())

    # New listings
    new_listingIds = incoming_listingIds - existing_listingIds

    celery_logger.info(f'Adding image embedding for {len(new_listingIds)} new listings to Redis')
    process_redis_docs(image_embeddings_df, new_listingIds, datastore, 'image_name', listing_df, embedding_type='I')
    celery_logger.info(f'Adding text embedding for {len(new_listingIds)} new listings to Redis')
    process_redis_docs(text_embeddings_df, new_listingIds, datastore, 'remark_chunk_id', listing_df, embedding_type='T')

    # Updated listings
    updated_listingIds = incoming_listingIds.intersection(existing_listingIds)
    datastore.delete_listings(listing_ids=updated_listingIds)  # delete and reinsert

    celery_logger.info(f'Updating image embeddings for {len(updated_listingIds)} listings in Redis')
    process_redis_docs(image_embeddings_df, updated_listingIds, datastore, 'image_name', listing_df, embedding_type='I')

    celery_logger.info(f'Updating text embeddings for {len(updated_listingIds)} listings in Redis')
    process_redis_docs(text_embeddings_df, updated_listingIds, datastore, 'remark_chunk_id', listing_df, embedding_type='T')

  # Clean up current job id specific files
  for job_id in job_ids:
    if (working_folder/f'{job_id}_image_embeddings_df').exists():
      os.remove(working_folder/f'{job_id}_image_embeddings_df')
    if (working_folder/f'{job_id}_text_embeddings_df').exists():
      os.remove(working_folder/f'{job_id}_text_embeddings_df')
    if (working_folder/f'{job_id}_listing_df').exists():
      os.remove(working_folder/f'{job_id}_listing_df')
  
@celery.task
def update_inactive_embeddings(img_cache_folder: str):
  """
  Remove inactive listings from faiss indexes and listing_df
  """

  # get env vars to set up ES client.
  _ = load_dotenv(find_dotenv())
  if "ES_HOST" in os.environ and "ES_PORT" and "ES_LISTING_INDEX_NAME" in os.environ:
    es_host = Path(os.environ["ES_HOST"])
    es_port = Path(os.environ["ES_PORT"])
    listing_index_name = Path(os.environ["ES_LISTING_INDEX_NAME"])
  else:
    raise ValueError("ES_HOST, ES_PORT and ES_LISTING_INDEX_NAME not found in .env")
  
  es = ESClient(host=es_host, port=es_port, index_name=listing_index_name)
  if not es.ping():
    celery_logger.info('ES is not accessible. Exiting...')
    return
  
  img_cache_folder = Path(img_cache_folder)
  working_folder = img_cache_folder/f'{model_name}_{pretrained}'

  if (working_folder/'faiss_image_index.index').exists() and (working_folder/'faiss_image_index.aux_info_df').exists() and (working_folder/'listing_df').exists():
    faiss_image_index = FaissIndex(filepath=working_folder/'faiss_image_index')
    faiss_text_index = FaissIndex(filepath=working_folder/'faiss_text_index')
    existing_listing_df = pd.read_feather(working_folder/'listing_df')

    listingIds = existing_listing_df.jumpId.unique().tolist()

    # use ES to get active the listings in listingIds
    listing_docs = es.get_active_listings(listingIds)

    active_listingIds = [doc['jumpId'] for doc in listing_docs]

    inactive_listingIds = set(listingIds) - set(active_listingIds)

    # remove items from faiss indexes and listing_df  
    if len(inactive_listingIds) > 0:
      celery_logger.info(f'Removing {len(inactive_listingIds)} inactive listings from indexes and listing_df')

      items_to_remove = faiss_image_index.aux_info.q("listing_id.isin(@inactive_listingIds)")[faiss_image_index.aux_key].values.tolist()
      faiss_image_index.remove(items_to_remove)

      items_to_remove = faiss_text_index.aux_info.q("listing_id.isin(@inactive_listingIds)")[faiss_text_index.aux_key].unique().tolist()
      faiss_text_index.remove(items_to_remove)

      existing_listing_df.drop(index=existing_listing_df.q("jumpId.isin(@inactive_listingIds)").index, inplace=True)
      existing_listing_df.defrag_index(inplace=True)

      # save the index to disk
      faiss_image_index.save(working_folder/'faiss_image_index')
      faiss_text_index.save(working_folder/'faiss_text_index')

      existing_listing_df.to_feather(working_folder/'listing_df')
    else:
      celery_logger.info('No inactive listings to remove from indexes and listing_df')

      


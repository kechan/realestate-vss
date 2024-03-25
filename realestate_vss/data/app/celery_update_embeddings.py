from typing import Optional

import shutil, os, re
from pathlib import Path
import pandas as pd
import numpy as np

from celery import Celery
from celery.utils.log import get_task_logger

import realestate_core.common.class_extensions
from realestate_core.common.utils import join_df
from realestate_vss.data.index import FaissIndex

from dotenv import load_dotenv, find_dotenv

celery = Celery('update_embeddings_app', broker='pyamqp://guest@localhost//')
celery_logger = get_task_logger(__name__)

# celery -A celery_update_embeddings.celery worker --loglevel=info --logfile=celery_update_embeddings.log -P solo -Q update_embed_queue --detach --hostname=update_embeddings_worker@%h

# ps aux | grep 'celeryd' | awk '{print $2}' | xargs kill -9
# ps aux | grep 'celeryd' | grep update_embeddings_worker | awk '{print $2}' | xargs kill -9

model_name = 'ViT-L-14'
pretrained='laion2b_s32b_b82k'

@celery.task
def update_embeddings(img_cache_folder: str):
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

  incoming_listingIds = set(image_embeddings_df.listing_id.unique())

  assert set(image_embeddings_df.listing_id) == set(text_embeddings_df.listing_id), 'listing_id in image_embeddings_df and text_embeddings_df must be the same'

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

    # 2) Additions
    # detect new listing 
    new_listingIds = incoming_listingIds - existing_listingIds
    if len(new_listingIds):
      # Continue here to add new listing images and remark chunks to the index

      # add new image embeddings
      images_to_add = list(image_embeddings_df.q("listing_id.isin(@new_listingIds)").image_name.values)
      new_image_embeddings_df = image_embeddings_df.q("image_name.isin(@images_to_add)")
      aux_info = new_image_embeddings_df.drop(columns=['embedding'])
      embeddings = np.stack(new_image_embeddings_df.embedding.values)

      faiss_image_index.add(embeddings=embeddings, aux_info=aux_info)

      # add new text embeddings
      text_chunks_to_add = list(text_embeddings_df.q("listing_id.isin(@new_listingIds)").remark_chunk_id.values)
      new_text_embeddings_df = text_embeddings_df.q("remark_chunk_id.isin(@text_chunks_to_add)")
      aux_info = new_text_embeddings_df.drop(columns=['embedding'])
      embeddings = np.stack(new_text_embeddings_df.embedding.values)

      faiss_text_index.add(embeddings=embeddings, aux_info=aux_info)
    else:
      celery_logger.info('No new listings to add to index.')

    # 3) Update existing listings
    updated_listingIds = incoming_listingIds.intersection(existing_listingIds)

    # update image embeddings
    images_to_update = list(image_embeddings_df.q("listing_id.isin(@updated_listingIds)").image_name.values)
    updated_image_embeddings_df = image_embeddings_df.q("image_name.isin(@images_to_update)")
    aux_info = updated_image_embeddings_df.drop(columns=['embedding'])
    embeddings = np.stack(updated_image_embeddings_df.embedding.values)

    faiss_image_index.update(embeddings=embeddings, aux_info=aux_info)

    # update text embeddings
    text_chunks_to_update = list(text_embeddings_df.q("listing_id.isin(@updated_listingIds)").remark_chunk_id.values)
    updated_text_embeddings_df = text_embeddings_df.q("remark_chunk_id.isin(@text_chunks_to_update)")
    aux_info = updated_text_embeddings_df.drop(columns=['embedding'])
    embeddings = np.stack(updated_text_embeddings_df.embedding.values)

    faiss_text_index.update(embeddings=embeddings, aux_info=aux_info)

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

  # Clean up current job id specific files
  for job_id in job_ids:
    if (working_folder/f'{job_id}_image_embeddings_df').exists():
      os.remove(working_folder/f'{job_id}_image_embeddings_df')
    if (working_folder/f'{job_id}_text_embeddings_df').exists():
      os.remove(working_folder/f'{job_id}_text_embeddings_df')
    if (working_folder/f'{job_id}_listing_df').exists():
      os.remove(working_folder/f'{job_id}_listing_df')
  





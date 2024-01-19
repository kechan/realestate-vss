from typing import List, Tuple, Dict, Union

import shutil
from pathlib import Path
import pandas as pd
import tensorflow as tf
from PIL import Image

from realestate_core.common.utils import load_from_pickle
from realestate_vision.common.utils import get_listingId_from_image_name

class ImageUnstacker:
  def __init__(self, archive_folder, staging_folder, computed_embeddings_df: List[pd.DataFrame]):
    self.archive_folder = archive_folder
    self.staging_folder = staging_folder

    # work out unique listing ids from each folder, the name of the subfolders are listing ids
    self.listings_in_staging = list(set([p.name for p in self.staging_folder.lfre('\d+')]))  # List[str]
    self.listings_in_archive = list(set([p.name for p in self.archive_folder.lfre('\d+')]))  # List[str]

    self.computed_embeddings_df = computed_embeddings_df
    # assert listingId, image_name, embedding are in datafraome
    for df in self.computed_embeddings_df:
      assert 'image_name' in df.columns
      assert 'embedding' in df.columns
      assert 'listing_id' in df.columns


  def unstack_images_in_staging(self) -> Dict[str, bool]:
    """
    1. Unstack images in staging folder.
    2. Remove the version in archive folder if listingId is in archive folder (photo updates)

    Returns:
    A message return some states
    """
    states = {}

    listing_folders = [self.staging_folder/listingId for listingId in self.listings_in_staging]
    if len(listing_folders) == 0: 
      return {'photo_updates_detected': False}
    
    self.unstack_deployment_image_dumps(listing_folders)

    # for listingIds that are previously already in self.listings_in_archive,
    # delete the version in archive folder
    listingIds_to_remove = [listingId for listingId in self.listings_in_staging if listingId in self.listings_in_archive]
    print(f'# of listings to remove: {len(listingIds_to_remove)}')
    if len(listingIds_to_remove) > 0:
      states['photo_updates_detected'] = True
      for listingId in listingIds_to_remove:
        listing_dir = self.archive_folder/listingId
        shutil.rmtree(listing_dir)

      # also remove the listing entries from the dataframe storing prior computed embeddings
      for df in self.computed_embeddings_df:
        df.drop(index=df.q("listing_id.isin(@listingIds_to_remove)").index, inplace=True)   # ensure df are saved to disk externally
        df.defrag_index(inplace=True)
    else:
      states['photo_updates_detected'] = False

    return states

  def unstack_deployment_image_dumps(self, listing_folders: List[Path]):
    """
    Listing image dumps are vertically stacked into 1 long and thin image. 
    This function unstacks the images and saves them as individual images.
    All metadata were stored as pickles and read here. 
    This method also output the listing remarks.

    Args:
      listing_folders: List of listing folders containing corresponding "stacked" image and metadata pickles.

    Returns:
      listing_remarks: A dictionary of listingId: remarks.                      
    """

    def unstack_images(filename, n_images):
      img_str = tf.io.read_file(str(filename))
      stacked_img = tf.image.decode_jpeg(img_str, channels=3)

      imgs = []
      for k in range(n_images):
        img = stacked_img[k*416:(k+1)*416, :]
        img = tf.io.encode_jpeg(img)
        imgs.append(img)

      return imgs
    
    # listing_remarks = {}

    for listing_dir in listing_folders:
      listingId = listing_dir.parts[-1]
      image_ids_pickle = listing_dir/'image_ids.pickle'
      if not image_ids_pickle.exists(): continue  # if image ids is missing, then unable to unstack images

      # check if images are already unstacked (excl. stacked_resized_*.jpg during checking), and if so, skip
      if len(listing_dir.lf('*.jpg')) > 1: continue

      image_ids = load_from_pickle(image_ids_pickle)  
      # photo_ids = load_from_pickle(listing_dir/'photo_ids.pickle')

      f = listing_dir.lf('stacked_resized_*.jpg')[0]    # only 1 stacked image expected
      imgs = unstack_images(f, len(image_ids))

      for img, id in zip(imgs, image_ids):
        tf.io.write_file(str(listing_dir/f'{listingId}_{id}.jpg'), img)

      # remarks_pickle = listing_dir/'remarks.pickle'
      # if remarks_pickle.exists():
      #   remarks = load_from_pickle(remarks_pickle)
      # else:
      #   remarks = ''
      # listing_remarks[listingId] = remarks





def read_and_preprocess_image(image_path, preprocess):
  """
  Read and preprocess image from the given path using preprocess.
  For OpenClip image embedding, preprocess can come from:
     model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
  """
  img = Image.open(image_path).convert("RGB")
  img = preprocess(img)
  
  return img
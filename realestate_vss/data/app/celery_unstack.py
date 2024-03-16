from typing import Optional

from celery import Celery
from celery.utils.log import get_task_logger

from realestate_core.common.utils import save_to_pickle
from pathlib import Path
from PIL import Image

import realestate_core.common.class_extensions
from realestate_core.common.utils import load_from_pickle

# celery -A celery_unstack.celery worker --loglevel=info --logfile=celery_unstack.log --detach
# celery -A celery_unstack.celery worker --loglevel=info --logfile=celery_unstack.log -Q unstack_queue --detach --hostname=unstack_worker@%h

# ps aux | grep 'celeryd' | awk '{print $2}' | xargs kill -9

# Initialize Celery
celery = Celery('unstack_app', broker='pyamqp://guest@localhost//')
celery_logger = get_task_logger(__name__)

@celery.task
def unstack(
  listing_folder: str,
  comma_sep_image_id: Optional[str], 
  comma_sep_photo_id: Optional[str], 
  comma_sep_aspect_ratio: Optional[str], 
  remarks: Optional[str],
  resize_height: Optional[int] = None,
  resize_width: Optional[int] = None
):
  celery_logger.info(f"Unstacking images in {listing_folder}")
  # The file has already been saved, so we just need to handle the metadata here
  listing_folder = Path(listing_folder)

  # Metadata handling and saving to pickle

  image_ids = comma_sep_image_id.split(',') if comma_sep_image_id else []
  # save_to_pickle(image_ids, listing_folder / 'image_ids.pickle')
    
  # photo_ids = comma_sep_photo_id.split(',') if comma_sep_photo_id else []
  # save_to_pickle(photo_ids, listing_folder / 'photo_ids.pickle')

  aspect_ratios = comma_sep_aspect_ratio.split(',') if comma_sep_aspect_ratio else []
  save_to_pickle(aspect_ratios, listing_folder / 'aspect_ratios.pickle')

  if remarks:
    save_to_pickle(remarks, listing_folder / 'remarks.pickle')

  # unstack(listing_folder=listing_folder)
  def unstack_images(filename, n_images):
    with Image.open(filename) as img:
      width, height = img.size
      slice_height = height // n_images
      
      imgs = []
      for k in range(n_images):
        cropped_img = img.crop((0, k*slice_height, width, (k+1)*slice_height))
        if resize_height is not None and resize_width is not None:
          cropped_img = cropped_img.resize((resize_width, resize_height), Image.BICUBIC)
        imgs.append(cropped_img)
      return imgs
    
  def save_img(filename, img):
    img.save(filename)

  listingId = listing_folder.parts[-1]
  if len(image_ids) == 0: return  # if image ids is missing, then unable to unstack images

  # check if images are already unstacked (excl. stacked_resized_*.jpg during checking), and if so, skip
  if len(listing_folder.lf('*.jpg')) > 1: return

  stacked_imgs = listing_folder.lf('stacked_resized_*.jpg')

  if len(stacked_imgs) > 0:
    # only 1 stacked image expected
    imgs = unstack_images(stacked_imgs[0], len(image_ids))

    for img, id in zip(imgs, image_ids):
      save_img(listing_folder/f'{listingId}_{id}.jpg', img)
  else:
    celery_logger.error(f"No stacked image found for {listingId}")
  

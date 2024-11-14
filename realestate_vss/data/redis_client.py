from typing import Optional, Dict
import redis 

from redis.commands.json.path import Path as RedisPath

class RedisClient:
  def __init__(self, host='localhost', port=6379, db=0, doc_prefix=None):
    self.client = redis.Redis(host=host, port=port, db=db)
    self.doc_prefix = doc_prefix

  def get(self, obj_id: Optional[str] = None):
    if obj_id is None:
      redis_keys = self.client.keys(f"{self.doc_prefix}:*")

      return [self.client.json().get(redis_key) for redis_key in redis_keys]
    else:
      redis_key = f"{self.doc_prefix}:{obj_id}"
      data = self.client.json().get(redis_key)

      return data

  def insert(self, key: str, json: Dict):
    """
    Perform necessary preprocessing on listing_doc and insert a listing into Redis.
    """
    redis_key = f"{self.doc_prefix}:{key}"
    try:
      self.client.json().set(redis_key, RedisPath.root_path(), json)
      print(f"{redis_key} inserted into Redis.")

      # self.client.save()   # TODO: does this need optimization, maybe we don't need to save every time?
    except Exception as e:
      print(f"Error inserting data into Redis: {e}")
      


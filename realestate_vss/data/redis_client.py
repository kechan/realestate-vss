from typing import Optional
import redis 

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

      


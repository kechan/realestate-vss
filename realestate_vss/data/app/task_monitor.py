from typing import Dict
import redis
import json
import os
from datetime import datetime
from dotenv import load_dotenv
from realestate_vss.utils.email import send_email_alert

class TaskMonitor:
  def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379, db: int = 1):
    self.redis = redis.Redis(host=redis_host, port=redis_port, db=1)

  def get_task_info(self, task_id: str) -> dict:
    data = self.redis.get(task_id)
    # print(f'data: {data}')
    if data:
      return json.loads(data)
    return {}

  def get_task_duration_hours(self, key: bytes) -> float:
    # Default TTL is 604800 seconds (7 days)
    ttl = self.redis.ttl(key)
    if ttl > 0:
        # Age = default TTL - current TTL
        age_in_seconds = 604800 - ttl
        return age_in_seconds / 3600
    return 0
  
  def _detect_task_type(self, task_info: Dict) -> str:
    # Check hostname in STARTED state
    if isinstance(task_info.get('result'), dict):
      hostname = task_info['result'].get('hostname', '')
      if 'delete_inactive_worker' in hostname:
        return 'delete_inactive'
      if 'embed_index_worker' in hostname:
        return 'embed_and_index'
      
      # Check stats in result for SUCCESS state
      stats = task_info['result'].get('stats', {})
      if 'image_embeddings_inserted' in stats:
        return 'embed_and_index'
      elif 'total_listings_deleted' in stats:
        return 'delete_inactive'
        
    return 'unknown'

  def alert_long_running(self, task_id: str, task_info: Dict, duration_hours: float) -> None:
    task_type = self._detect_task_type(task_info)

    subject = f"VSS Task Duration Warning - {task_type}"
    
    result_info = task_info.get('result', {})
    result_html = "".join([f"<li>{k}: {v}</li>" for k, v in result_info.items()])
    
    html_content = f"""
    <html><body>
      <h2>VSS Task Running Too Long</h2>
      <ul>
        <li>Task ID: {task_id}</li>
        <li>Task Type: {task_type}</li>
        <li>Running for: {duration_hours:.1f} hours</li>
        <li>Max allowed: 2.0 hours</li>
        <li>Status: {task_info.get('status', 'unknown')}</li>
      </ul>
      <h3>Task Result Details:</h3>
      <ul>
        {result_html}
      </ul>
    </body></html>
    """
    
    load_dotenv()
    send_email_alert(
      subject=subject,
      html_content=html_content,
      sender_email=os.getenv('VSS_SENDER_EMAIL'),
      receiver_emails=os.getenv('VSS_RECEIVER_EMAILS', '').split(','),
      password=os.getenv('VSS_EMAIL_PASSWORD')
    )

  def check_tasks(self):
    task_keys = self.redis.keys('celery-task-meta-*')
    
    for key in task_keys:
      # print(key)
      task_info = self.get_task_info(key)
      
      if task_info.get('status') in ['STARTED', 'PENDING']:
        duration = self.get_task_duration_hours(key)
        if duration > 2.0:
          self.alert_long_running(key, task_info, duration)
          

def main():
  monitor = TaskMonitor(
    redis_host=os.getenv('CELERY_BACKEND_REDIS_HOST_IP', 'localhost'),
    redis_port=int(os.getenv('CELERY_BACKEND_REDIS_PORT', 6379)),
    db=int(os.getenv('CELERY_BACKEND_REDIS_DB_INDEX', 1))
  )
  monitor.check_tasks()

if __name__ == "__main__":
  main()
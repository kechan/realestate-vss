from typing import List, Tuple
from pathlib import Path
import requests
import threading
import multiprocessing as mp
from multiprocessing import Queue
import time
import json
import random
from collections import namedtuple

# Configurable constants
NUM_REQUESTS_PER_BATCH = 20  # Number of requests per batch
TEST_DURATION_SECONDS = 60  # Total duration of the test in seconds

# Global flag to switch between threading and multiprocessing
use_thread = False  # Set to False to use processes instead

# Define a named tuple for request results
RequestResult = namedtuple('RequestResult', ['latency', 'status', 'result_size'])

# Endpoint URL
URL = 'http://localhost:8002/search'
image_dir = Path('data')

# Prepare a list of image file paths
images = [image_dir / img_name for img_name in [
  'bathroom-with-fireplace.jpg', 'nice_kitchen.jpeg', 'ugly_kitchen.jpg', 'cn_tower.jpg', 'niagara_fall.jpeg',
  'cn_tower_2.jpg', 'cn_tower_3.jpeg', 'ev_charger_port_perry.jpeg', 'resized_cn_tower.jpg', '13280246_7.jpg',
  'tesla_in_garage.jpg', 'infinity_pool.jpg', 'IMG_0846.jpeg', 'mls202324861.jpg', '18489643_28.jpg',
  '109_Bottomley_Ave_N_Saskatoon_SK.jpg', 'testing.jpg', 'cat.jpg', '1405397457296.jpeg', '1405375881692.jpeg'
]]

def generate_random_phrase():
  phrases = [
    'View of CN Tower', 'View of Niagara Fall', 'Victoria Style House', 'Cape Cod Style House',
    'Treehouse in the backyard', 'Gourmet kitchen with stainless steel appliance', 'Bathroom with a fireplace',
    'Traditional kitchen with detailed woodwork, classic finishes, and warmer color palettes',
    'Kitchen with premium brands like Viking, Sub-Zero, or Thermador',
    'Open concept kitchen that flow into the living space'
  ]
  return random.choice(phrases)

def send_request(image_path: Path, phrase: str, index: int = None, result_queue: Queue = None):
  """Unified request sender that works for both threading and multiprocessing"""
  start_time = time.time()
  files = {}
  f = None
  try:
    session = requests if use_thread else requests.Session()
    
    f = image_path.open('rb')
    files['file'] = (image_path.name, f, 'image/jpeg')
    query_body = {'phrase': phrase}
    files['query_body'] = (None, json.dumps(query_body), 'application/json')
    
    response = session.post(URL, files=files)
    latency = time.time() - start_time
    
    result_size = len(response.json()) if response.status_code == 200 else 0
    result = RequestResult(latency, response.status_code, result_size)
    
    if not use_thread:
      result_queue.put((index, image_path.name, result))
      if not isinstance(session, type(requests)):
        session.close()
    return (index, image_path.name, result)
    
  except Exception as e:
    print(f"Error during request: {e}")
    latency = time.time() - start_time
    result = RequestResult(latency, None, 0)
    if not use_thread:
      result_queue.put((index, image_path.name, result))
    return (index, image_path.name, result)
  finally:
    if f:
      f.close()

def launch_requests():
  if use_thread:
    return launch_threaded_requests()
  else:
    return launch_process_requests()

def launch_threaded_requests():
  """Thread-based implementation"""
  results = []
  threads = []

  selected_images = random.sample(images, NUM_REQUESTS_PER_BATCH)
  selected_phrases = [generate_random_phrase() for _ in range(NUM_REQUESTS_PER_BATCH)]

  def target(image_path, phrase, index):
    result = send_request(image_path, phrase)
    results.append(result)  # Store (index, image_name, result)

  for i in range(NUM_REQUESTS_PER_BATCH):
    t = threading.Thread(target=target, args=(selected_images[i], selected_phrases[i], i))
    threads.append(t)
    t.start()

  for t in threads:
    t.join()

  return results

def launch_process_requests():
  """Process-based implementation"""
  result_queue = mp.Queue()
  processes = []
  
  selected_images = random.sample(images, NUM_REQUESTS_PER_BATCH)
  selected_phrases = [generate_random_phrase() for _ in range(NUM_REQUESTS_PER_BATCH)]
  
  for i in range(NUM_REQUESTS_PER_BATCH):
    p = mp.Process(
      target=send_request,
      args=(selected_images[i], selected_phrases[i], i, result_queue)
    )
    processes.append(p)
    p.start()
  
  results = []
  for _ in range(NUM_REQUESTS_PER_BATCH):
    results.append(result_queue.get())
  
  results.sort(key=lambda x: x[0])
  return results

def main():
  total_results = []
  total_requests = 0
  total_errors = 0
  error_images = []

  print(f"\nStarting performance test using {'threads' if use_thread else 'processes'}...")
  print("=" * 80)

  for i in range(TEST_DURATION_SECONDS):  # Run for TEST_DURATION_SECONDS
    batch_start_time = time.time()
    batch_results = launch_requests()
    batch_elapsed = time.time() - batch_start_time
    total_results.extend(batch_results)
    total_requests += len(batch_results)
    errors_in_batch = sum(1 for _, _, result in batch_results if result.status != 200)
    total_errors += errors_in_batch

    # Print batch results
    print(f'\nBatch {i+1}:')
    print("-" * 40)
    for index, image_name, result in batch_results:
      status_display = result.status if result.status is not None else 'Error'
      print(f'  {"Thread" if use_thread else "Process"} {index+1}: '
            f'Latency = {result.latency*1000:.2f} ms, '
            f'Status = {status_display}, '
            f'Results = {result.result_size}, '
            f'Image = {image_name}')
      
      if result.status != 200:
        error_images.append(image_name)

    print(f'Batch {i+1} completed in {batch_elapsed:.2f} seconds with {errors_in_batch} errors.')

    # Sleep to maintain 1-second intervals
    sleep_time = max(0, 1 - batch_elapsed)
    time.sleep(sleep_time)

  # Print final statistics
  print("\nFinal Results")
  print("=" * 80)
  successful_results = [result for _, _, result in total_results if result.status == 200]
  
  if successful_results:
    successful_latencies = [result.latency for result in successful_results]
    successful_result_sizes = [result.result_size for result in successful_results]
    
    avg_latency = sum(successful_latencies) / len(successful_latencies)
    max_latency = max(successful_latencies)
    min_latency = min(successful_latencies)
    
    avg_result_size = sum(successful_result_sizes) / len(successful_result_sizes)
    max_result_size = max(successful_result_sizes)
    min_result_size = min(successful_result_sizes)
    
    print(f'Using: {"Threads" if use_thread else "Processes"}')
    print(f'Total requests: {total_requests}')
    print(f'Successful requests: {len(successful_results)}')
    print(f'Failed requests: {total_errors}')
    
    if error_images:
      print(f'\nImages responsible for errors:')
      for img in set(error_images):  # Print unique error-causing images
        print(f'  - {img}')
    
    print(f'\nLatency Statistics:')
    print(f'  Average: {avg_latency*1000:.2f} ms')
    print(f'  Min: {min_latency*1000:.2f} ms')
    print(f'  Max: {max_latency*1000:.2f} ms')
    print(f'\nResult Set Statistics:')
    print(f'  Average size: {avg_result_size:.1f}')
    print(f'  Min size: {min_result_size}')
    print(f'  Max size: {max_result_size}')
  else:
    print('All requests failed.')

if __name__ == '__main__':
  mp.freeze_support()
  main()


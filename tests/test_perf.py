import time
import requests
import random
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
import multiprocessing as mp

# Configurable constants
NUM_REQUESTS_PER_BATCH = 10      # Number of requests per batch (sent every second)
TEST_DURATION_SECONDS = 60       # Duration (in seconds) for scheduling new batches
URL = 'http://localhost:8002/search'
image_dir = Path('data')

# Global flag: set to True for threads, False for processes.
use_thread = True  # Change to False to use ProcessPoolExecutor

# Prepare a list of image file paths
images = [image_dir / img_name for img_name in [
    'bathroom-with-fireplace.jpg', 'nice_kitchen.jpeg', 'ugly_kitchen.jpg', 'cn_tower.jpg', 'niagara_fall.jpeg',
    'cn_tower_2.jpg', 'cn_tower_3.jpeg', 'ev_charger_port_perry.jpeg', 'resized_cn_tower.jpg', '13280246_7.jpg',
    'tesla_in_garage.jpg', 'infinity_pool.jpg', 
    #'IMG_0846.jpeg', 
    'mls202324861.jpg', '18489643_28.jpg',
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

def send_request(image_path: Path, phrase: str, batch_num: int, local_index: int):
    """
    Sends a POST request to the endpoint with an image and phrase.
    Returns a tuple:
       (batch_num, local_index, image_name, latency, status, result_size)
    """
    start_time = time.time()
    try:
        with image_path.open('rb') as f:
            files = {
                'file': (image_path.name, f, 'image/jpeg'),
                'query_body': (None, json.dumps({'phrase': phrase}), 'application/json')
            }
            response = requests.post(URL, files=files)
        latency = time.time() - start_time
        result_size = len(response.json()) if response.status_code == 200 else 0
        return (batch_num, local_index, image_path.name, latency, response.status_code, result_size)
    except Exception as e:
        latency = time.time() - start_time
        print(f"Error during request for batch {batch_num} index {local_index} ({image_path.name}): {e}")
        return (batch_num, local_index, image_path.name, latency, None, 0)

# Global structures for storing intermediate results and aggregating all results.
batch_results = {}
all_results = []
batch_lock = threading.Lock()

def future_callback(future):
    """
    Callback function that is called when each request completes.
    It appends the result to the global batch_results and all_results.
    When a batch is complete, it prints the intermediate results.
    """
    result = future.result()  # (batch_num, local_index, image_name, latency, status, result_size)
    batch_num, local_index, image_name, latency, status, result_size = result
    with batch_lock:
        all_results.append(result)
        if batch_num not in batch_results:
            batch_results[batch_num] = []
        batch_results[batch_num].append(result)
        # When all requests for this batch have completed, print intermediate results.
        if len(batch_results[batch_num]) == NUM_REQUESTS_PER_BATCH:
            # Sort by local index for ordered printing.
            batch_results[batch_num].sort(key=lambda x: x[1])
            print(f"\nBatch {batch_num+1}:")
            print("-" * 40)
            for res in batch_results[batch_num]:
                _, idx, img_name, lat, st, size = res
                status_display = st if st is not None else "Error"
                print(f"  {'Thread' if use_thread else 'Process'} {idx+1}: Latency = {lat*1000:.2f} ms, "
                      f"Status = {status_display}, Results = {size}, Image = {img_name}")
            print(f"Batch {batch_num+1} complete.\n")
            # Optionally remove the batch from the dictionary.
            del batch_results[batch_num]

def print_final_report(results):
    """
    Generates and prints the final aggregated report.
    """
    total_requests = len(results)
    total_errors = sum(1 for r in results if r[4] != 200)
    successful_results = [r for r in results if r[4] == 200]

    print("\nFinal Results")
    print("=" * 80)
    print(f"Total requests: {total_requests}")
    print(f"Successful requests: {len(successful_results)}")
    print(f"Failed requests: {total_errors}\n")

    if successful_results:
        latencies = [r[3] for r in successful_results]
        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)
        min_latency = min(latencies)
        sizes = [r[5] for r in successful_results]
        avg_size = sum(sizes) / len(sizes)
        max_size = max(sizes)
        min_size = min(sizes)

        print("Latency Statistics (ms):")
        print(f"  Average: {avg_latency*1000:.2f} ms")
        print(f"  Min: {min_latency*1000:.2f} ms")
        print(f"  Max: {max_latency*1000:.2f} ms\n")
        print("Result Size Statistics:")
        print(f"  Average: {avg_size:.1f}")
        print(f"  Min: {min_size}")
        print(f"  Max: {max_size}")
    else:
        print("All requests failed.")

def main():
    request_counter = 0
    # Choose the executor based on the concurrency model.
    executor_class = ThreadPoolExecutor if use_thread else ProcessPoolExecutor
    max_workers = NUM_REQUESTS_PER_BATCH * 2  # Adjust as needed

    print(f"Starting performance test using {'threads' if use_thread else 'processes'}...")
    print("=" * 80)

    with executor_class(max_workers=max_workers) as executor:
        # Schedule a new batch every second.
        for batch in range(TEST_DURATION_SECONDS):
            batch_start_time = time.time()
            for i in range(NUM_REQUESTS_PER_BATCH):
                selected_image = random.choice(images)
                phrase = generate_random_phrase()
                future = executor.submit(send_request, selected_image, phrase, batch, i)
                future.add_done_callback(future_callback)
            elapsed = time.time() - batch_start_time
            if elapsed < 1.0:
                time.sleep(1.0 - elapsed)
        # Wait for all submitted requests to finish.
        executor.shutdown(wait=True)

    # Print final aggregated report.
    with batch_lock:
        results_copy = list(all_results)
    print_final_report(results_copy)

if __name__ == '__main__':
    mp.freeze_support()  # Required for Windows if using processes
    main()


from locust import HttpUser, task, TaskSet, between, events
from pathlib import Path
import random
import json

# How to run:
# locust -f test_perf_with_locust.py --host=http://localhost:8002 --headless \
#     --users 5 \
#     --spawn-rate 5 \
#     --run-time 1m \
#     --only-summary

class SearchAPIUser(HttpUser):
    wait_time = between(0, 0)
    
    def on_start(self):
        self.image_dir = Path("data")
        self.images = [
            'bathroom-with-fireplace.jpg', 'nice_kitchen.jpeg', 'ugly_kitchen.jpg', 
            'cn_tower.jpg', 'niagara_fall.jpeg', 'cn_tower_2.jpg', 'cn_tower_3.jpeg', 
            'ev_charger_port_perry.jpeg', 'resized_cn_tower.jpg', '13280246_7.jpg'
        ]
        self.phrases = [
            'View of CN Tower', 'View of Niagara Fall', 'Victoria Style House', 
            'Cape Cod Style House', 'Treehouse in the backyard', 
            'Gourmet kitchen with stainless steel appliance', 'Bathroom with a fireplace',
            'Traditional kitchen with detailed woodwork, classic finishes, and warmer color palettes',
            'Kitchen with premium brands like Viking, Sub-Zero, or Thermador',
            'Open concept kitchen that flow into the living space'
        ]

    @task
    def combined_search(self):
        # Randomly select image and phrase
        image_path = self.image_dir / random.choice(self.images)
        phrase = random.choice(self.phrases)
        
        # Format exactly as your original script
        try:
            with open(image_path, "rb") as image:
                files = {
                    "file": image,  # FastAPI expects this exact field name
                    "query_body": (None, json.dumps({"phrase": phrase}))  # Form field, not file
                }
                with self.client.post("/search", 
                                    files=files, 
                                    catch_response=True) as response:
                    if response.status_code == 200:
                        results = response.json()
                        # Verify we got results
                        if len(results) > 0:
                            response.success()
                        else:
                            response.failure("No results returned")
                    else:
                        response.failure(f"Failed with status {response.status_code}")
        except Exception as e:
            self.environment.runner.quit()
            print(f"Critical error: {e}")


class StepLoadShape:
    def __init__(self, num_users=5, spawn_rate=5, run_time=60):
        self.num_users = num_users
        self.spawn_rate = spawn_rate
        self.run_time = run_time
        self.start_time = None

    def tick(self):
        if self.start_time is None:
            self.start_time = events.global_stats.start_time
        
        run_time = round(events.global_stats.total_run_time())
        
        if run_time >= self.run_time:
            return None
            
        return self.num_users, self.spawn_rate
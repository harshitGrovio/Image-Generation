import json
import urllib.request
import urllib.error
import time

BASE_URL = "http://localhost:5005/models"

queries = [
    # Social Media
    "instagram-post",
    "instagram-story",
    "facebook-ads",
    "linkedin-content",
    "tiktok-thumbnails",
    "youtube-thumbnails",
    "pinterest-pins",
    
    # Marketing & Business
    "display-ads",
    "product-listings",
    "hero-banners",
    "logos-posters",
    "headshots",
    "print-billboard",
    "quick-mockups",
    
    # General Creation & Editing
    "create-images",
    "edit-images",
    "image-editing",
    "image-upscale",
    "vector-icons",
    "fashion-tryon",
    "character-multiple-angles"
]

results = {}

print(f"Starting to query {len(queries)} use-cases from {BASE_URL}...")

for q in queries:
    url = f"{BASE_URL}?query={q}"
    print(f"Querying: {q}...", end=" ")
    try:
        with urllib.request.urlopen(url) as response:
            if response.status == 200:
                data = json.loads(response.read().decode())
                results[q] = data
                print("✅")
            else:
                print(f"❌ (Status {response.status})")
                results[q] = {"error": f"HTTP {response.status}"}
    except urllib.error.URLError as e:
        print(f"❌ (Error: {e})")
        results[q] = {"error": str(e)}
    except Exception as e:
        print(f"❌ (Exception: {e})")
        results[q] = {"error": str(e)}
    
    time.sleep(0.1)  # Be nice to the server

output_file = "query_responses.json"
with open(output_file, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nDone! Saved responses to {output_file}")

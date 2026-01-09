import json

INPUT_FILE = "query_responses.json"
OUTPUT_FILE = "extracted_models.py"

# Mapping price_per_mp to credits based on api_server.py logic
# 0.005 -> 1
# 0.015 -> 2
# 0.027 -> 3
# 0.035 -> 4
# 0.03  -> 5  (Wait, 0.03 is 5? distinct from 0.027)
# 0.05  -> 6
# 0.15  -> 7
PRICE_TO_CREDITS = {
    0.005: 1,
    0.015: 2,
    0.027: 3, # close to 0.03?
    0.024: 3, # Seen in errors 0.024
    0.03: 5,
    0.035: 4,
    0.04: 4, # Seen in errors
    0.05: 6,
    0.15: 7
}

def get_credits(price):
    if price is None: return 1
    # Find closest match
    closest = min(PRICE_TO_CREDITS.keys(), key=lambda x: abs(x - price))
    return PRICE_TO_CREDITS[closest]

def main():
    try:
        with open(INPUT_FILE, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {INPUT_FILE} not found")
        return

    models_map = {}

    for query, response in data.items():
        models = response.get("models", [])
        for m in models:
            mid = m.get("model") or m.get("id") # Some might rely on ID? 
            # Actually json uses "model": "flux2-pro" inside the object usually
            if not mid: continue
            
            # If we saw this model before, check if we have better data now
            # We want the most descriptive version
            
            # Extract basic fields
            entry = {
                "model_id": mid,
                "name": m.get("name"), 
                "fal_endpoint": m.get("id"),
                "description": m.get("description"),
                "credits": get_credits(m.get("price_per_mp")), # Still useful for logic
                "price_per_mp": m.get("price_per_mp"), # Extract exact price
                "tier": m.get("tier"), # If in JSON (it isn't usually, but maybe for some?)
                "is_edit_model": m.get("type") == "editing" or m.get("supports_editing") is True,
                "enabled": True,
                "max_input_images": m.get("max_input_images"),
                "max_output_images": m.get("max_output_images"),
                "sample_prompt": m.get("sample_prompt"),
                "supported_sizes": m.get("supported_sizes"), # Capture the whole dict!
                # We still need model_size for billing logic if api_server uses it?
                # api_server uses model_size for credit mults.
                # seed_models.py defines model_size.
                # We can keep model_size as generic copy for now.
                "model_size": "SIZE_MULTIPLIERS.copy()", # Placeholder str to be replaced
            }
            
            # Store/Update
            if mid not in models_map:
                models_map[mid] = entry
            else:
                # Update if new one has more info (e.g. supported_sizes might be fuller in create-images than others?)
                # create-images lists native sizes. valid.
                curr = models_map[mid]
                
                # If current doesn't have supported_sizes but new one does, take it
                if not curr.get("supported_sizes") and entry.get("supported_sizes"):
                    curr["supported_sizes"] = entry["supported_sizes"]
                
                # If current supported_sizes lacks "platform_sizes" key (good), but new one has it (bad - likely specific use case override),
                # we prefer the generic one for the seed model.
                # ACTUALLY, "create-images" usually has the generic full list.
                # "hero-banners" has specific overrides.
                # We should prefer the one WTIHOUT "platform_sizes" key inside supported_sizes as the base truth.
                
                if entry.get("supported_sizes") and "platform_sizes" not in entry["supported_sizes"]:
                     curr["supported_sizes"] = entry["supported_sizes"]
                     
                # Update max_output_images if present
                if entry.get("max_output_images") is not None:
                     curr["max_output_images"] = entry["max_output_images"]
                     
                if entry.get("price_per_mp") is not None:
                    curr["price_per_mp"] = entry["price_per_mp"]

    # Convert to list
    image_models = list(models_map.values())
    
    # Sort for stability
    image_models.sort(key=lambda x: x["model_id"])

    # Generate Python code
    py_content = "IMAGE_MODELS = [\n"
    for m in image_models:
        py_content += "    {\n"
        for k, v in m.items():
            if k == "model_size" and v == "SIZE_MULTIPLIERS.copy()":
                 py_content += f'        "{k}": SIZE_MULTIPLIERS.copy(),\n'
            else:
                 py_content += f'        "{k}": {repr(v)},\n'
        py_content += "    },\n"
    py_content += "]\n"

    with open(OUTPUT_FILE, "w") as f:
        f.write(py_content)
    
    print(f"Extracted {len(image_models)} models to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

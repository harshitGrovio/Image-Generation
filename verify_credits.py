
from seed_models import IMAGE_MODELS

EXPECTED_CREDITS = {
    # Baseline
    "z-image-turbo": 8,
    "flux2-dev": 20,
    "flux2-pro": 40,
    "seedream-v4": 40,
    "dreamina-v3.1": 40,
    "seedream-v4.5": 55,
    "reve": 55,
    "gemini-3-pro": 200,

    # Edit / Remix / Upscale
    "flux2-pro-edit": 40,
    "flux2-dev-edit": 35,
    "seedream-v4-edit": 40,
    "seedream-v4.5-edit": 55,
    "reve-edit": 55,
    "reve-remix": 55,       # "Reve Edit / Remix / Fast" -> 55
    "reve-fast-edit": 55,   # inferred
    "reve-fast-remix": 55,  # inferred
    "gemini-3-pro-edit": 200,
    "clarity-upscaler": 30,
    "creative-upscaler": 40,
    "gpt-image-1.5": 55,
    "ideogram-v3": 40,
    "ideogram-v3-reframe": 40,
    "multiple-angles": 65,  # Qwen Multi-Angle
    "recraft-upscale": 30,
    
    # Missing from table but exist in code?
    # "gpt-image-1.5-edit"? Table says "GPT Image 1.5" -> 55. Assuming edit is same/similar or not in table?
    # Actually table has "GPT Image 1.5" in the "Image Edit / Remix / Upscale" section too? No, it's in the list.
    # Wait, GPT Image 1.5 is in "IMAGE EDIT / REMIX / UPSCALE (FINAL)" table with 55 credits.
    # It corresponds to "gpt-image-1.5" (Base) and likely "gpt-image-1.5-edit".
    "gpt-image-1.5-edit": 55, 
}

def verify():
    print("Verifying credits in seed_models.py...")
    errors = []
    
    # Map model_id to model entry
    models_map = {m["model_id"]: m for m in IMAGE_MODELS}
    
    for model_id, expected_credit in EXPECTED_CREDITS.items():
        if model_id not in models_map:
            errors.append(f"Missing model: {model_id}")
            continue
            
        actual_credit = models_map[model_id]["credits"]
        if actual_credit != expected_credit:
            errors.append(f"Mismatch for {model_id}: Expected {expected_credit}, got {actual_credit}")
        else:
            print(f"OK: {model_id} = {actual_credit}")

    if errors:
        print("\n❌ Verification Failed:")
        for e in errors:
            print(f"  - {e}")
        exit(1)
    else:
        print("\n✅ Verification Passed! All credits match.")

if __name__ == "__main__":
    verify()

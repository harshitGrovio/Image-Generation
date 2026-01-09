
from seed_models import SIZE_MULTIPLIERS

EXPECTED_MULTIPLIERS = {
    "512x512": 1.0,
    "768x1024": 1.0,
    "576x1024": 1.0,
    "1024x768": 1.0,
    "1024x576": 1.0,
    "1024x1024": 1.0,
    "1536x640": 1.0,
    "640x1536": 1.0,
    "1024x1536": 1.6,
    "1536x1024": 1.6,
    "2048x2048": 2.2,
    "4096x4096": 2.2,
}

def verify():
    print("Verifying SIZE_MULTIPLIERS in seed_models.py...")
    errors = []
    
    for size, expected in EXPECTED_MULTIPLIERS.items():
        actual = SIZE_MULTIPLIERS.get(size)
        if actual is None:
            errors.append(f"Missing size: {size}")
        elif actual != expected:
            errors.append(f"Mismatch for {size}: Expected {expected}, got {actual}")
        else:
            print(f"OK: {size} = {actual}")

    if errors:
        print("\n❌ Verification Failed:")
        for e in errors:
            print(f"  - {e}")
        exit(1)
    else:
        print("\n✅ Verification Passed! All multipliers match.")

if __name__ == "__main__":
    verify()


import asyncio
import os
import sys

# Add current directory to path
sys.path.append(os.getcwd())

from bytedance_client import BytedanceSeedreamClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def test():
    print("Testing BytedanceSeedreamClient...")
    try:
        client = BytedanceSeedreamClient()
        print(f"Client initialized with API Key: {'*' * 8 if client.api_key else 'None'}")
        
        print("Sending request...")
        result = await client.generate_image(
            prompt="A cute dog",
            model="seedream-v4.5",
            width=1024,
            height=1024,
        )
        print(f"\nResult Success: {result.success}")
        print(f"Result Error: {result.error}")
        print(f"Result Images Count: {len(result.images)}")
        if result.success and result.images:
            print(f"Image URL: {result.images[0]['url']}")
        
        # Test 4.0 as well since logs mentioned seedream-v4
        if not result.success or len(result.images) == 0:
            print("\nTesting Seedream V4...")
            result_v4 = await client.generate_image(
                prompt="A cute dog",
                model="seedream-v4",
                width=1024,
                height=1024,
            )
            print(f"\nResult V4 Success: {result_v4.success}")
            print(f"Result V4 Error: {result_v4.error}")
            print(f"Result V4 Images Count: {len(result_v4.images)}")
            
    except Exception as e:
        print(f"Test Exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test())

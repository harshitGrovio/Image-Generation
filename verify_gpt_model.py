#!/usr/bin/env python3
"""
Quick verification script to check GPT Image 1.5 model in MongoDB
"""
import os
from pymongo import MongoClient

# Connect to MongoDB
connection_string = os.environ.get("MONGODB_CONNECTION_STRING") or "mongodb+srv://shivraj:9nq6GM5DGHX4irNd@shivrajcluster.dv2no.mongodb.net/?retryWrites=true&w=majority"
client = MongoClient(connection_string, tlsAllowInvalidCertificates=True)
db = client["Grovio_Mini_Apps"]
models_collection = db["Image_Models"]

# Find GPT Image 1.5 model
gpt_model = models_collection.find_one({"model_id": "gpt-image-1.5"})

if gpt_model:
    print("✅ Found GPT Image 1.5 model in MongoDB:")
    print(f"   Model ID: {gpt_model['model_id']}")
    print(f"   Name: {gpt_model['name']}")
    print(f"   Description: {gpt_model['description']}")
    print(f"\n📐 Supported Sizes:")
    supported_sizes = gpt_model.get('supported_sizes', {})
    print(f"   Family: {supported_sizes.get('family')}")
    print(f"   Sizes: {supported_sizes.get('sizes')}")
    print(f"   Aspect Ratios: {supported_sizes.get('aspect_ratios')}")
    print(f"   Max Resolution: {supported_sizes.get('max_resolution')}")
    print(f"   Notes: {supported_sizes.get('notes')}")

    # Verify correct sizes
    expected_sizes = ['1024x1024', '1024x1536', '1536x1024']
    actual_sizes = supported_sizes.get('sizes', [])

    if actual_sizes == expected_sizes:
        print("\n✅ VERIFICATION PASSED: Sizes are correct!")
    else:
        print(f"\n❌ VERIFICATION FAILED: Expected {expected_sizes}, got {actual_sizes}")
else:
    print("❌ GPT Image 1.5 model not found in database!")

client.close()

import json
import os
from pymongo import MongoClient

EXPECTED_FILE = "query_responses.json"
OUTPUT_FILE = "seed_models.py"

# Hardcoded pre-content to recover from empty file state
PRE_CONTENT = '''"""
Seed Script for Image Models Database
Populates the Image_Models collection with all models and their credit costs.

Usage:
    python3 seed_models.py
"""

import os
from datetime import datetime, timezone

# MongoDB imports - using pymongo (sync) instead of motor for simpler seeding
from pymongo import MongoClient

# =============================================================================
# TIER DEFINITIONS (Based on subscription plans)
# =============================================================================
# Starter: $19/mo 9,000 credits
# Pro: $49/mo 24,000 credits  
# Creator: $99/mo 48,000 credits
# Team/Studio: $199/mo 97,000 credits

TIERS = {
    "Starter": {"price": 19, "credits": 9000},
    "Pro": {"price": 49, "credits": 24000},
    "Creator": {"price": 99, "credits": 48000},
    "Team": {"price": 199, "credits": 97000},
}

# =============================================================================
# SIZE CREDIT MULTIPLIERS
# Smaller sizes cost less, larger sizes cost more
# =============================================================================
SIZE_MULTIPLIERS = {
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

# =============================================================================
# IMAGE MODELS - Complete list with credit costs
# =============================================================================
IMAGE_MODELS = [
    {
        "model_id": 'clarity-upscaler',
        "name": 'CLARITY_UPSCALER',
        "fal_endpoint": 'fal-ai/clarity-upscaler',
        "description": 'Clarity Upscaler - Sharpness and detail enhancement',
        "credits": 30,
        "price_per_mp": 0.02,
        "tier": None,
        "is_edit_model": True,
        "enabled": True,
        "max_input_images": 1,
        "max_output_images": 4,
        "sample_prompt": 'Upscale and enhance this image to higher resolution while maintaining sharpness and adding realistic detail. Preserve the original composition and style. Remove compression artifacts, enhance textures, and improve overall clarity. Suitable for print, large displays, or high-resolution viewing.',
        "supported_sizes": {'family': 'upscale', 'sizes': ['Up to 4x input size'], 'aspect_ratios': ['Preserves input aspect ratio'], 'max_resolution': '4096x4096', 'notes': 'Output size = input size × scale factor (1-4x)'},
        "model_size": SIZE_MULTIPLIERS.copy(),
    },
    {
        "model_id": 'creative-upscaler',
        "name": 'CREATIVE_UPSCALER',
        "fal_endpoint": 'fal-ai/creative-upscaler',
        "description": 'Creative Upscaler - AI enhancement and upscaling',
        "credits": 40,
        "price_per_mp": 0.02,
        "tier": None,
        "is_edit_model": True,
        "enabled": True,
        "max_input_images": 1,
        "max_output_images": 4,
        "sample_prompt": 'Upscale and enhance this image to higher resolution while maintaining sharpness and adding realistic detail. Preserve the original composition and style. Remove compression artifacts, enhance textures, and improve overall clarity. Suitable for print, large displays, or high-resolution viewing.',
        "supported_sizes": {'family': 'upscale', 'sizes': ['Up to 4x input size'], 'aspect_ratios': ['Preserves input aspect ratio'], 'max_resolution': '4096x4096', 'notes': 'Output size = input size × scale factor (1-4x)'},
        "model_size": SIZE_MULTIPLIERS.copy(),
    },
    {
        "model_id": 'dreamina-v3.1',
        "name": 'DREAMINA_V31',
        "fal_endpoint": 'fal-ai/bytedance/dreamina/v3.1/text-to-image',
        "description": 'ByteDance Dreamina V3.1 - Specialized portrait generation',
        "credits": 40,
        "price_per_mp": 0.027,
        "tier": None,
        "is_edit_model": False,
        "enabled": True,
        "max_input_images": 0,
        "max_output_images": 4,
        "sample_prompt": 'A minimalist flat lay of a morning coffee setup on a rustic wooden table, soft natural lighting, latte art, croissant on the side, cozy atmosphere, high resolution',
        "supported_sizes": {'family': 'general', 'sizes': ['1024x1024', '768x1024', '1024x768'], 'aspect_ratios': ['1:1', '3:4', '4:3'], 'max_resolution': '1024x1024', 'notes': 'Size typically inherited from input image'},
        "model_size": SIZE_MULTIPLIERS.copy(),
    },
    {
        "model_id": 'flux2-dev',
        "name": 'FLUX2_DEV',
        "fal_endpoint": 'fal-ai/flux-2',
        "description": 'FLUX.2 Dev - High quality generation from Black Forest Labs',
        "credits": 20,
        "price_per_mp": 0.012,
        "tier": None,
        "is_edit_model": False,
        "enabled": True,
        "max_input_images": 0,
        "max_output_images": 4,
        "sample_prompt": 'Eye-catching social media content, vibrant colors, engaging composition, modern design, scroll-stopping visual',
        "supported_sizes": {'family': 'flux2', 'sizes': ['1024x1024', '768x1024', '576x1024', '1024x768', '1024x576', '1536x640', '640x1536'], 'aspect_ratios': ['1:1', '3:4', '9:16', '4:3', '16:9', '21:9', '9:21'], 'max_resolution': '1536x1536', 'notes': 'Supports custom dimensions up to 1536px'},
        "model_size": SIZE_MULTIPLIERS.copy(),
    },
    {
        "model_id": 'flux2-dev-edit',
        "name": 'FLUX2_DEV_EDIT',
        "fal_endpoint": 'fal-ai/flux-2/edit',
        "description": 'FLUX.2 Dev Edit - Image editing with natural language',
        "credits": 35,
        "price_per_mp": 0.024,
        "tier": None,
        "is_edit_model": True,
        "enabled": True,
        "max_input_images": 3,
        "max_output_images": 4,
        "sample_prompt": 'Edit for marketing variation, maintain brand consistency, professional quality',
        "supported_sizes": {'family': 'editing', 'sizes': [], 'aspect_ratios': [], 'max_resolution': 'Same as input', 'notes': 'Inherits size and aspect ratio from input image'},
        "model_size": SIZE_MULTIPLIERS.copy(),
    },
    {
        "model_id": 'flux2-pro',
        "name": 'FLUX2_PRO',
        "fal_endpoint": 'fal-ai/flux-2-pro',
        "description": 'FLUX.2 Pro - Maximum quality from Black Forest Labs',
        "credits": 40,
        "price_per_mp": 0.03,
        "tier": None,
        "is_edit_model": False,
        "enabled": True,
        "max_input_images": 0,
        "max_output_images": 1,
        "sample_prompt": 'A minimalist flat lay of a morning coffee setup on a rustic wooden table, soft natural lighting, latte art, croissant on the side, cozy atmosphere, high resolution',
        "supported_sizes": {'family': 'flux2', 'sizes': ['1024x1024', '768x1024', '576x1024', '1024x768', '1024x576', '1536x640', '640x1536'], 'aspect_ratios': ['1:1', '3:4', '9:16', '4:3', '16:9', '21:9', '9:21'], 'max_resolution': '1536x1536', 'notes': 'Supports custom dimensions up to 1536px'},
        "model_size": SIZE_MULTIPLIERS.copy(),
    },
    {
        "model_id": 'flux2-pro-edit',
        "name": 'FLUX2_PRO_EDIT',
        "fal_endpoint": 'fal-ai/flux-2-pro/edit',
        "description": 'FLUX.2 Pro Edit - Premium image editing',
        "credits": 40,
        "price_per_mp": 0.03,
        "tier": None,
        "is_edit_model": True,
        "enabled": True,
        "max_input_images": 3,
        "max_output_images": 4,
        "sample_prompt": 'A minimalist flat lay of a morning coffee setup on a rustic wooden table, soft natural lighting, latte art, croissant on the side, cozy atmosphere, high resolution',
        "supported_sizes": {'family': 'editing', 'sizes': [], 'aspect_ratios': [], 'max_resolution': 'Same as input', 'notes': 'Inherits size and aspect ratio from input image'},
        "model_size": SIZE_MULTIPLIERS.copy(),
    },
    {
        "model_id": 'gemini-3-pro',
        "name": 'GEMINI3_PRO',
        "fal_endpoint": 'fal-ai/gemini-3-pro-image-preview',
        "description": 'Nano Banana Pro - Realism and typography with web search',
        "credits": 200,
        "price_per_mp": 0.15,
        "tier": None,
        "is_edit_model": False,
        "enabled": True,
        "max_input_images": 0,
        "max_output_images": 4,
        "sample_prompt": 'A minimalist flat lay of a morning coffee setup on a rustic wooden table, soft natural lighting, latte art, croissant on the side, cozy atmosphere, high resolution',
        "supported_sizes": {'family': 'nano-banana', 'sizes': ['1024x1024', '768x1024', '576x1024', '1024x768', '1024x576'], 'aspect_ratios': ['1:1', '3:4', '9:16', '4:3', '16:9'], 'max_resolution': '1024x1024', 'notes': "Google's Gemini-based model"},
        "model_size": SIZE_MULTIPLIERS.copy(),
    },
    {
        "model_id": 'gemini-3-pro-edit',
        "name": 'GEMINI3_PRO_EDIT',
        "fal_endpoint": 'fal-ai/gemini-3-pro-image-preview/edit',
        "description": 'Nano Banana Pro Edit - Multi-image editing',
        "credits": 200,
        "price_per_mp": 0.15,
        "tier": None,
        "is_edit_model": True,
        "enabled": True,
        "max_input_images": 10,
        "max_output_images": 4,
        "sample_prompt": 'A minimalist flat lay of a morning coffee setup on a rustic wooden table, soft natural lighting, latte art, croissant on the side, cozy atmosphere, high resolution',
        "supported_sizes": {'family': 'editing', 'sizes': [], 'aspect_ratios': [], 'max_resolution': 'Same as input', 'notes': 'Inherits size and aspect ratio from input image'},
        "model_size": SIZE_MULTIPLIERS.copy(),
    },
    {
        "model_id": 'gpt-image-1.5',
        "name": 'GPT_IMAGE_15',
        "fal_endpoint": 'fal-ai/gpt-image-1.5',
        "description": 'GPT Image 1.5 - Photorealistic generation with transparent backgrounds',
        "credits": 55,
        "price_per_mp": 0.05,
        "tier": None,
        "is_edit_model": False,
        "enabled": True,
        "max_input_images": 0,
        "max_output_images": 4,
        "sample_prompt": 'A minimalist flat lay of a morning coffee setup on a rustic wooden table, soft natural lighting, latte art, croissant on the side, cozy atmosphere, high resolution',
        "supported_sizes": {'family': 'general', 'sizes': ['1024x1024', '768x1024', '1024x768'], 'aspect_ratios': ['1:1', '3:4', '4:3'], 'max_resolution': '1024x1024', 'notes': 'Size typically inherited from input image'},
        "model_size": SIZE_MULTIPLIERS.copy(),
    },
    {
        "model_id": 'gpt-image-1.5-edit',
        "name": 'GPT_IMAGE_15_EDIT',
        "fal_endpoint": 'fal-ai/gpt-image-1.5/edit',
        "description": 'GPT Image 1.5 Edit - Multi-image editing with transparent backgrounds',
        "credits": 55,
        "price_per_mp": 0.05,
        "tier": None,
        "is_edit_model": True,
        "enabled": True,
        "max_input_images": 10,
        "max_output_images": 4,
        "sample_prompt": 'A minimalist flat lay of a morning coffee setup on a rustic wooden table, soft natural lighting, latte art, croissant on the side, cozy atmosphere, high resolution',
        "supported_sizes": {'family': 'editing', 'sizes': [], 'aspect_ratios': [], 'max_resolution': 'Same as input', 'notes': 'Inherits size and aspect ratio from input image'},
        "model_size": SIZE_MULTIPLIERS.copy(),
    },
    {
        "model_id": 'ideogram-v3',
        "name": 'IDEOGRAM_V3',
        "fal_endpoint": 'fal-ai/ideogram/v3',
        "description": 'Ideogram V3 - Superior typography and style presets',
        "credits": 40,
        "price_per_mp": 0.08,
        "tier": None,
        "is_edit_model": False,
        "enabled": True,
        "max_input_images": 0,
        "max_output_images": 4,
        "sample_prompt": 'A minimalist flat lay of a morning coffee setup on a rustic wooden table, soft natural lighting, latte art, croissant on the side, cozy atmosphere, high resolution',
        "supported_sizes": {'family': 'ideogram', 'sizes': ['1024x1024', '768x1024', '576x1024', '1024x768', '1024x576', '1024x1536', '1536x1024'], 'aspect_ratios': ['1:1', '3:4', '9:16', '4:3', '16:9', '2:3', '3:2'], 'max_resolution': '1536x1536', 'notes': 'Best for logos and text-heavy images'},
        "model_size": SIZE_MULTIPLIERS.copy(),
    },
    {
        "model_id": 'ideogram-v3-reframe',
        "name": 'IDEOGRAM_V3_REFRAME',
        "fal_endpoint": 'fal-ai/ideogram/v3/reframe',
        "description": 'Ideogram V3 Reframe - Image resizing and outpainting',
        "credits": 40,
        "price_per_mp": 0.08,
        "tier": None,
        "is_edit_model": True,
        "enabled": True,
        "max_input_images": 1,
        "max_output_images": 4,
        "sample_prompt": 'A minimalist flat lay of a morning coffee setup on a rustic wooden table, soft natural lighting, latte art, croissant on the side, cozy atmosphere, high resolution',
        "supported_sizes": {'family': 'general', 'sizes': ['1024x1024', '768x1024', '1024x768'], 'aspect_ratios': ['1:1', '3:4', '4:3'], 'max_resolution': '1024x1024', 'notes': 'Size typically inherited from input image'},
        "model_size": SIZE_MULTIPLIERS.copy(),
    },
    {
        "model_id": 'multiple-angles',
        "name": 'QWEN_MULTIPLE_ANGLES',
        "fal_endpoint": 'fal-ai/qwen-image-edit-2509-lora-gallery/multiple-angles',
        "description": 'Qwen Multiple Angles - Generate multi-view perspectives',
        "credits": 65,
        "price_per_mp": 0.03,
        "tier": None,
        "is_edit_model": True,
        "enabled": True,
        "max_input_images": 1,
        "max_output_images": 8,
        "sample_prompt": 'Generate multiple camera angles of this character, front view, side view, back view, consistent character sheet style',
        "supported_sizes": {'family': 'general', 'sizes': ['1024x1024', '768x1024', '1024x768'], 'aspect_ratios': ['1:1', '3:4', '4:3'], 'max_resolution': '1024x1024', 'notes': 'Size typically inherited from input image'},
        "model_size": SIZE_MULTIPLIERS.copy(),
    },
    {
        "model_id": 'recraft-upscale',
        "name": 'RECRAFT_UPSCALE',
        "fal_endpoint": 'fal-ai/recraft/upscale/creative',
        "description": 'Recraft Upscale - Vector-style creative upscaling',
        "credits": 30,
        "price_per_mp": 0.02,
        "tier": None,
        "is_edit_model": True,
        "enabled": True,
        "max_input_images": 1,
        "max_output_images": 4,
        "sample_prompt": 'Upscale and enhance this image to higher resolution while maintaining sharpness and adding realistic detail. Preserve the original composition and style. Remove compression artifacts, enhance textures, and improve overall clarity. Suitable for print, large displays, or high-resolution viewing.',
        "supported_sizes": {'family': 'upscale', 'sizes': ['Up to 4x input size'], 'aspect_ratios': ['Preserves input aspect ratio'], 'max_resolution': '4096x4096', 'notes': 'Output size = input size × scale factor (1-4x)'},
        "model_size": SIZE_MULTIPLIERS.copy(),
    },
    {
        "model_id": 'reve',
        "name": 'REVE',
        "fal_endpoint": 'fal-ai/reve/text-to-image',
        "description": 'Reve - State-of-the-art generation',
        "credits": 55,
        "price_per_mp": 0.04,
        "tier": None,
        "is_edit_model": False,
        "enabled": True,
        "max_input_images": 0,
        "max_output_images": 4,
        "sample_prompt": 'A minimalist flat lay of a morning coffee setup on a rustic wooden table, soft natural lighting, latte art, croissant on the side, cozy atmosphere, high resolution',
        "supported_sizes": {'family': 'general', 'sizes': ['1024x1024', '768x1024', '1024x768'], 'aspect_ratios': ['1:1', '3:4', '4:3'], 'max_resolution': '1024x1024', 'notes': 'Size typically inherited from input image'},
        "model_size": SIZE_MULTIPLIERS.copy(),
    },
    {
        "model_id": 'reve-edit',
        "name": 'REVE_EDIT',
        "fal_endpoint": 'fal-ai/reve/edit',
        "description": 'Reve Edit - Professional image editing',
        "credits": 55,
        "price_per_mp": 0.04,
        "tier": None,
        "is_edit_model": True,
        "enabled": True,
        "max_input_images": 1,
        "max_output_images": 4,
        "sample_prompt": 'A minimalist flat lay of a morning coffee setup on a rustic wooden table, soft natural lighting, latte art, croissant on the side, cozy atmosphere, high resolution',
        "supported_sizes": {'family': 'editing', 'sizes': [], 'aspect_ratios': [], 'max_resolution': 'Same as input', 'notes': 'Inherits size and aspect ratio from input image'},
        "model_size": SIZE_MULTIPLIERS.copy(),
    },
    {
        "model_id": 'reve-fast-edit',
        "name": 'REVE_FAST_EDIT',
        "fal_endpoint": 'fal-ai/reve/fast/edit',
        "description": 'Reve Fast Edit - Quick iteration editing',
        "credits": 55,
        "price_per_mp": 0.04,
        "tier": None,
        "is_edit_model": True,
        "enabled": True,
        "max_input_images": 1,
        "max_output_images": 4,
        "sample_prompt": 'A minimalist flat lay of a morning coffee setup on a rustic wooden table, soft natural lighting, latte art, croissant on the side, cozy atmosphere, high resolution',
        "supported_sizes": {'family': 'editing', 'sizes': [], 'aspect_ratios': [], 'max_resolution': 'Same as input', 'notes': 'Inherits size and aspect ratio from input image'},
        "model_size": SIZE_MULTIPLIERS.copy(),
    },
    {
        "model_id": 'reve-fast-remix',
        "name": 'REVE_FAST_REMIX',
        "fal_endpoint": 'fal-ai/reve/fast/remix',
        "description": 'Reve Fast Remix - Quick multi-image remixing',
        "credits": 55,
        "price_per_mp": 0.04,
        "tier": None,
        "is_edit_model": True,
        "enabled": True,
        "max_input_images": 6,
        "max_output_images": 4,
        "sample_prompt": 'Fast creative remix, professional quality, quick output',
        "supported_sizes": {'family': 'general', 'sizes': ['1024x1024', '768x1024', '1024x768'], 'aspect_ratios': ['1:1', '3:4', '4:3'], 'max_resolution': '1024x1024', 'notes': 'Size typically inherited from input image'},
        "model_size": SIZE_MULTIPLIERS.copy(),
    },
    {
        "model_id": 'reve-remix',
        "name": 'REVE_REMIX',
        "fal_endpoint": 'fal-ai/reve/remix',
        "description": 'Reve Remix - Creative multi-image composition',
        "credits": 55,
        "price_per_mp": 0.04,
        "tier": None,
        "is_edit_model": True,
        "enabled": True,
        "max_input_images": 6,
        "max_output_images": 4,
        "sample_prompt": 'Creative multi-image remix, seamless composition, marketing-ready',
        "supported_sizes": {'family': 'general', 'sizes': ['1024x1024', '768x1024', '1024x768'], 'aspect_ratios': ['1:1', '3:4', '4:3'], 'max_resolution': '1024x1024', 'notes': 'Size typically inherited from input image'},
        "model_size": SIZE_MULTIPLIERS.copy(),
    },
    {
        "model_id": 'seedream-v4',
        "name": 'SEEDREAM_V4',
        "fal_endpoint": 'fal-ai/bytedance/seedream/v4/text-to-image',
        "description": 'ByteDance Seedream V4 - 4K High Resolution Generation',
        "credits": 40,
        "price_per_mp": 0.03,
        "tier": None,
        "is_edit_model": False,
        "enabled": True,
        "max_input_images": 0,
        "max_output_images": 6,
        "sample_prompt": 'A minimalist flat lay of a morning coffee setup on a rustic wooden table, soft natural lighting, latte art, croissant on the side, cozy atmosphere, high resolution',
        "supported_sizes": {'family': 'general', 'sizes': ['1024x1024', '768x1024', '1024x768'], 'aspect_ratios': ['1:1', '3:4', '4:3'], 'max_resolution': '1024x1024', 'notes': 'Size typically inherited from input image'},
        "model_size": SIZE_MULTIPLIERS.copy(),
    },
    {
        "model_id": 'seedream-v4-edit',
        "name": 'SEEDREAM_V4_EDIT',
        "fal_endpoint": 'fal-ai/bytedance/seedream/v4/edit',
        "description": 'ByteDance Seedream V4 Edit - 4K High Resolution Editing',
        "credits": 40,
        "price_per_mp": 0.03,
        "tier": None,
        "is_edit_model": True,
        "enabled": True,
        "max_input_images": 10,
        "max_output_images": 6,
        "sample_prompt": 'A minimalist flat lay of a morning coffee setup on a rustic wooden table, soft natural lighting, latte art, croissant on the side, cozy atmosphere, high resolution',
        "supported_sizes": {'family': 'editing', 'sizes': [], 'aspect_ratios': [], 'max_resolution': 'Same as input', 'notes': 'Inherits size and aspect ratio from input image'},
        "model_size": SIZE_MULTIPLIERS.copy(),
    },
    {
        "model_id": 'seedream-v4.5',
        "name": 'SEEDREAM_V45',
        "fal_endpoint": 'fal-ai/bytedance/seedream/v4.5/text-to-image',
        "description": 'ByteDance Seedream V4.5 - Stylized content generation',
        "credits": 55,
        "price_per_mp": 0.04,
        "tier": None,
        "is_edit_model": False,
        "enabled": True,
        "max_input_images": 0,
        "max_output_images": 4,
        "sample_prompt": 'A minimalist flat lay of a morning coffee setup on a rustic wooden table, soft natural lighting, latte art, croissant on the side, cozy atmosphere, high resolution',
        "supported_sizes": {'family': 'seedream', 'sizes': ['1024x1024', '768x1024', '576x1024', '1024x768', '1024x576'], 'aspect_ratios': ['1:1', '3:4', '9:16', '4:3', '16:9'], 'max_resolution': '1024x1024', 'notes': "ByteDance's latest model"},
        "model_size": SIZE_MULTIPLIERS.copy(),
    },
    {
        "model_id": 'seedream-v4.5-edit',
        "name": 'SEEDREAM_V45_EDIT',
        "fal_endpoint": 'fal-ai/bytedance/seedream/v4.5/edit',
        "description": 'ByteDance Seedream Edit - Multi-image editing',
        "credits": 55,
        "price_per_mp": 0.04,
        "tier": None,
        "is_edit_model": True,
        "enabled": True,
        "max_input_images": 3,
        "max_output_images": 4,
        "sample_prompt": 'A minimalist flat lay of a morning coffee setup on a rustic wooden table, soft natural lighting, latte art, croissant on the side, cozy atmosphere, high resolution',
        "supported_sizes": {'family': 'editing', 'sizes': [], 'aspect_ratios': [], 'max_resolution': 'Same as input', 'notes': 'Inherits size and aspect ratio from input image'},
        "model_size": SIZE_MULTIPLIERS.copy(),
    },
    {
        "model_id": 'z-image-turbo',
        "name": 'Z_IMAGE_TURBO',
        "fal_endpoint": 'fal-ai/z-image/turbo',
        "description": 'Z-Image Turbo - Ultra fast 6B model from Tongyi-MAI',
        "credits": 8,
        "price_per_mp": 0.005,
        "tier": None,
        "is_edit_model": False,
        "enabled": True,
        "max_input_images": 0,
        "max_output_images": 4,
        "sample_prompt": 'A minimalist flat lay of a morning coffee setup on a rustic wooden table, soft natural lighting, latte art, croissant on the side, cozy atmosphere, high resolution',
        "supported_sizes": {'family': 'z-image', 'sizes': ['1024x1024', '768x1024', '576x1024', '1024x768', '1024x576'], 'aspect_ratios': ['1:1', '3:4', '9:16', '4:3', '16:9'], 'max_resolution': '1024x1024', 'notes': 'Ultra-fast 6B model from Tongyi-MAI'},
        "model_size": SIZE_MULTIPLIERS.copy(),
    },
]
'''

def main():
    try:
        with open(EXPECTED_FILE, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find {EXPECTED_FILE}")
        return

    new_use_cases = []
    sorted_keys = sorted(data.keys())
    
    for key in sorted_keys:
        item = data[key]

        # Check if original data has editing_models field
        has_editing_models_field = "editing_models" in item

        if has_editing_models_field:
            # If editing_models exists in original, use it directly
            gen_models = []
            edit_models = item.get("editing_models", [])

            # Extract gen models from models list
            models_list = item.get("models", [])
            for m in models_list:
                mid = m.get("model")
                if mid and mid not in edit_models:
                    gen_models.append(mid)
        else:
            # If no editing_models in original, all models go to models list
            models_list = item.get("models", [])
            gen_models = []
            edit_models = []

            for m in models_list:
                mid = m.get("model")
                if mid and mid not in gen_models:
                    gen_models.append(mid)

        # Name inferencing
        desc = item.get("description", "")
        name = key.replace("-", " ").title()
        if desc.startswith("Recommended models for "):
            name = desc.replace("Recommended models for ", "")

        uc = {
            "use_case_id": key,
            "name": name,
            "description": desc,
            "models": gen_models,
            "system_prompt": item.get("system_prompt"),
            "sample_prompt": item.get("sample_prompt"),
            "prompt_tips": item.get("prompt_tips"),
            "platform_sizes": item.get("platform_sizes"),
            "recommended_size": item.get("recommended_size") or item.get("recommended"),
            "notes": item.get("notes"),
            "enabled": True
        }

        # Only add editing_models if it existed in original query_responses.json
        if has_editing_models_field and edit_models:
            uc["editing_models"] = edit_models
        new_use_cases.append(uc)

    # Generate new USE_CASES strings
    use_cases_str = "USE_CASES = [\n"
    for uc in new_use_cases:
        use_cases_str += "    {\n"
        for k, v in uc.items():
            if v is not None:
                use_cases_str += f'        "{k}": {repr(v)},\n'
        use_cases_str += "    },\n"
    use_cases_str += "]\n"

    # Define the seed_models function and main block
    final_seed_logic = """
def seed_models():
    # Get mongo string from env or fallback
    connection_string = os.environ.get("MONGODB_CONNECTION_STRING") or "mongodb+srv://shivraj:9nq6GM5DGHX4irNd@shivrajcluster.dv2no.mongodb.net/?retryWrites=true&w=majority"
    
    print(f"Connecting to MongoDB...")
    client = MongoClient(connection_string, tlsAllowInvalidCertificates=True)
    db = client["Grovio_Mini_Apps"]
    
    # 1. Seed Image_Models
    print("Seeding Image_Models...")
    models_collection = db["Image_Models"]
    models_collection.delete_many({})
    
    # IMAGE_MODELS should be in global scope from PRE_CONTENT
    # We write it to file, so it is global.
    if 'IMAGE_MODELS' in globals():
        models_list = globals()['IMAGE_MODELS']
        if models_list:
            models_collection.insert_many(models_list)
            print(f"Inserted {len(models_list)} models.")
    else:
        print("Warning: IMAGE_MODELS not found")

    # 2. Seed Use_Cases
    print("Seeding Use_Cases...")
    use_cases_collection = db["Use_Cases"]
    use_cases_collection.delete_many({})
    
    if USE_CASES:
        use_cases_collection.insert_many(USE_CASES)
    print(f"Inserted {len(USE_CASES)} use cases.")
    
    print("✅ Database seeded successfully!")

if __name__ == "__main__":
    seed_models()
"""

    with open(OUTPUT_FILE, "w") as f:
        f.write(PRE_CONTENT + "\n\n" + use_cases_str + "\n" + final_seed_logic)
    
    print(f"✅ Automatically updated {OUTPUT_FILE} with {len(new_use_cases)} use cases and fixed seed logic.")

if __name__ == "__main__":
    main()

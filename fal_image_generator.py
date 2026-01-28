"""
Grovio AI Image Generator - Core Module
Handles image generation for marketing creatives with 50+ optimized models
Powered by fal.ai infrastructure
"""

import os
import asyncio
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from enum import Enum

# Grovio AI uses fal.ai infrastructure
try:
    import fal_client
    GROVIO_AI_AVAILABLE = True
except ImportError:
    GROVIO_AI_AVAILABLE = False
    print("⚠️ Grovio AI client not installed. Run: pip install fal-client")

# Bytedance Seedream client for direct API access
try:
    from bytedance_client import BytedanceSeedreamClient, get_bytedance_client
    BYTEDANCE_AVAILABLE = True
except ImportError:
    BYTEDANCE_AVAILABLE = False
    print("⚠️ Bytedance client not available")


class ImageModel(str, Enum):
    """
    Grovio AI image models for marketing creatives.
    
    Available Models:
    - FLUX 2: Latest flagship model from Black Forest Labs
    - Reve: State-of-the-art image generation and editing
    - Seedream: ByteDance's unified generation/editing model
    - Dreamina: ByteDance's portrait-focused model
    - Z-Image: Ultra-fast generation from Tongyi-MAI
    - Nano Banana Pro: Google's latest image model with web search
    - Enhancement: Upscaling, object removal, style transfer, etc.
    """
    # ==========================================================================
    # FLUX 2 Models (Latest 2025) - BEST FOR MARKETING
    # ==========================================================================
    FLUX2_PRO = "fal-ai/flux-2-pro"                           # Premium quality - $0.03/MP
    FLUX2_PRO_EDIT = "fal-ai/flux-2-pro/edit"                 # Premium editing
    FLUX2_DEV = "fal-ai/flux-2"                              # High quality - $0.025/MP
    FLUX2_DEV_EDIT = "fal-ai/flux-2/edit"                    # Dev editing

    # ==========================================================================
    # NANO BANANA PRO (GOOGLE GEMINI 3 PRO)
    # ==========================================================================
    GEMINI3_PRO = "fal-ai/gemini-3-pro-image-preview"        # Nano Banana Pro - $0.02/MP
    GEMINI3_PRO_EDIT = "fal-ai/gemini-3-pro-image-preview/edit"  # Nano Banana Pro Edit
    
    # ==========================================================================
    # REVE - STATE-OF-THE-ART IMAGE GENERATION & EDITING
    # ==========================================================================
    REVE = "fal-ai/reve/text-to-image"                       # State-of-the-art generation - $0.04/image
    REVE_EDIT = "fal-ai/reve/edit"                           # Image editing
    REVE_FAST_EDIT = "fal-ai/reve/fast/edit"                 # Fast image editing
    REVE_REMIX = "fal-ai/reve/remix"                         # Multi-image remixing (up to 6 images)
    REVE_FAST_REMIX = "fal-ai/reve/fast/remix"               # Fast multi-image remixing
    
    # ==========================================================================
    # BYTEDANCE - SEEDREAM V4.5
    # ==========================================================================
    SEEDREAM_V45 = "fal-ai/bytedance/seedream/v4.5/text-to-image"  # ByteDance latest
    SEEDREAM_V45_EDIT = "fal-ai/bytedance/seedream/v4.5/edit"      # ByteDance editing
    
    # ==========================================================================
    # BYTEDANCE - SEEDREAM V4 (4K Support)
    # ==========================================================================
    SEEDREAM_V4 = "fal-ai/bytedance/seedream/v4/text-to-image"     # 4K resolution support - $0.03/image
    SEEDREAM_V4_EDIT = "fal-ai/bytedance/seedream/v4/edit"         # 4K editing (up to 10 input images)
    
    # ==========================================================================
    # BYTEDANCE - DREAMINA V3.1 (Portrait-focused)
    # ==========================================================================
    DREAMINA_V31 = "fal-ai/bytedance/dreamina/v3.1/text-to-image"  # Portrait photography - $0.027/image
    
    # ==========================================================================
    # Z-IMAGE - TONGYI-MAI FAST MODEL
    # ==========================================================================
    Z_IMAGE_TURBO = "fal-ai/z-image/turbo"                   # Super fast 6B model
    
    # ==========================================================================
    # IDEOGRAM V3 - TYPOGRAPHY & STYLE PRESETS
    # ==========================================================================
    IDEOGRAM_V3 = "fal-ai/ideogram/v3"                       # Typography, style presets, up to 8 images
    IDEOGRAM_V3_REFRAME = "fal-ai/ideogram/v3/reframe"       # Image reframing/resizing with AI fill
    
    # ==========================================================================
    # GPT IMAGE 1.5 - OPENAI IMAGE MODEL VIA FAL
    # ==========================================================================
    GPT_IMAGE_15 = "fal-ai/gpt-image-1.5"                     # GPT Image 1.5, up to 4 images, transparent bg
    GPT_IMAGE_15_EDIT = "fal-ai/gpt-image-1.5/edit"           # GPT Image 1.5 Edit, image editing
    
    # ==========================================================================
    # UPSCALING MODELS
    # ==========================================================================
    CREATIVE_UPSCALER = "fal-ai/creative-upscaler"           # Creative upscaling with AI enhancement
    CLARITY_UPSCALER = "fal-ai/clarity-upscaler"             # Clarity-focused upscaling
    RECRAFT_UPSCALE = "fal-ai/recraft/upscale/creative"      # Recraft creative upscale
    
    # ==========================================================================
    # OBJECT & TEXT REMOVAL
    # ==========================================================================
    OBJECT_REMOVAL = "fal-ai/object-removal"                 # Remove objects with prompt/mask
    BRIA_ERASER = "fal-ai/bria/eraser"                       # Commercial-safe object eraser
    TEXT_REMOVAL = "fal-ai/text-removal"                     # Automatic text removal
    
    # ==========================================================================
    # STYLE TRANSFER & BACKGROUND
    # ==========================================================================
    STYLE_TRANSFER = "fal-ai/style-transfer"                 # Apply artistic styles
    BACKGROUND_CHANGE = "fal-ai/background-change"           # Change image background
    ADD_BACKGROUND = "fal-ai/add-background"                 # Add background to subject
    RELIGHTING = "fal-ai/relighting"                         # Adjust image lighting
    
    # ==========================================================================
    # CHARACTER MULTIPLE ANGLES (Camera control)
    # ==========================================================================
    QWEN_MULTIPLE_ANGLES = "fal-ai/qwen-image-edit-2509-lora-gallery/multiple-angles"  # Generate multiple camera angles


# Backward compatibility alias
ImageEditModel = ImageModel


# =============================================================================
# MODEL SETTINGS & MARKETING PROMPTS
# =============================================================================

MODEL_SETTINGS = {
    # FLUX 2 Pro - Premium quality marketing
    "fal-ai/flux-2-pro": {
        "guidance_scale": 3.5,
        "num_inference_steps": 28,
        "best_for": ["hero images", "product launches", "brand campaigns", "premium content"],
        "marketing_prompt": "Professional marketing photo, commercial quality, studio lighting, high-end brand aesthetic, clean composition, 8K resolution",
    },
    "fal-ai/flux-2-pro/edit": {
        "guidance_scale": 3.5,
        "num_inference_steps": 28,
        "best_for": ["premium editing", "hero image variations", "campaign assets"],
        "marketing_prompt": "Premium quality edit, maximum detail preservation, commercial grade",
    },
    # FLUX 2 Dev - High quality marketing
    "fal-ai/flux-2": {
        "guidance_scale": 2.5,
        "num_inference_steps": 28,
        "best_for": ["social media posts", "blog headers", "ad creatives"],
        "marketing_prompt": "Eye-catching social media content, vibrant colors, engaging composition, modern design, scroll-stopping visual",
    },
    "fal-ai/flux-2/edit": {
        "guidance_scale": 2.5,
        "num_inference_steps": 28,
        "best_for": ["image editing", "style changes", "modifications"],
        "marketing_prompt": "Edit for marketing variation, maintain brand consistency, professional quality",
    },
    
    # Nano Banana Pro
    "fal-ai/gemini-3-pro-image-preview": {
        "guidance_scale": 0,  # Uses aspect_ratio instead
        "num_inference_steps": 0,
        "best_for": ["realistic images", "typography", "web search enabled content"],
        "marketing_prompt": "Photorealistic marketing image, accurate text rendering, professional quality",
    },
    "fal-ai/gemini-3-pro-image-preview/edit": {
        "guidance_scale": 0,
        "num_inference_steps": 0,
        "best_for": ["multi-image editing", "combining images", "creative compositions"],
        "marketing_prompt": "Professional multi-image edit, seamless composition, marketing-ready",
    },
    
    # Reve - State-of-the-art generation & editing
    "fal-ai/reve/text-to-image": {
        "guidance_scale": 3.5,
        "num_inference_steps": 28,
        "best_for": ["high quality generation", "creative content", "photorealistic images"],
        "marketing_prompt": "State-of-the-art quality, photorealistic marketing visual, premium aesthetic",
    },
    "fal-ai/reve/edit": {
        "guidance_scale": 3.5,
        "num_inference_steps": 28,
        "best_for": ["image editing", "style changes", "object modifications"],
        "marketing_prompt": "Professional image edit, maintain quality, marketing-ready output",
    },
    "fal-ai/reve/fast/edit": {
        "guidance_scale": 3.5,
        "num_inference_steps": 20,
        "best_for": ["quick editing", "fast iterations", "rapid changes"],
        "marketing_prompt": "Fast quality edit, professional output, quick turnaround",
    },
    "fal-ai/reve/remix": {
        "guidance_scale": 3.5,
        "num_inference_steps": 28,
        "best_for": ["multi-image compositions", "creative remixing", "combining elements"],
        "marketing_prompt": "Creative multi-image remix, seamless composition, marketing-ready",
    },
    "fal-ai/reve/fast/remix": {
        "guidance_scale": 3.5,
        "num_inference_steps": 20,
        "best_for": ["fast multi-image compositions", "quick remixing", "rapid iterations"],
        "marketing_prompt": "Fast creative remix, professional quality, quick output",
    },
    
    # ByteDance Seedream V4.5
    "fal-ai/bytedance/seedream/v4.5/text-to-image": {
        "guidance_scale": 5.0,
        "num_inference_steps": 30,
        "best_for": ["stylized content", "creative campaigns", "social media"],
        "marketing_prompt": "Stylized marketing visual, creative aesthetic, social media optimized, trending style",
    },
    "fal-ai/bytedance/seedream/v4.5/edit": {
        "guidance_scale": 5.0,
        "num_inference_steps": 30,
        "best_for": ["image editing", "multi-image compositions", "creative edits"],
        "marketing_prompt": "Creative image edit, stylized output, social media ready",
    },
    
    # ByteDance Seedream V4 (4K Support)
    "fal-ai/bytedance/seedream/v4/text-to-image": {
        "guidance_scale": 5.0,
        "num_inference_steps": 30,
        "best_for": ["high resolution images", "4K content", "stylized art"],
        "marketing_prompt": "High resolution marketing visual, 4K quality, premium stylized content",
    },
    "fal-ai/bytedance/seedream/v4/edit": {
        "guidance_scale": 5.0,
        "num_inference_steps": 30,
        "best_for": ["multi-image editing", "high resolution edits", "complex compositions"],
        "marketing_prompt": "High resolution edit, 4K quality, professional multi-image composition",
    },
    
    # ByteDance Dreamina V3.1 (Portrait-focused)
    "fal-ai/bytedance/dreamina/v3.1/text-to-image": {
        "guidance_scale": 5.0,
        "num_inference_steps": 30,
        "best_for": ["portrait photography", "selfies", "photorealistic humans", "korean style"],
        "marketing_prompt": "Photorealistic portrait, professional headshot quality, natural lighting, marketing-ready",
    },
    
    # Z-Image Turbo
    "fal-ai/z-image/turbo": {
        "guidance_scale": 0,  # Uses num_inference_steps (1-8)
        "num_inference_steps": 8,
        "best_for": ["quick iterations", "rapid brainstorming", "mockups"],
        "marketing_prompt": "Fast marketing visual, professional quality, quick turnaround",
    },
    
    # Ideogram V3 - Typography & Style Presets
    "fal-ai/ideogram/v3": {
        "guidance_scale": 0,  # Uses style presets instead
        "num_inference_steps": 0,
        "best_for": ["typography", "logos", "text in images", "style presets", "posters"],
        "marketing_prompt": "Perfect text rendering, clean typography, professional marketing visual with accurate lettering",
    },
    "fal-ai/ideogram/v3/reframe": {
        "guidance_scale": 0,
        "num_inference_steps": 0,
        "best_for": ["image reframing", "aspect ratio change", "canvas expansion", "outpainting"],
        "marketing_prompt": "Seamless canvas expansion, intelligent background fill, perfect aspect ratio adjustment",
    },
    
    # Upscaling Models
    "fal-ai/creative-upscaler": {
        "guidance_scale": 0,
        "num_inference_steps": 0,
        "best_for": ["creative upscaling", "AI enhancement", "resolution boost"],
        "marketing_prompt": "Enhanced resolution, creative detail enhancement, print-ready quality",
    },
    "fal-ai/clarity-upscaler": {
        "guidance_scale": 0,
        "num_inference_steps": 0,
        "best_for": ["clarity enhancement", "sharpness boost", "detail preservation"],
        "marketing_prompt": "Crystal clear upscale, sharp details, professional clarity",
    },
    "fal-ai/recraft/upscale/creative": {
        "guidance_scale": 0,
        "num_inference_steps": 0,
        "best_for": ["creative upscaling", "artistic enhancement", "vector-style upscale"],
        "marketing_prompt": "Creative upscale, artistic enhancement, high resolution output",
    },
    
    # Object & Text Removal
    "fal-ai/object-removal": {
        "guidance_scale": 0,
        "num_inference_steps": 0,
        "best_for": ["object removal", "cleanup", "unwanted element removal"],
        "marketing_prompt": "Clean removal, seamless background fill, professional result",
    },
    "fal-ai/bria/eraser": {
        "guidance_scale": 0,
        "num_inference_steps": 0,
        "best_for": ["commercial-safe removal", "object erasing", "clean edits"],
        "marketing_prompt": "Commercial-safe erasure, clean professional result",
    },
    "fal-ai/text-removal": {
        "guidance_scale": 0,
        "num_inference_steps": 0,
        "best_for": ["text removal", "watermark removal", "overlay cleanup"],
        "marketing_prompt": "Clean text removal, seamless background restoration",
    },
    
    # Style & Background
    "fal-ai/style-transfer": {
        "guidance_scale": 0,
        "num_inference_steps": 0,
        "best_for": ["style transfer", "artistic transformation", "creative styling"],
        "marketing_prompt": "Artistic style transfer, creative transformation, unique visual",
    },
    "fal-ai/background-change": {
        "guidance_scale": 0,
        "num_inference_steps": 0,
        "best_for": ["background replacement", "scene change", "context modification"],
        "marketing_prompt": "Professional background replacement, seamless integration",
    },
    "fal-ai/add-background": {
        "guidance_scale": 0,
        "num_inference_steps": 0,
        "best_for": ["add background", "subject extraction", "scene creation"],
        "marketing_prompt": "Professional background addition, seamless subject integration",
    },
    "fal-ai/relighting": {
        "guidance_scale": 0,
        "num_inference_steps": 0,
        "best_for": ["lighting adjustment", "mood change", "illumination correction"],
        "marketing_prompt": "Professional relighting, mood-appropriate illumination, studio quality",
    },
    
    # Character Multiple Angles (Camera control)
    "fal-ai/qwen-image-edit-2509-lora-gallery/multiple-angles": {
        "guidance_scale": 1,
        "num_inference_steps": 6,
        "best_for": ["multiple angles", "character turns", "3D view generation", "product angles"],
        "marketing_prompt": "Multiple camera angles, consistent character, 360-degree view generation",
    },
    
    # GPT Image 1.5 - OpenAI image model via Fal
    "fal-ai/gpt-image-1.5": {
        "guidance_scale": 0,
        "num_inference_steps": 0,
        "best_for": ["photorealistic images", "transparent backgrounds", "product shots", "logos"],
        "marketing_prompt": "GPT-powered image generation, photorealistic quality, clean transparent backgrounds",
    },
    "fal-ai/gpt-image-1.5/edit": {
        "guidance_scale": 0,
        "num_inference_steps": 0,
        "best_for": ["image editing", "transparent backgrounds", "product modifications", "style changes"],
        "marketing_prompt": "GPT-powered image editing, seamless modifications, transparent background support",
    },
}

# Default settings for models not in the list
DEFAULT_MODEL_SETTINGS = {
    "guidance_scale": 3.5,
    "num_inference_steps": 28,
    "best_for": ["general marketing"],
    "marketing_prompt": "Professional marketing visual, commercial quality, engaging design",
}


# =============================================================================
# MARKETING USE CASE PROMPTS
# =============================================================================

MARKETING_PROMPTS = {
    "social_media_post": {
        "instagram": "Eye-catching Instagram post, vibrant colors, lifestyle aesthetic, scroll-stopping visual, square format optimized, trending style",
        "facebook": "Engaging Facebook ad creative, clear message, professional quality, click-worthy design, social proof elements",
        "twitter": "Bold Twitter/X visual, attention-grabbing, concise messaging space, high contrast, shareable content",
        "linkedin": "Professional LinkedIn content, corporate aesthetic, thought leadership visual, business-appropriate, credibility-building",
        "tiktok": "Trendy TikTok thumbnail, Gen-Z aesthetic, bold colors, dynamic composition, viral potential",
        "pinterest": "Pinterest-optimized vertical image, aspirational lifestyle, save-worthy design, rich colors, DIY/inspiration focus",
    },
    "advertising": {
        "display_ad": "High-converting display ad, clear CTA space, brand colors, attention-grabbing, banner-ready composition",
        "hero_banner": "Premium hero banner, full-width impact, storytelling visual, brand hero image, above-fold optimized",
        "product_ad": "Product-focused advertisement, benefit-highlighting, purchase-driving visual, e-commerce optimized",
        "retargeting": "Retargeting ad creative, reminder visual, urgency elements, conversion-focused design",
        "video_thumbnail": "Click-worthy video thumbnail, curiosity-inducing, play button friendly, YouTube/social optimized",
    },
    "e_commerce": {
        "product_listing": "E-commerce product photo, white background, multiple angles implied, detail-focused, Amazon/Shopify ready",
        "lifestyle_shot": "Lifestyle product photography, in-context usage, aspirational setting, purchase-inspiring",
        "comparison": "Product comparison visual, side-by-side layout, feature highlighting, decision-helping design",
        "bundle": "Product bundle showcase, value visualization, complementary items, upsell-friendly composition",
    },
    "branding": {
        "logo_showcase": "Brand logo presentation, clean background, professional display, brand guidelines compliant",
        "brand_story": "Brand storytelling visual, emotional connection, values-aligned imagery, authentic feel",
        "team_photo": "Professional team/company photo, approachable yet professional, trust-building, about-page ready",
        "office_culture": "Company culture showcase, authentic workplace, employer branding, recruitment-friendly",
    },
    "seasonal": {
        "holiday": "Holiday marketing visual, festive elements, seasonal colors, celebration mood, gift-giving context",
        "summer": "Summer campaign visual, bright sunny aesthetic, outdoor lifestyle, vacation vibes, seasonal products",
        "winter": "Winter marketing image, cozy aesthetic, holiday shopping mood, warm tones, seasonal appeal",
        "black_friday": "Black Friday sale visual, urgency-inducing, deal-highlighting, shopping excitement, limited-time feel",
    },
    "content_marketing": {
        "blog_header": "Blog post header image, topic-relevant visual, readable text space, SEO-friendly, shareable",
        "infographic": "Infographic-style visual, data visualization, educational content, shareable format, brand colors",
        "case_study": "Case study visual, results-highlighting, professional presentation, credibility-building",
        "whitepaper": "Whitepaper cover design, professional B2B aesthetic, thought leadership, download-worthy",
    },
}


class ImageSize(str, Enum):
    """Predefined image sizes"""
    SQUARE_512 = "square"           # 512x512
    SQUARE_1024 = "square_hd"       # 1024x1024
    PORTRAIT_3_4 = "portrait_3_4"   # 768x1024
    PORTRAIT_9_16 = "portrait_9_16" # 576x1024
    LANDSCAPE_4_3 = "landscape_4_3" # 1024x768
    LANDSCAPE_16_9 = "landscape_16_9" # 1024x576
    LANDSCAPE_21_9 = "landscape_21_9" # 1536x640
    PORTRAIT_9_21 = "portrait_9_21"   # 640x1536


# Size mappings for custom dimensions
SIZE_DIMENSIONS = {
    "square": {"width": 512, "height": 512},
    "square_hd": {"width": 1024, "height": 1024},
    "portrait_3_4": {"width": 768, "height": 1024},
    "portrait_9_16": {"width": 576, "height": 1024},
    "landscape_4_3": {"width": 1024, "height": 768},
    "landscape_16_9": {"width": 1024, "height": 576},
    "landscape_21_9": {"width": 1536, "height": 640},
    "portrait_9_21": {"width": 640, "height": 1536},
}


@dataclass
class GenerationResult:
    """Result of image generation"""
    success: bool
    images: List[Dict[str, Any]]  # List of {url, width, height, content_type}
    seed: Optional[int] = None
    prompt: Optional[str] = None
    model: Optional[str] = None
    error: Optional[str] = None
    timings: Optional[Dict] = None
    cost_estimate: Optional[float] = None


class GrovioImageGenerator:
    """
    Grovio AI Image Generator for Marketing Creatives
    
    50+ optimized models for social media, advertising, e-commerce, and branding.
    Features auto-settings and marketing prompt enhancement.
    
    Usage:
        generator = GrovioImageGenerator(api_key="your-api-key")
        result = await generator.generate(
            prompt="A beautiful sunset over mountains",
            model=ImageModel.FLUX2_DEV,
            size=ImageSize.LANDSCAPE_16_9,
            num_images=1
        )
    """
    
    # Model pricing per image (1024x1024 = 1 megapixel) - Updated from fal.ai
    MODEL_PRICING = {
        # FLUX 2 Pro - Premium ($0.03 for first MP, $0.015 each additional)
        "fal-ai/flux-2-pro": 0.03,
        "fal-ai/flux-2-pro/edit": 0.03,
        # FLUX 2 Dev ($0.012/MP)
        "fal-ai/flux-2": 0.012,
        "fal-ai/flux-2/edit": 0.024,  # Input + Output MP
        # Google Gemini 3 Pro ($0.15/image)
        "fal-ai/gemini-3-pro-image-preview": 0.15,
        "fal-ai/gemini-3-pro-image-preview/edit": 0.15,
        # Reve ($0.04/image)
        "fal-ai/reve/text-to-image": 0.04,
        "fal-ai/reve/edit": 0.04,
        "fal-ai/reve/fast/edit": 0.04,
        "fal-ai/reve/remix": 0.04,
        "fal-ai/reve/fast/remix": 0.04,
        # ByteDance Seedream V4.5 ($0.04/image)
        "fal-ai/bytedance/seedream/v4.5/text-to-image": 0.04,
        "fal-ai/bytedance/seedream/v4.5/edit": 0.04,
        # ByteDance Seedream V4 ($0.03/image)
        "fal-ai/bytedance/seedream/v4/text-to-image": 0.03,
        "fal-ai/bytedance/seedream/v4/edit": 0.03,
        # ByteDance Dreamina V3.1 ($0.027/image)
        "fal-ai/bytedance/dreamina/v3.1/text-to-image": 0.027,
        # Z-Image Turbo ($0.005/MP - ultra cheap)
        "fal-ai/z-image/turbo": 0.005,
        # Ideogram V3 ($0.08/image)
        "fal-ai/ideogram/v3": 0.08,
        "fal-ai/ideogram/v3/reframe": 0.08,
        # Upscalers
        "fal-ai/creative-upscaler": 0.02,
        "fal-ai/clarity-upscaler": 0.02,
        "fal-ai/recraft/upscale/creative": 0.02,
        # Object & Text Removal
        "fal-ai/object-removal": 0.02,
        "fal-ai/bria/eraser": 0.02,
        "fal-ai/text-removal": 0.02,
        # Style & Background
        "fal-ai/style-transfer": 0.02,
        "fal-ai/background-change": 0.02,
        "fal-ai/add-background": 0.02,
        "fal-ai/relighting": 0.02,
        # Character Multiple Angles
        "fal-ai/qwen-image-edit-2509-lora-gallery/multiple-angles": 0.03,
        # GPT Image 1.5 - OpenAI model via Fal
        "fal-ai/gpt-image-1.5": 0.05,
        "fal-ai/gpt-image-1.5/edit": 0.05,
    }
    
    # Models that support image editing (require image_url input)
    EDIT_MODELS = {
        "fal-ai/flux-2-pro/edit",
        "fal-ai/flux-2/edit",
        "fal-ai/gemini-3-pro-image-preview/edit",
        "fal-ai/reve/edit",
        "fal-ai/reve/fast/edit",
        "fal-ai/reve/remix",
        "fal-ai/reve/fast/remix",
        "fal-ai/bytedance/seedream/v4.5/edit",
        "fal-ai/bytedance/seedream/v4/edit",
        # Ideogram V3 Reframe (requires input image)
        "fal-ai/ideogram/v3/reframe",
        # GPT Image 1.5 Edit (requires input image)
        "fal-ai/gpt-image-1.5/edit",
        # Enhancement models (all require input image)
        "fal-ai/creative-upscaler",
        "fal-ai/clarity-upscaler",
        "fal-ai/recraft/upscale/creative",
        "fal-ai/object-removal",
        "fal-ai/bria/eraser",
        "fal-ai/text-removal",
        "fal-ai/style-transfer",
        "fal-ai/background-change",
        "fal-ai/add-background",
        "fal-ai/relighting",
        # Character Multiple Angles
        "fal-ai/qwen-image-edit-2509-lora-gallery/multiple-angles",
    }
    
    # Models that support text-to-image generation
    GENERATION_MODELS = {
        "fal-ai/flux-2-pro",
        "fal-ai/flux-2",
        "fal-ai/gemini-3-pro-image-preview",
        "fal-ai/reve/text-to-image",
        "fal-ai/bytedance/seedream/v4.5/text-to-image",
        "fal-ai/bytedance/seedream/v4/text-to-image",
        "fal-ai/bytedance/dreamina/v3.1/text-to-image",
        "fal-ai/z-image/turbo",
        "fal-ai/ideogram/v3",
    }
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Grovio AI Image Generator
        
        Args:
            api_key: Grovio AI API key. If not provided, uses GROVIO_API_KEY or FAL_KEY env variable
        """
        if not GROVIO_AI_AVAILABLE:
            raise ImportError("Grovio AI client not installed. Run: pip install fal-client")
        
        # Set API key (supports both GROVIO_API_KEY and FAL_KEY for backward compatibility)
        self.api_key = api_key or os.getenv("GROVIO_API_KEY") or os.getenv("FAL_KEY")
        if not self.api_key:
            raise ValueError("GROVIO_API_KEY not provided. Set GROVIO_API_KEY environment variable or pass api_key parameter")
        
        # Configure client
        fal_client.api_key = self.api_key
        print("✅ Grovio AI Image Generator initialized")
    
    def requires_image(self, model_id: str) -> bool:
        """
        Check if a model requires an input image (like editing, inpainting, try-on).
        
        Args:
            model_id: The full fal.ai model ID (e.g. 'fal-ai/flux-2-pro/edit')
            
        Returns:
            True if the model requires an input image, False otherwise
        """
        return model_id in self.EDIT_MODELS

    
    def _calculate_megapixels(self, width: int, height: int) -> float:
        """Calculate megapixels from dimensions"""
        return (width * height) / 1_000_000
    
    def _estimate_cost(self, model: str, width: int, height: int, num_images: int) -> float:
        """Estimate cost for generation"""
        price_per_unit = self.MODEL_PRICING.get(model, 0.025)
        
        # Seedream is per-image pricing
        if "seedream" in model.lower():
            return price_per_unit * num_images
        
        # Others are per-megapixel
        mp = self._calculate_megapixels(width, height)
        return price_per_unit * mp * num_images
    
    def get_model_settings(self, model: ImageModel) -> Dict[str, Any]:
        """
        Get optimal settings for a model including guidance_scale, steps, and marketing info.
        
        Args:
            model: ImageModel enum
            
        Returns:
            Dict with guidance_scale, num_inference_steps, best_for, marketing_prompt
        """
        return MODEL_SETTINGS.get(model.value, DEFAULT_MODEL_SETTINGS)
    
    def get_marketing_prompt(self, use_case: str, platform: str = None) -> str:
        """
        Get marketing-optimized prompt suffix for a specific use case.
        
        Args:
            use_case: Category like 'social_media_post', 'advertising', 'e_commerce', etc.
            platform: Specific platform like 'instagram', 'facebook', 'display_ad', etc.
            
        Returns:
            Marketing prompt string to append to user prompt
        """
        if use_case in MARKETING_PROMPTS:
            category = MARKETING_PROMPTS[use_case]
            if platform and platform in category:
                return category[platform]
            # Return first available if platform not specified
            return list(category.values())[0]
        return DEFAULT_MODEL_SETTINGS["marketing_prompt"]
    
    def enhance_prompt_for_marketing(
        self, 
        prompt: str, 
        model: ImageModel,
        use_case: str = None,
        platform: str = None
    ) -> str:
        """
        Enhance user prompt with marketing-optimized suffix based on model and use case.
        
        Args:
            prompt: User's original prompt
            model: Selected model
            use_case: Marketing use case (social_media_post, advertising, etc.)
            platform: Specific platform (instagram, facebook, etc.)
            
        Returns:
            Enhanced prompt with marketing optimization
        """
        # Get model-specific marketing prompt
        settings = self.get_model_settings(model)
        model_prompt = settings.get("marketing_prompt", "")
        
        # Get use-case specific prompt if provided
        use_case_prompt = ""
        if use_case:
            use_case_prompt = self.get_marketing_prompt(use_case, platform)
        
        # Combine prompts
        enhanced = prompt
        if use_case_prompt:
            enhanced = f"{prompt}, {use_case_prompt}"
        elif model_prompt:
            enhanced = f"{prompt}, {model_prompt}"
        
        return enhanced
    
    def get_recommended_model(self, use_case: str) -> ImageModel:
        """
        Get recommended model for a specific marketing use case.
        
        Args:
            use_case: Marketing use case
            
        Returns:
            Recommended ImageModel
        """
        recommendations = {
            # Social Media - Use FLUX2_DEV for quality, Z_IMAGE_TURBO for speed
            "instagram": ImageModel.FLUX2_DEV,
            "facebook": ImageModel.FLUX2_DEV,
            "twitter": ImageModel.Z_IMAGE_TURBO,
            "linkedin": ImageModel.FLUX2_DEV,
            "tiktok": ImageModel.Z_IMAGE_TURBO,
            "pinterest": ImageModel.FLUX2_DEV,
            
            # Advertising - Use FLUX2_PRO for premium, FLUX2_DEV for quality
            "display_ad": ImageModel.FLUX2_PRO,
            "hero_banner": ImageModel.FLUX2_PRO,
            "product_ad": ImageModel.FLUX2_PRO,
            "video_thumbnail": ImageModel.GEMINI3_PRO,
            
            # E-commerce
            "product_listing": ImageModel.FLUX2_PRO,
            "lifestyle_shot": ImageModel.SEEDREAM_V45,
            "product_photoshoot": ImageModel.FLUX2_PRO,
            
            # Typography & Logos - Use Gemini 3 Pro for text
            "logo": ImageModel.GEMINI3_PRO,
            "poster": ImageModel.GEMINI3_PRO,
            "banner": ImageModel.GEMINI3_PRO,
            "text_heavy": ImageModel.GEMINI3_PRO,
            
            # Stylized Content
            "stylized": ImageModel.SEEDREAM_V45,
            "creative": ImageModel.SEEDREAM_V45,
            "artistic": ImageModel.SEEDREAM_V45,
            
            # Quick iterations - Use Z-Image Turbo
            "quick": ImageModel.Z_IMAGE_TURBO,
            "brainstorm": ImageModel.Z_IMAGE_TURBO,
            "mockup": ImageModel.Z_IMAGE_TURBO,
            
            # High quality & Premium - Use FLUX2_PRO
            "premium": ImageModel.FLUX2_PRO,
            "print": ImageModel.FLUX2_PRO,
            "billboard": ImageModel.FLUX2_PRO,
            "magazine": ImageModel.FLUX2_PRO,
            "editorial": ImageModel.FLUX2_PRO,
        }
        
        return recommendations.get(use_case.lower(), ImageModel.FLUX2_DEV)
    
    async def generate(
        self,
        prompt: str,
        model: ImageModel = ImageModel.FLUX2_DEV,
        size: ImageSize = ImageSize.SQUARE_1024,
        num_images: int = 1,
        guidance_scale: float = 3.5,
        num_inference_steps: int = 28,
        seed: Optional[int] = None,
        negative_prompt: Optional[str] = None,
        enable_safety_checker: bool = True,
        output_format: str = "jpeg",
        lora_url: Optional[str] = None,
        lora_scale: float = 1.0,
        custom_width: Optional[int] = None,
        custom_height: Optional[int] = None,
    ) -> GenerationResult:
        """
        Generate images using fal.ai API
        
        Args:
            prompt: Text description of the image to generate
            model: Model to use for generation
            size: Predefined image size
            num_images: Number of images to generate (1-4)
            guidance_scale: How closely to follow the prompt (1.0-20.0)
            num_inference_steps: Number of denoising steps (1-50)
            seed: Random seed for reproducibility
            negative_prompt: What to avoid in the image
            enable_safety_checker: Enable NSFW filter
            output_format: Output format (jpeg or png)
            lora_url: URL to LoRA weights (for flux-lora model)
            lora_scale: LoRA influence scale (0.0-1.0)
            custom_width: Custom width (overrides size preset)
            custom_height: Custom height (overrides size preset)
            
        Returns:
            GenerationResult with generated images or error
        """
        try:
            # Get dimensions
            if custom_width and custom_height:
                width, height = custom_width, custom_height
            else:
                dims = SIZE_DIMENSIONS.get(size.value, SIZE_DIMENSIONS["square_hd"])
                width, height = dims["width"], dims["height"]
            
            # Build arguments based on model
            model_id = model.value
            
            # ====================================================================
            # BYTEDANCE SEEDREAM ROUTING
            # Route Seedream models through Bytedance API instead of fal.ai
            # ====================================================================
            if BYTEDANCE_AVAILABLE and "seedream" in model_id.lower():
                print(f"🔀 Routing {model_id} through Bytedance API")
                bytedance_client = get_bytedance_client()
                if bytedance_client:
                    # Convert model_id to short name for Bytedance client
                    short_model = bytedance_client.FAL_TO_SHORT.get(model_id, "seedream-v4.5")
                    
                    bd_result = await bytedance_client.generate_image(
                        prompt=prompt,
                        model=short_model,
                        width=width,
                        height=height,
                        num_images=num_images,
                        seed=seed,
                    )
                    
                    if bd_result.success:
                        cost_estimate = self._estimate_cost(model_id, width, height, num_images)
                        return GenerationResult(
                            success=True,
                            images=bd_result.images,
                            seed=bd_result.seed,
                            prompt=bd_result.revised_prompt or prompt,
                            model=model_id,
                            cost_estimate=cost_estimate,
                        )
                    else:
                        print(f"⚠️ Bytedance failed, falling back to fal.ai: {bd_result.error}")
                        # Fall through to fal.ai
            # ====================================================================
            # END BYTEDANCE ROUTING
            # ====================================================================
            
            arguments = {
                "prompt": prompt,
                "image_size": {"width": width, "height": height},
                "num_images": min(num_images, 4),  # Max 4 images per request
                "enable_safety_checker": enable_safety_checker,
                "output_format": output_format,
            }
            
            # Model-specific parameters
            # GPT Image 1.5 has unique API requirements
            if "gpt-image" in model_id:
                # GPT Image 1.5 uses string format for image_size
                # Valid sizes: 1024x1024 (Square), 1536x1024 (Landscape), 1024x1536 (Portrait)
                # Map requested dimensions to closest valid GPT Image size
                size_str = f"{width}x{height}"

                # Exact match check first
                if size_str in ["1024x1024", "1536x1024", "1024x1536"]:
                    gpt_size = size_str
                else:
                    # Map to closest valid size based on aspect ratio
                    aspect_ratio = width / height
                    if aspect_ratio > 1.2:  # Landscape
                        gpt_size = "1536x1024"
                    elif aspect_ratio < 0.8:  # Portrait
                        gpt_size = "1024x1536"
                    else:  # Square-ish
                        gpt_size = "1024x1024"

                arguments = {
                    "prompt": prompt,
                    "image_size": gpt_size,
                    "num_images": min(num_images, 4),
                    "quality": "high",  # low, medium, high
                    "background": "auto",  # auto, transparent, opaque
                    "output_format": output_format,
                }
            elif "schnell" in model_id:
                # Schnell uses fewer steps (1-4)
                arguments["num_inference_steps"] = min(num_inference_steps, 4)
            else:
                arguments["num_inference_steps"] = num_inference_steps
                arguments["guidance_scale"] = guidance_scale
            
            # Add optional parameters
            if seed is not None:
                arguments["seed"] = seed
            
            if negative_prompt and "flux" not in model_id.lower():
                # FLUX models don't support negative prompts well
                arguments["negative_prompt"] = negative_prompt
            
            # LoRA support removed - FLUX_LORA model no longer available
            # if lora_url:
            #     arguments["loras"] = [{"path": lora_url, "scale": lora_scale}]
            
            # Estimate cost
            cost_estimate = self._estimate_cost(model_id, width, height, num_images)
            
            print(f"🎨 Generating {num_images} image(s) with {model_id}")
            print(f"   Prompt: {prompt[:50]}...")
            print(f"   Size: {width}x{height}")
            print(f"   Estimated cost: ${cost_estimate:.4f}")
            
            # Call fal.ai API (async)
            result = await asyncio.to_thread(
                fal_client.subscribe,
                model_id,
                arguments=arguments
            )
            
            # Parse result
            images = []
            for img in result.get("images", []):
                images.append({
                    "url": img.get("url"),
                    "width": img.get("width", width),
                    "height": img.get("height", height),
                    "content_type": f"image/{output_format}",
                })
            
            return GenerationResult(
                success=True,
                images=images,
                seed=result.get("seed"),
                prompt=result.get("prompt", prompt),
                model=model_id,
                timings=result.get("timings"),
                cost_estimate=cost_estimate,
            )
            
        except Exception as e:
            print(f"❌ Generation failed: {e}")
            return GenerationResult(
                success=False,
                images=[],
                error=str(e),
                model=model.value,
            )
    
    def is_edit_model(self, model: ImageModel) -> bool:
        """Check if model supports image editing"""
        return model.value in self.EDIT_MODELS
    
    def is_edit_model_by_id(self, model_id: str) -> bool:
        """Check if model supports image editing by model ID string"""
        return model_id in self.EDIT_MODELS
    
    def is_generation_model(self, model: ImageModel) -> bool:
        """Check if model supports text-to-image generation"""
        return model.value in self.GENERATION_MODELS
    
    async def edit_image(
        self,
        image_url: str,
        prompt: str,
        model: ImageModel = ImageModel.FLUX2_DEV_EDIT,
        image_urls: Optional[List[str]] = None,  # Multiple images support
        style_image_url: Optional[str] = None,  # Style reference for style transfer
        garment_image_url: Optional[str] = None,  # Garment for virtual try-on
        mask_url: Optional[str] = None,
        strength: float = 0.85,
        num_inference_steps: int = 28,
        guidance_scale: float = 3.5,
        num_images: int = 1,  # Number of output images (1-4)
        seed: Optional[int] = None,
        output_format: str = "jpeg",
        size: ImageSize = None,  # Output image size
        **kwargs  # Accept and ignore extra kwargs like enable_safety_checker
    ) -> GenerationResult:
        """
        Edit an existing image using fal.ai image editing models.
        Supports multiple image inputs for various editing scenarios.
        
        Args:
            image_url: Primary URL of the image to edit
            prompt: Description of the edit to make
            model: Image editing model to use
            image_urls: List of image URLs for multi-image editing
            style_image_url: Style reference image for style transfer models
            garment_image_url: Garment image for virtual try-on models
            mask_url: Optional mask URL for inpainting (white = edit area)
            strength: How much to change the image (0.0-1.0)
            num_inference_steps: Number of denoising steps
            guidance_scale: How closely to follow the prompt
            seed: Random seed for reproducibility
            output_format: Output format (jpeg or png)
            
        Returns:
            GenerationResult with edited image
        """
        try:
            model_id = model.value
            
            # ====================================================================
            # BYTEDANCE SEEDREAM EDIT ROUTING
            # Route Seedream edit models through Bytedance API instead of fal.ai
            # ====================================================================
            if BYTEDANCE_AVAILABLE and "seedream" in model_id.lower():
                print(f"🔀 Routing {model_id} edit through Bytedance API")
                bytedance_client = get_bytedance_client()
                if bytedance_client:
                    # Convert model_id to short name
                    short_model = bytedance_client.FAL_TO_SHORT.get(model_id, "seedream-v4.5-edit")
                    
                    # Get dimensions from size if provided
                    width, height = 1024, 1024
                    if size:
                        dims = SIZE_DIMENSIONS.get(size.value, SIZE_DIMENSIONS["square_hd"])
                        width, height = dims["width"], dims["height"]
                    
                    # Collect all image URLs
                    all_urls = image_urls if image_urls else ([image_url] if image_url else [])
                    
                    bd_result = await bytedance_client.edit_image(
                        image_url=all_urls[0] if all_urls else "",
                        prompt=prompt,
                        model=short_model,
                        image_urls=all_urls,
                        width=width,
                        height=height,
                    )
                    
                    if bd_result.success:
                        cost_estimate = self.MODEL_PRICING.get(model_id, 0.04)
                        return GenerationResult(
                            success=True,
                            images=bd_result.images,
                            seed=bd_result.seed,
                            prompt=prompt,
                            model=model_id,
                            cost_estimate=cost_estimate,
                        )
                    else:
                        print(f"⚠️ Bytedance edit failed, falling back to fal.ai: {bd_result.error}")
                        # Fall through to fal.ai
            # ====================================================================
            # END BYTEDANCE EDIT ROUTING
            # ====================================================================
            
            # Build base arguments
            arguments = {
                "prompt": prompt,
                "output_format": output_format,
            }
            
            # Determine which image URLs to use
            # Priority: image_urls (multiple) > single image_url
            all_image_urls = []
            if image_urls and len(image_urls) > 0:
                all_image_urls = image_urls
            elif image_url:
                all_image_urls = [image_url]
            
            # Models that use image_urls (array) instead of image_url
            uses_image_urls = any(x in model_id.lower() for x in [
                "nano-banana", "flux-2", "seedream", "ideogram", "gemini-3-pro", "gpt-image"
            ])
            
            if uses_image_urls:
                arguments["image_urls"] = all_image_urls
            else:
                arguments["image_url"] = all_image_urls[0] if all_image_urls else image_url
            
            # Handle style transfer models - add style reference image
            if style_image_url:
                if "style-transfer" in model_id.lower():
                    arguments["style_image_url"] = style_image_url
                elif uses_image_urls and len(all_image_urls) == 1:
                    # Some models accept style as second image
                    arguments["image_urls"].append(style_image_url)
                else:
                    arguments["style_reference_url"] = style_image_url
                print(f"   Style reference: {style_image_url[:50]}...")
            
            # Handle virtual try-on models - add garment image
            if garment_image_url:
                if "tryon" in model_id.lower() or "try-on" in model_id.lower():
                    # Different try-on models use different parameter names
                    if "fashn" in model_id.lower():
                        arguments["garment_image_url"] = garment_image_url
                    elif "kling" in model_id.lower():
                        arguments["cloth_image_url"] = garment_image_url
                    else:
                        arguments["garment_url"] = garment_image_url
                    print(f"   Garment image: {garment_image_url[:50]}...")
            
            # Log multi-image info
            if len(all_image_urls) > 1:
                print(f"   Multiple images: {len(all_image_urls)} images provided")
            
            # GPT Image 1.5 Edit uses unique API format
            if "gpt-image" in model_id.lower():
                arguments = {
                    "prompt": prompt,
                    "image_urls": all_image_urls,
                    "image_size": "auto",  # auto, 1024x1024, 1536x1024, 1024x1536
                    "quality": "high",  # low, medium, high
                    "input_fidelity": "high",  # low, high
                    "background": "auto",  # auto, transparent, opaque
                    "num_images": min(num_images, 4),
                    "output_format": output_format,
                }
            # Kontext models use different parameters
            elif "kontext" in model_id.lower():
                arguments["guidance_scale"] = guidance_scale
                arguments["num_inference_steps"] = num_inference_steps
                if seed is not None:
                    arguments["seed"] = seed
            
            # Inpainting models need mask
            elif "inpainting" in model_id.lower():
                if mask_url:
                    arguments["mask_url"] = mask_url
                arguments["strength"] = strength
                arguments["guidance_scale"] = guidance_scale
                arguments["num_inference_steps"] = num_inference_steps
                if seed is not None:
                    arguments["seed"] = seed
            
            # Upscaler models (creative-upscaler, clarity-upscaler, recraft-upscale)
            elif "upscale" in model_id.lower():
                scale_factor = kwargs.get("scale", 2)  # Default to 2x
                if scale_factor not in [1, 2, 4]:
                    scale_factor = 2
                arguments = {
                    "image_url": all_image_urls[0] if all_image_urls else image_url,
                    "scale": scale_factor,
                    "output_format": output_format,
                }
                if "creative" in model_id.lower():
                    arguments["creativity"] = 0.5  # Balanced creativity
                    arguments["detail"] = 1.0  # Full detail
                    arguments["resemblance"] = 1.0  # High resemblance
            
            # FLUX 2 edit models
            elif "flux-2" in model_id.lower() and "edit" in model_id.lower():
                arguments["guidance_scale"] = guidance_scale
                arguments["num_inference_steps"] = num_inference_steps
                arguments["num_images"] = min(num_images, 4)  # Max 4 images
                # Add image_size if specified
                if size is not None:
                    # Use SIZE_DIMENSIONS to get width/height
                    dims = SIZE_DIMENSIONS.get(size, SIZE_DIMENSIONS[ImageSize.SQUARE_1024])
                    arguments["image_size"] = {"width": dims["width"], "height": dims["height"]}
                if seed is not None:
                    arguments["seed"] = seed
            
            # Nano Banana / Seedream / Ideogram edit
            elif any(x in model_id.lower() for x in ["nano-banana", "seedream", "ideogram"]):
                arguments["guidance_scale"] = guidance_scale
                if seed is not None:
                    arguments["seed"] = seed
            
            # Default edit parameters
            else:
                arguments["num_inference_steps"] = num_inference_steps
                arguments["guidance_scale"] = guidance_scale
                if seed is not None:
                    arguments["seed"] = seed
            
            # Estimate cost (assume 1MP for edits)
            cost_estimate = self.MODEL_PRICING.get(model_id, 0.025)
            
            print(f"✏️ Editing image with {model_id}")
            print(f"   Prompt: {prompt[:50]}...")
            print(f"   Estimated cost: ${cost_estimate:.4f}")
            
            # Call fal.ai API
            result = await asyncio.to_thread(
                fal_client.subscribe,
                model_id,
                arguments=arguments
            )
            
            # Parse result
            images = []
            # Handle different response formats
            if "images" in result:
                for img in result["images"]:
                    images.append({
                        "url": img.get("url"),
                        "width": img.get("width", 1024),
                        "height": img.get("height", 1024),
                        "content_type": f"image/{output_format}",
                    })
            elif "image" in result:
                img = result["image"]
                if isinstance(img, dict):
                    images.append({
                        "url": img.get("url"),
                        "width": img.get("width", 1024),
                        "height": img.get("height", 1024),
                        "content_type": f"image/{output_format}",
                    })
                elif isinstance(img, str):
                    images.append({
                        "url": img,
                        "width": 1024,
                        "height": 1024,
                        "content_type": f"image/{output_format}",
                    })
            
            return GenerationResult(
                success=True,
                images=images,
                seed=result.get("seed"),
                prompt=prompt,
                model=model_id,
                timings=result.get("timings"),
                cost_estimate=cost_estimate,
            )
            
        except Exception as e:
            print(f"❌ Image editing failed: {e}")
            return GenerationResult(
                success=False,
                images=[],
                error=str(e),
                model=model.value,
            )
    
    async def generate_or_edit(
        self,
        prompt: str,
        model: ImageModel = ImageModel.FLUX2_DEV,
        image_url: str = None,
        image_urls: List[str] = None,  # Multiple images support
        style_image_url: str = None,  # Style reference for style transfer
        garment_image_url: str = None,  # Garment for virtual try-on
        mask_url: str = None,
        size: ImageSize = ImageSize.SQUARE_1024,
        num_images: int = 1,
        guidance_scale: float = 3.5,
        num_inference_steps: int = 28,
        seed: int = None,
        strength: float = 0.85,
        scale: int = None,  # Upscale factor (1, 2, or 4) for upscaler models
        **kwargs
    ) -> GenerationResult:
        """
        Unified method: automatically routes to generate() or edit_image() based on model type.
        Supports multiple image inputs for various editing scenarios.
        
        Args:
            prompt: Text description
            model: Any ImageModel (generation or editing)
            image_url: Primary source image URL (required for edit models)
            image_urls: List of image URLs for multi-image editing
            style_image_url: Style reference image for style transfer models
            garment_image_url: Garment image for virtual try-on models
            mask_url: Mask URL for inpainting models
            size: Image size (for generation)
            num_images: Number of images (for generation)
            guidance_scale: Prompt adherence
            num_inference_steps: Denoising steps
            seed: Random seed
            strength: Edit strength (for editing)
            **kwargs: Additional model-specific parameters
            
        Returns:
            GenerationResult
        """
        # Check if this is an edit model
        if self.is_edit_model(model):
            # Determine primary image URL
            primary_image_url = image_url
            if not primary_image_url and image_urls:
                primary_image_url = image_urls[0]
            
            if not primary_image_url:
                return GenerationResult(
                    success=False,
                    images=[],
                    error=f"Model {model.value} requires image_url for editing",
                    model=model.value,
                )
            
            # Pass all image URLs to edit_image
            return await self.edit_image(
                image_url=primary_image_url,
                image_urls=image_urls,  # Multiple images
                style_image_url=style_image_url,  # Style reference
                garment_image_url=garment_image_url,  # Garment for try-on
                prompt=prompt,
                model=model,
                mask_url=mask_url,
                strength=strength,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                num_images=num_images,  # Number of output images
                size=size,  # Output size
                seed=seed,
                **kwargs
            )
        else:
            # Generation model
            return await self.generate(
                prompt=prompt,
                model=model,
                size=size,
                num_images=num_images,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                seed=seed,
                **kwargs
            )
    
    async def generate_batch(
        self,
        prompts: List[str],
        model: ImageModel = ImageModel.FLUX2_DEV,
        size: ImageSize = ImageSize.SQUARE_1024,
        **kwargs
    ) -> List[GenerationResult]:
        """
        Generate images for multiple prompts concurrently
        
        Args:
            prompts: List of prompts to generate
            model: Model to use
            size: Image size
            **kwargs: Additional arguments passed to generate()
            
        Returns:
            List of GenerationResult for each prompt
        """
        tasks = [
            self.generate(prompt=p, model=model, size=size, **kwargs)
            for p in prompts
        ]
        return await asyncio.gather(*tasks)
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of all available models with pricing info and type"""
        models = []
        for model in ImageModel:
            model_type = "generation"
            if model.value in self.EDIT_MODELS:
                model_type = "editing"
            
            models.append({
                "id": model.value,
                "name": model.name,
                "price_per_mp": self.MODEL_PRICING.get(model.value, 0.025),
                "description": self._get_model_description(model),
                "type": model_type,
                "supports_generation": model.value in self.GENERATION_MODELS,
                "supports_editing": model.value in self.EDIT_MODELS,
                "sample_prompt": MODEL_SETTINGS.get(model.value, DEFAULT_MODEL_SETTINGS).get("marketing_prompt"),
            })
        return models
    
    def _get_model_description(self, model: ImageModel) -> str:
        """Get description for a model"""
        descriptions = {
            # FLUX 2 Pro - Premium
            ImageModel.FLUX2_PRO: "FLUX.2 Pro - Maximum quality from Black Forest Labs",
            ImageModel.FLUX2_PRO_EDIT: "FLUX.2 Pro Edit - Premium image editing",
            # FLUX 2 Dev
            ImageModel.FLUX2_DEV: "FLUX.2 Dev - High quality generation from Black Forest Labs",
            ImageModel.FLUX2_DEV_EDIT: "FLUX.2 Dev Edit - Image editing with natural language",
            # Nano Banana Pro
            ImageModel.GEMINI3_PRO: "Nano Banana Pro - Realism and typography with web search",
            ImageModel.GEMINI3_PRO_EDIT: "Nano Banana Pro Edit - Multi-image editing",
            # ByteDance Seedream
            ImageModel.SEEDREAM_V45: "ByteDance Seedream V4.5 - Stylized content generation",
            ImageModel.SEEDREAM_V45_EDIT: "ByteDance Seedream Edit - Multi-image editing",
            # Z-Image Turbo
            ImageModel.Z_IMAGE_TURBO: "Z-Image Turbo - Ultra fast 6B model from Tongyi-MAI",
            
            # ByteDance Seedream V4 (4K)
            ImageModel.SEEDREAM_V4: "ByteDance Seedream V4 - 4K High Resolution Generation",
            ImageModel.SEEDREAM_V4_EDIT: "ByteDance Seedream V4 Edit - 4K High Resolution Editing",
            
            # Reve Models
            ImageModel.REVE: "Reve - State-of-the-art generation",
            ImageModel.REVE_EDIT: "Reve Edit - Professional image editing",
            ImageModel.REVE_FAST_EDIT: "Reve Fast Edit - Quick iteration editing",
            ImageModel.REVE_REMIX: "Reve Remix - Creative multi-image composition",
            ImageModel.REVE_FAST_REMIX: "Reve Fast Remix - Quick multi-image remixing",
            
            # Dreamina
            ImageModel.DREAMINA_V31: "ByteDance Dreamina V3.1 - Specialized portrait generation",
            
            # Ideogram
            ImageModel.IDEOGRAM_V3: "Ideogram V3 - Superior typography and style presets",
            ImageModel.IDEOGRAM_V3_REFRAME: "Ideogram V3 Reframe - Image resizing and outpainting",
            
            # Upscaling
            ImageModel.CREATIVE_UPSCALER: "Creative Upscaler - AI enhancement and upscaling",
            ImageModel.CLARITY_UPSCALER: "Clarity Upscaler - Sharpness and detail enhancement",
            ImageModel.RECRAFT_UPSCALE: "Recraft Upscale - Vector-style creative upscaling",
            
            # Removal
            ImageModel.OBJECT_REMOVAL: "Object Removal - Erase unwanted elements",
            ImageModel.BRIA_ERASER: "Bria Eraser - Commercial-safe object removal",
            ImageModel.TEXT_REMOVAL: "Text Removal - Automatic text erasure",
            
            # Style & Background
            ImageModel.STYLE_TRANSFER: "Style Transfer - Apply artistic styles to images",
            ImageModel.BACKGROUND_CHANGE: "Background Change - seamless background replacement",
            ImageModel.ADD_BACKGROUND: "Add Background - Generate new background for subject",
            ImageModel.RELIGHTING: "Relighting - Adjust scene illumination",
            
            # Camera Control
            ImageModel.QWEN_MULTIPLE_ANGLES: "Qwen Multiple Angles - Generate multi-view perspectives",
            
            # GPT Image 1.5
            ImageModel.GPT_IMAGE_15: "GPT Image 1.5 - Photorealistic generation with transparent backgrounds",
            ImageModel.GPT_IMAGE_15_EDIT: "GPT Image 1.5 Edit - Multi-image editing with transparent backgrounds",
        }
        return descriptions.get(model, "Image model")
    
    def get_generation_models(self) -> List[Dict[str, Any]]:
        """Get list of text-to-image generation models"""
        return [
            {
                "id": model.value,
                "name": model.name,
                "price_per_mp": self.MODEL_PRICING.get(model.value, 0.025),
                "description": self._get_model_description(model),
                "type": "generation",
            }
            for model in ImageModel
            if model.value in self.GENERATION_MODELS
        ]
    
    def get_edit_models(self) -> List[Dict[str, Any]]:
        """Get list of image editing models"""
        return [
            {
                "id": model.value,
                "name": model.name,
                "price_per_mp": self.MODEL_PRICING.get(model.value, 0.025),
                "description": self._get_model_description(model),
                "type": "editing",
            }
            for model in ImageModel
            if model.value in self.EDIT_MODELS
        ]


# Convenience function for quick generation
async def generate_image(
    prompt: str,
    model: str = "fal-ai/flux-2",
    width: int = 1024,
    height: int = 1024,
    num_images: int = 1,
    api_key: Optional[str] = None,
) -> GenerationResult:
    """
    Quick function to generate images
    
    Args:
        prompt: Text description
        model: Model endpoint ID
        width: Image width
        height: Image height
        num_images: Number of images
        api_key: Optional API key
        
    Returns:
        GenerationResult
    """
    generator = FalImageGenerator(api_key=api_key)
    
    # Find matching model enum or use default
    model_enum = ImageModel.FLUX2_DEV
    for m in ImageModel:
        if m.value == model:
            model_enum = m
            break
    
    return await generator.generate(
        prompt=prompt,
        model=model_enum,
        custom_width=width,
        custom_height=height,
        num_images=num_images,
    )


if __name__ == "__main__":
    # Test the generator
    async def test():
        try:
            generator = FalImageGenerator()
            result = await generator.generate(
                prompt="A beautiful sunset over mountains, photorealistic, 8k",
                model=ImageModel.Z_IMAGE_TURBO,  # Use fast model for testing
                size=ImageSize.SQUARE_512,
                num_images=1,
            )
            
            if result.success:
                print(f"✅ Generated {len(result.images)} image(s)")
                for i, img in enumerate(result.images):
                    print(f"   Image {i+1}: {img['url']}")
                print(f"   Seed: {result.seed}")
                print(f"   Cost: ${result.cost_estimate:.4f}")
            else:
                print(f"❌ Failed: {result.error}")
                
        except Exception as e:
            print(f"❌ Test failed: {e}")
    
    asyncio.run(test())


# =============================================================================
# BACKWARD COMPATIBILITY ALIASES
# =============================================================================

# Alias for backward compatibility with existing code
FalImageGenerator = GrovioImageGenerator

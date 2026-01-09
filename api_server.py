"""
Grovio AI Image Generation API Server
FastAPI server for marketing creatives with 50+ optimized models

Usage:
    uvicorn api_server:app --reload --port 8000
    
Endpoints:
    POST /generate - Generate images (JSON response)
    POST /stream   - Generate images with SSE streaming + auto-settings
    GET  /models   - List 50 marketing-optimized models
    GET  /models/{model_id}/settings - Get optimal settings for model
    GET  /marketing/prompts - Marketing prompt templates
    GET  /marketing/recommend - Get recommended model for use case
    POST /marketing/enhance-prompt - Enhance prompt for marketing
    GET  /history  - User generation history
    GET  /health   - Health check
"""

from fastapi import FastAPI, HTTPException, Form, Query, BackgroundTasks, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum
import asyncio
import json
import time
import os
import uuid
from datetime import datetime, timezone

# AWS S3 imports
try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    S3_AVAILABLE = True
    print("✅ AWS S3 dependencies available")
except ImportError:
    S3_AVAILABLE = False
    print("⚠️ AWS S3 dependencies not available. Install: pip install boto3")

# MongoDB imports
try:
    from motor.motor_asyncio import AsyncIOMotorClient
    from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
    MONGODB_AVAILABLE = True
    print("✅ MongoDB dependencies available")
except ImportError:
    MONGODB_AVAILABLE = False
    print("⚠️ MongoDB dependencies not available. Install: pip install motor")

# OpenAI imports for GPT-powered prompt enhancement
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
    print("✅ OpenAI dependencies available")
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️ OpenAI not available. Install: pip install openai")

# Import Grovio AI generator
from fal_image_generator import (
    GrovioImageGenerator,
    ImageModel,
    ImageSize,
    GenerationResult,
    SIZE_DIMENSIONS,
    MODEL_SETTINGS,
    DEFAULT_MODEL_SETTINGS,
    MARKETING_PROMPTS,
)

# Import Brand Memory for brand-aware generation
try:
    from brand_memory import BrandMemory, brand_memory
    BRAND_MEMORY_AVAILABLE = True
    print("✅ Brand Memory module available")
except ImportError:
    BRAND_MEMORY_AVAILABLE = False
    brand_memory = None
    print("⚠️ Brand Memory module not available")

# =============================================================================
# FASTAPI APP CONFIGURATION
# =============================================================================

app = FastAPI(
    title="Grovio AI Image Generation API",
    description="Marketing-focused image generation API with 50+ optimized models, auto-settings, and prompt enhancement",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add global exception handler for debugging 500 errors
import traceback
from fastapi import Request
from fastapi.responses import JSONResponse

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    error_msg = "".join(traceback.format_exception(None, exc, exc.__traceback__))
    print(f"CRITICAL ERROR: {error_msg}")  # This will show up in docker logs
    return JSONResponse(
        status_code=500,
        content={"message": "Internal Server Error", "detail": str(exc), "traceback": error_msg},
    )

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class ModelEnum(str, Enum):
    """Available models for API - 9 verified models"""
    # FLUX 2 Pro - Premium
    FLUX2_PRO = "flux2-pro"
    FLUX2_PRO_EDIT = "flux2-pro-edit"
    # FLUX 2 Dev
    FLUX2_DEV = "flux2-dev"
    FLUX2_DEV_EDIT = "flux2-dev-edit"
    # Nano Banana Pro (Google Gemini 3 Pro)
    GEMINI3_PRO = "gemini-3-pro"
    GEMINI3_PRO_EDIT = "gemini-3-pro-edit"
    # ByteDance Seedream
    SEEDREAM_V45 = "seedream-v4.5"
    SEEDREAM_V45_EDIT = "seedream-v4.5-edit"
    # Z-Image Turbo
    Z_IMAGE_TURBO = "z-image-turbo"


class SizeEnum(str, Enum):
    """Available image sizes"""
    SQUARE_512 = "512x512"
    SQUARE_1024 = "1024x1024"
    PORTRAIT_768x1024 = "768x1024"
    PORTRAIT_576x1024 = "576x1024"
    PORTRAIT_640x1536 = "640x1536"  # Ultra tall
    LANDSCAPE_1024x768 = "1024x768"
    LANDSCAPE_1024x576 = "1024x576"
    LANDSCAPE_1536x640 = "1536x640"  # Ultra wide
    CUSTOM = "custom"


class GenerateRequest(BaseModel):
    """Request body for image generation"""
    prompt: str = Field(..., description="Text description of the image to generate", min_length=1, max_length=2000)
    model: ModelEnum = Field(default=ModelEnum.FLUX2_DEV, description="Model to use for generation")
    size: SizeEnum = Field(default=SizeEnum.SQUARE_1024, description="Image size preset")
    width: Optional[int] = Field(default=None, ge=256, le=2048, description="Custom width (if size=custom)")
    height: Optional[int] = Field(default=None, ge=256, le=2048, description="Custom height (if size=custom)")
    num_images: int = Field(default=1, ge=1, le=4, description="Number of images to generate (1-4)")
    guidance_scale: float = Field(default=3.5, ge=1.0, le=20.0, description="Prompt adherence strength")
    num_inference_steps: int = Field(default=28, ge=1, le=50, description="Denoising steps")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")
    negative_prompt: Optional[str] = Field(default=None, max_length=1000, description="What to avoid")
    enable_safety_checker: bool = Field(default=True, description="Enable NSFW filter")
    output_format: str = Field(default="jpeg", pattern="^(jpeg|png)$", description="Output format")
    lora_url: Optional[str] = Field(default=None, description="LoRA weights URL")
    lora_scale: float = Field(default=1.0, ge=0.0, le=1.0, description="LoRA influence scale")
    
    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "A beautiful sunset over mountains, photorealistic, 8k resolution",
                "model": "flux2-dev",
                "size": "1024x1024",
                "num_images": 1,
                "guidance_scale": 3.5,
                "num_inference_steps": 28,
            }
        }


class GenerateResponse(BaseModel):
    """Response for image generation"""
    success: bool
    images: List[Dict[str, Any]]
    seed: Optional[int] = None
    prompt: str
    model: str
    cost_estimate: Optional[float] = None
    generation_time: float
    error: Optional[str] = None


class ModelInfo(BaseModel):
    """Model information"""
    id: str
    name: str
    price_per_mp: float
    description: str


# =============================================================================
# MODEL MAPPING
# =============================================================================

MODEL_MAP = {
    # FLUX 2 Pro - Premium
    "flux2-pro": ImageModel.FLUX2_PRO,
    "flux2-pro-edit": ImageModel.FLUX2_PRO_EDIT,
    # FLUX 2 Dev
    "flux2-dev": ImageModel.FLUX2_DEV,
    "flux2-dev-edit": ImageModel.FLUX2_DEV_EDIT,
    # Nano Banana Pro
    "gemini-3-pro": ImageModel.GEMINI3_PRO,
    "gemini-3-pro-edit": ImageModel.GEMINI3_PRO_EDIT,
    # Reve - State-of-the-art
    "reve": ImageModel.REVE,
    "reve-edit": ImageModel.REVE_EDIT,
    "reve-fast-edit": ImageModel.REVE_FAST_EDIT,
    "reve-remix": ImageModel.REVE_REMIX,
    "reve-fast-remix": ImageModel.REVE_FAST_REMIX,
    # ByteDance Seedream V4.5
    "seedream-v4.5": ImageModel.SEEDREAM_V45,
    "seedream-v4.5-edit": ImageModel.SEEDREAM_V45_EDIT,
    # ByteDance Seedream V4 (4K)
    "seedream-v4": ImageModel.SEEDREAM_V4,
    "seedream-v4-edit": ImageModel.SEEDREAM_V4_EDIT,
    # ByteDance Dreamina V3.1 (Portrait)
    "dreamina-v3.1": ImageModel.DREAMINA_V31,
    # Z-Image Turbo
    "z-image-turbo": ImageModel.Z_IMAGE_TURBO,
    # Ideogram V3 - Typography & Styles
    "ideogram-v3": ImageModel.IDEOGRAM_V3,
    "ideogram-v3-reframe": ImageModel.IDEOGRAM_V3_REFRAME,
    # Upscalers
    "creative-upscaler": ImageModel.CREATIVE_UPSCALER,
    "clarity-upscaler": ImageModel.CLARITY_UPSCALER,
    "recraft-upscale": ImageModel.RECRAFT_UPSCALE,
    # Object & Text Removal
    "object-removal": ImageModel.OBJECT_REMOVAL,
    "bria-eraser": ImageModel.BRIA_ERASER,
    "text-removal": ImageModel.TEXT_REMOVAL,
    # Style & Background
    "style-transfer": ImageModel.STYLE_TRANSFER,
    "background-change": ImageModel.BACKGROUND_CHANGE,
    "add-background": ImageModel.ADD_BACKGROUND,
    "relighting": ImageModel.RELIGHTING,
    # Character Multiple Angles
    "multiple-angles": ImageModel.QWEN_MULTIPLE_ANGLES,
    # GPT Image 1.5 - OpenAI model via Fal
    "gpt-image-1.5": ImageModel.GPT_IMAGE_15,
    "gpt-image-1.5-edit": ImageModel.GPT_IMAGE_15_EDIT,
}

# Reverse mapping: fal.ai model ID -> short name (for frontend display)
MODEL_ID_TO_SHORT_NAME = {model_enum.value: short_name for short_name, model_enum in MODEL_MAP.items()}

SIZE_MAP = {
    # Dimension format
    "512x512": ImageSize.SQUARE_512,
    "1024x1024": ImageSize.SQUARE_1024,
    "768x1024": ImageSize.PORTRAIT_3_4,
    "576x1024": ImageSize.PORTRAIT_9_16,
    "640x1536": ImageSize.PORTRAIT_9_21,  # Ultra tall (21:9)
    "1024x768": ImageSize.LANDSCAPE_4_3,
    "1024x576": ImageSize.LANDSCAPE_16_9,
    "1536x640": ImageSize.LANDSCAPE_21_9,  # Ultra wide (21:9)
    # Named format (user-friendly)
    "square": ImageSize.SQUARE_512,
    "square_hd": ImageSize.SQUARE_1024,
    "portrait": ImageSize.PORTRAIT_3_4,          # 3:4 vertical
    "portrait_3_4": ImageSize.PORTRAIT_3_4,      # 768x1024
    "portrait_9_16": ImageSize.PORTRAIT_9_16,    # 576x1024 (Instagram Story/Reels)
    "story": ImageSize.PORTRAIT_9_16,            # Instagram Story alias
    "reels": ImageSize.PORTRAIT_9_16,            # Instagram Reels alias
    "landscape": ImageSize.LANDSCAPE_4_3,        # 4:3 horizontal
    "landscape_4_3": ImageSize.LANDSCAPE_4_3,    # 1024x768
    "landscape_16_9": ImageSize.LANDSCAPE_16_9,  # 1024x576 (YouTube thumbnail)
    "widescreen": ImageSize.LANDSCAPE_16_9,      # Widescreen alias
    "youtube": ImageSize.LANDSCAPE_16_9,         # YouTube alias
}


# =============================================================================
# MODEL_SPECIFICATIONS and USE_CASE_MODELS removed - now stored in MongoDB
# Collections: Image_Models and Use_Cases
# Run seed_models.py to populate the database
# =============================================================================


# =============================================================================
# AWS S3 STORAGE CLASS
# =============================================================================

class S3ImageStorage:
    """Handle AWS S3 storage for uploaded and generated images"""
    
    def __init__(self):
        self.s3_client = None
        self.upload_bucket = os.getenv("AWS_S3_UPLOAD_BUCKET", "uploadminiapp")  # User uploads
        self.generated_bucket = os.getenv("AWS_S3_GENERATED_BUCKET", "generatedminiapp")  # Generated images
        self.aws_region = os.getenv("AWS_REGION", "ap-south-1")
        self._setup_s3_client()
    
    def _setup_s3_client(self):
        """Initialize S3 client with credentials from environment"""
        if not S3_AVAILABLE:
            print("⚠️ S3 not available - storage disabled")
            return
        
        try:
            aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
            aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
            
            if not aws_access_key or not aws_secret_key:
                print("⚠️ AWS credentials not found in environment")
                return
            
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
                region_name=self.aws_region
            )
            
            print(f"✅ S3 client initialized - upload: {self.upload_bucket}, generated: {self.generated_bucket}")
            
        except Exception as e:
            print(f"❌ S3 setup failed: {e}")
            self.s3_client = None
    
    def _generate_unique_key(self, user_id: str, org_id: str, file_extension: str, prefix: str = "images") -> str:
        """
        Generate unique S3 key: userid_orgid_datetime_uuid.ext
        
        Args:
            user_id: User identifier
            org_id: Organization identifier
            file_extension: File extension (jpg, png, etc.)
            prefix: Folder prefix (images, uploads, etc.)
        
        Returns:
            Unique S3 key path
        """
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        unique_id = str(uuid.uuid4())[:8]
        # Sanitize IDs (remove special chars)
        safe_user_id = user_id.replace('/', '_').replace(':', '_')[:50] if user_id else "unknown_user"
        safe_org_id = org_id.replace('/', '_').replace(':', '_')[:50] if org_id else "unknown_org"
        
        filename = f"{safe_user_id}_{safe_org_id}_{timestamp}_{unique_id}.{file_extension}"
        return f"{prefix}/{filename}"
    
    async def upload_user_image(self, image_bytes: bytes, mime_type: str, user_id: str, org_id: str) -> str:
        """
        Upload user-provided image to S3 upload bucket
        
        Returns:
            str: Public S3 URL or empty string if upload fails
        """
        if not self.s3_client:
            print("❌ S3 client not available")
            return ""
        
        try:
            file_extension = {
                'image/jpeg': 'jpg',
                'image/png': 'png',
                'image/gif': 'gif',
                'image/webp': 'webp'
            }.get(mime_type, 'jpg')
            
            s3_key = self._generate_unique_key(user_id, org_id, file_extension, "uploads")
            
            self.s3_client.put_object(
                Bucket=self.upload_bucket,
                Key=s3_key,
                Body=image_bytes,
                ContentType=mime_type,
                ACL='public-read',
                Metadata={
                    'user_id': user_id or '',
                    'org_id': org_id or '',
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
            )
            
            s3_url = f"https://{self.upload_bucket}.s3.{self.aws_region}.amazonaws.com/{s3_key}"
            print(f"✅ User image uploaded to S3: {s3_url}")
            return s3_url
            
        except Exception as e:
            print(f"❌ Failed to upload user image to S3: {e}")
            return ""
    
    async def save_generated_image(self, image_url: str, user_id: str, org_id: str, prompt: str) -> str:
        """
        Download generated image from fal.ai and save to S3 generated bucket
        
        Returns:
            str: S3 URL of saved image or original URL if save fails
        """
        if not self.s3_client:
            return image_url
        
        try:
            import aiohttp
            
            # Download image from fal.ai
            async with aiohttp.ClientSession() as session:
                async with session.get(image_url) as response:
                    if response.status != 200:
                        print(f"❌ Failed to download generated image: HTTP {response.status}")
                        return image_url
                    
                    image_bytes = await response.read()
                    content_type = response.headers.get('content-type', 'image/jpeg')
            
            # Determine file extension
            file_extension = 'jpg'
            if 'png' in content_type:
                file_extension = 'png'
            elif 'webp' in content_type:
                file_extension = 'webp'
            
            s3_key = self._generate_unique_key(user_id, org_id, file_extension, "generated")
            
            self.s3_client.put_object(
                Bucket=self.generated_bucket,
                Key=s3_key,
                Body=image_bytes,
                ContentType=content_type,
                ACL='public-read',
                Metadata={
                    'user_id': user_id or '',
                    'org_id': org_id or '',
                    'prompt': prompt[:200] if prompt else '',
                    'original_url': image_url[:500],
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
            )
            
            s3_url = f"https://{self.generated_bucket}.s3.{self.aws_region}.amazonaws.com/{s3_key}"
            print(f"✅ Generated image saved to S3: {s3_url}")
            return s3_url
            
        except Exception as e:
            print(f"❌ Failed to save generated image to S3: {e}")
            return image_url


# =============================================================================
# GPT PROMPT ENHANCER CLASS
# =============================================================================

class GPTPromptEnhancer:
    """
    GPT-4o powered prompt enhancement for marketing creatives.
    Transforms basic prompts into detailed, marketing-optimized descriptions.
    """
    
    def __init__(self):
        self.client = None
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o")
        self._initialize()
    
    def _initialize(self):
        """Initialize OpenAI client"""
        if not OPENAI_AVAILABLE:
            print("⚠️ OpenAI not available - GPT enhancement disabled")
            return
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("⚠️ OPENAI_API_KEY not set - GPT enhancement disabled")
            return
        
        try:
            self.client = OpenAI(api_key=api_key)
            print(f"✅ OpenAI GPT Enhancer initialized (model: {self.model})")
        except Exception as e:
            print(f"❌ OpenAI initialization failed: {e}")
            self.client = None
    
    async def enhance_prompt(
        self,
        prompt: str,
        model: str = None,
        use_case: str = None,
        platform: str = None,
        style: str = None,
        brand_context: str = None
    ) -> dict:
        """
        Enhance a basic prompt using GPT-4o for marketing-optimized image generation.
        
        Args:
            prompt: Original user prompt
            model: Target image model (for context)
            use_case: Marketing use case (social_media, advertising, e_commerce, etc.)
            platform: Target platform (instagram, facebook, linkedin, etc.)
            style: Desired style (photorealistic, illustration, minimalist, etc.)
            brand_context: Brand guidelines to inject (colors, style, personality)
        
        Returns:
            dict with original_prompt, enhanced_prompt, and metadata
        """
        if self.client is None:
            # Fallback to basic enhancement if GPT not available
            return self._fallback_enhance(prompt, model, use_case, platform)
        
        try:
            # Build context for GPT
            context_parts = []
            if model:
                model_info = MODEL_SETTINGS.get(model, {})
                if model_info.get("best_for"):
                    context_parts.append(f"Target model is best for: {', '.join(model_info['best_for'])}")
            if use_case:
                context_parts.append(f"Use case: {use_case.replace('_', ' ')}")
            if platform:
                context_parts.append(f"Target platform: {platform.replace('_', ' ')}")
            if style:
                context_parts.append(f"Desired style: {style}")
            
            context = ". ".join(context_parts) if context_parts else "General creative"
            
            # GPT system prompt for prompt enhancement
            system_prompt = """You are an AI image prompt enhancer. 
Your task is to take simple, short prompts and expand them into detailed, descriptive prompts optimized for AI image generation.

Guidelines:
1. Keep the core subject/concept from the original prompt
2. Add specific visual details: lighting, composition, colors, textures
3. Add technical quality keywords: high resolution, sharp focus, professional quality
4. Consider the target platform and use case for optimal composition
5. Keep the enhanced prompt concise but detailed (50-150 words)
6. Do NOT add text/words to appear in the image unless specifically requested
7. Focus on visual elements that work well for AI image generation

Output ONLY the enhanced prompt, nothing else."""
            
            # Add brand context if provided
            if brand_context:
                system_prompt += brand_context

            user_message = f"""Original prompt: "{prompt}"

Context: {context}

Transform this into a detailed, marketing-optimized image generation prompt."""

            # Call GPT (supports gpt-5.1 and gpt-4o)
            # Use max_completion_tokens for newer models (gpt-5.x), max_tokens for older
            api_params = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                "temperature": 0.7
            }
            # GPT-5.x models use max_completion_tokens instead of max_tokens
            if "gpt-5" in self.model or "o1" in self.model or "o3" in self.model:
                api_params["max_completion_tokens"] = 300
            else:
                api_params["max_tokens"] = 300
            
            response = self.client.chat.completions.create(**api_params)
            
            enhanced_prompt = response.choices[0].message.content.strip()
            
            # Remove quotes if GPT wrapped the response
            if enhanced_prompt.startswith('"') and enhanced_prompt.endswith('"'):
                enhanced_prompt = enhanced_prompt[1:-1]
            
            return {
                "original_prompt": prompt,
                "enhanced_prompt": enhanced_prompt,
                "model": model,
                "use_case": use_case,
                "platform": platform,
                "style": style,
                "enhancement_method": self.model,  # Use actual model name (gpt-5.1, gpt-4o, etc.)
                "tokens_used": response.usage.total_tokens if response.usage else None
            }
            
        except Exception as e:
            print(f"⚠️ GPT enhancement failed, using fallback: {e}")
            return self._fallback_enhance(prompt, model, use_case, platform)
    
    def _fallback_enhance(self, prompt: str, model: str = None, use_case: str = None, platform: str = None) -> dict:
        """Fallback enhancement using predefined marketing prompts"""
        enhanced = prompt
        
        # Add model-specific marketing prompt
        if model and model in MODEL_SETTINGS:
            marketing_prompt = MODEL_SETTINGS[model].get("marketing_prompt", "")
            if marketing_prompt:
                enhanced = f"{prompt}, {marketing_prompt}"
        
        # Add use case specific suffix
        if use_case and use_case in MARKETING_PROMPTS:
            use_case_prompts = MARKETING_PROMPTS[use_case]
            if platform and platform in use_case_prompts:
                enhanced = f"{enhanced}, {use_case_prompts[platform]}"
            elif "default" in use_case_prompts:
                enhanced = f"{enhanced}, {use_case_prompts['default']}"
        
        return {
            "original_prompt": prompt,
            "enhanced_prompt": enhanced,
            "model": model,
            "use_case": use_case,
            "platform": platform,
            "style": None,
            "enhancement_method": "fallback",
            "tokens_used": None
        }


# =============================================================================
# MONGODB STORAGE CLASS
# =============================================================================

class ImageMiniAppStorage:
    """MongoDB storage for Image Mini App records"""
    
    def __init__(self):
        self.client = None
        self.db = None
        self.collection = None
        # Get database and collection names from environment variables
        self.database_name = os.getenv("MONGODB_DATABASE", "Grovio_Mini_Apps")
        self.collection_name = os.getenv("MONGODB_COLLECTION", "Image_Mini_App")
        self._connect()
    
    def _get_connection_string(self) -> str:
        """Get MongoDB connection string from environment"""
        connection_string = os.getenv("MONGODB_CONNECTION_STRING")
        if not connection_string:
            connection_string = "mongodb+srv://shivraj:9nq6GM5DGHX4irNd@shivrajcluster.dv2no.mongodb.net/?retryWrites=true&w=majority"
            print("⚠️ Using fallback MongoDB connection string")
        return connection_string
    
    def _connect(self):
        """Establish MongoDB connection"""
        if not MONGODB_AVAILABLE:
            print("⚠️ MongoDB not available - storage disabled")
            return
        
        try:
            connection_string = self._get_connection_string()
            self.client = AsyncIOMotorClient(
                connection_string,
                serverSelectionTimeoutMS=30000,
                connectTimeoutMS=30000,
                maxPoolSize=50
            )
            
            self.db = self.client[self.database_name]
            self.collection = self.db[self.collection_name]
            
            print(f"✅ MongoDB connected: {self.database_name}.{self.collection_name}")
            
        except Exception as e:
            print(f"❌ MongoDB connection failed: {e}")
            self.client = None
    
    async def get_user_generations(
        self,
        user_id: str,
        org_id: str = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Get user's image generation history
        
        Returns:
            List of generation records
        """
        if self.collection is None:
            return []
        
        try:
            query = {"user_id": user_id}
            if org_id:
                query["org_id"] = org_id
            
            cursor = self.collection.find(query).sort("created_at", -1).skip(offset).limit(limit)
            records = await cursor.to_list(length=limit)
            
            # Convert ObjectId to string
            for record in records:
                record["_id"] = str(record["_id"])
            
            return records
            
        except Exception as e:
            print(f"❌ Failed to get user generations: {e}")
            return []


# =============================================================================
# DYNAMIC MODELS STORAGE CLASS
# =============================================================================

class DynamicModelsStorage:
    """
    MongoDB storage for dynamic model management.
    Fetches models from Image_Models collection with credits, tiers, and size multipliers.
    """
    
    def __init__(self):
        self.client = None
        self.db = None
        self.collection = None
        self._models_cache = {}  # In-memory cache for models list
        self._model_by_id_cache = {}  # Cache for individual model lookups
        self._cache_time = 0
        self._cache_ttl = 300  # 5 minutes cache
        self._initialize()
    
    def _initialize(self):
        """Initialize MongoDB connection for models"""
        if not MONGODB_AVAILABLE:
            print("⚠️ MongoDB not available - dynamic models disabled")
            return
        
        connection_string = os.getenv("MONGODB_CONNECTION_STRING")
        if not connection_string:
            # Fallback connection string for development
            connection_string = "mongodb+srv://shivraj:9nq6GM5DGHX4irNd@shivrajcluster.dv2no.mongodb.net/?retryWrites=true&w=majority"
            print("⚠️ Using fallback MongoDB connection string")
        
        try:
            self.client = AsyncIOMotorClient(connection_string)
            db_name = os.getenv("MONGODB_DATABASE", "Grovio_Mini_Apps")
            self.db = self.client[db_name]
            self.collection = self.db["Image_Models"]  # New collection for image models
            print(f"✅ Image Models Storage initialized: {db_name}.Image_Models")
        except Exception as e:
            print(f"❌ Failed to initialize Image Models Storage: {e}")
    
    async def get_all_models(self, include_disabled: bool = False) -> List[Dict[str, Any]]:
        """Get all models from database"""
        if self.collection is None:
            return []
        
        try:
            query = {} if include_disabled else {"enabled": True}
            cursor = self.collection.find(query).sort("category", 1)
            models = await cursor.to_list(length=200)
            for model in models:
                model["_id"] = str(model["_id"])
            return models
        except Exception as e:
            print(f"❌ Failed to get models: {e}")
            return []
    
    async def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get a single model by ID"""
        if self.collection is None:
            return None
        
        try:
            model = await self.collection.find_one({"model_id": model_id})
            if model:
                model["_id"] = str(model["_id"])
            return model
        except Exception as e:
            print(f"❌ Failed to get model {model_id}: {e}")
            return None
    
    async def add_model(self, model_data: Dict[str, Any]) -> str:
        """Add a new model"""
        if self.collection is None:
            return ""
        
        try:
            # Check if model already exists
            existing = await self.collection.find_one({"model_id": model_data["model_id"]})
            if existing:
                raise ValueError(f"Model {model_data['model_id']} already exists")
            
            model_data["created_at"] = datetime.now(timezone.utc)
            model_data["updated_at"] = datetime.now(timezone.utc)
            model_data["enabled"] = model_data.get("enabled", True)
            
            result = await self.collection.insert_one(model_data)
            self._models_cache = {}  # Clear cache
            print(f"✅ Model added: {model_data['model_id']}")
            return str(result.inserted_id)
        except Exception as e:
            print(f"❌ Failed to add model: {e}")
            raise
    
    async def update_model(self, model_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing model"""
        if self.collection is None:
            return False
        
        try:
            updates["updated_at"] = datetime.now(timezone.utc)
            result = await self.collection.update_one(
                {"model_id": model_id},
                {"$set": updates}
            )
            self._models_cache = {}  # Clear cache
            return result.modified_count > 0
        except Exception as e:
            print(f"❌ Failed to update model {model_id}: {e}")
            return False
    
    async def delete_model(self, model_id: str) -> bool:
        """Delete a model (or disable it)"""
        if self.collection is None:
            return False
        
        try:
            # Soft delete - just disable
            result = await self.collection.update_one(
                {"model_id": model_id},
                {"$set": {"enabled": False, "updated_at": datetime.now(timezone.utc)}}
            )
            self._models_cache = {}  # Clear cache
            return result.modified_count > 0
        except Exception as e:
            print(f"❌ Failed to delete model {model_id}: {e}")
            return False
    
    async def hard_delete_model(self, model_id: str) -> bool:
        """Permanently delete a model"""
        if self.collection is None:
            return False
        
        try:
            result = await self.collection.delete_one({"model_id": model_id})
            self._models_cache = {}  # Clear cache
            return result.deleted_count > 0
        except Exception as e:
            print(f"❌ Failed to hard delete model {model_id}: {e}")
            return False
    
    async def get_models_map(self) -> Dict[str, str]:
        """Get model_id -> fal_endpoint mapping for generation"""
        # Check cache
        if self._models_cache and (time.time() - self._cache_time) < self._cache_ttl:
            return self._models_cache
        
        models = await self.get_all_models(include_disabled=False)
        self._models_cache = {m["model_id"]: m["fal_endpoint"] for m in models}
        self._cache_time = time.time()
        return self._models_cache
    
    async def get_model_by_id(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get a single model by its model_id with full details including credits"""
        # Check cache first
        if model_id in self._model_by_id_cache:
            cached = self._model_by_id_cache[model_id]
            if (time.time() - cached["_cache_time"]) < self._cache_ttl:
                return cached
        
        if self.collection is None:
            return None
        
        try:
            model = await self.collection.find_one({"model_id": model_id, "enabled": True})
            if model:
                model["_id"] = str(model["_id"])
                model["_cache_time"] = time.time()
                self._model_by_id_cache[model_id] = model
            return model
        except Exception as e:
            print(f"❌ Failed to get model {model_id}: {e}")
            return None
    
    def calculate_credits(self, model: Dict[str, Any], size: str = "1024x1024", num_images: int = 1) -> int:
        """
        Calculate total credits for a generation request.
        
        Formula: base_credits * size_multiplier * num_images
        
        Args:
            model: Model document from database with credits and model_size fields
            size: Image size (e.g., "1024x1024")
            num_images: Number of images to generate
            
        Returns:
            Total credits to deduct (rounded up)
        """
        base_credits = model.get("credits", 1)
        size_multipliers = model.get("model_size", {})
        
        # Get size multiplier, default to 1.0 if not found
        multiplier = size_multipliers.get(size, 1.0)
        
        # Calculate total: base * multiplier * count, round up
        import math
        total = math.ceil(base_credits * multiplier * num_images)
        
        return max(1, total)  # Minimum 1 credit
    
    async def get_all_models_formatted(self, platform_sizes: Optional[Dict] = None, sample_prompt: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all models formatted for the /models endpoint with full details"""
        models = await self.get_all_models(include_disabled=False)
        
        # Credit to price_per_mp mapping (approximate based on credit costs)
        CREDIT_PRICE_MAP = {
            1: 0.005, 2: 0.015, 3: 0.027, 4: 0.04, 5: 0.03, 6: 0.05, 7: 0.15,
            8: 0.08  # Added for instagram-story models needing 0.08
        }
        
        # Format each model for the API response
        formatted = []
        for m in models:
            is_edit = m.get("is_edit_model", False)
            credits = m.get("credits", 1)
            
            # Build supported_sizes
            # If the model has a stored supported_sizes (from strict seeding), use it as base.
            # Otherwise fall back to synthetic.
            stored_supported_sizes = m.get("supported_sizes")
            
            if stored_supported_sizes:
                supported_sizes = stored_supported_sizes.copy()
                
                # If platform_sizes argument is present (from use case), inject/override it.
                if platform_sizes and isinstance(platform_sizes, dict):
                    supported_sizes["platform_sizes"] = platform_sizes
                    
                    # Aspect ratios: preserve order from platform_sizes values
                    supported_sizes["aspect_ratios"] = list(dict.fromkeys(
                        p.get("aspect_ratio", "1:1") for p in platform_sizes.values() if isinstance(p, dict)
                    ))
                    
                    # Sizes
                    ordered_sizes = []
                    seen_sizes = set()
                    for p in platform_sizes.values():
                        if not isinstance(p, dict): continue
                        size = p.get("closest_size")
                        if size and size not in seen_sizes:
                            ordered_sizes.append(size)
                            seen_sizes.add(size)
                    
                    if ordered_sizes:
                        supported_sizes["sizes"] = ordered_sizes
            else:
                 # Fallback to legacy logic
                 model_size = m.get("model_size", {})
                 supported_sizes = {
                    "family": "platform" if platform_sizes else m.get("category", "standard"),
                    "sizes": list(model_size.keys()) if model_size else ["1024x1024"],
                 }
                 if platform_sizes and isinstance(platform_sizes, dict):
                    supported_sizes["aspect_ratios"] = list(dict.fromkeys(
                        p.get("aspect_ratio", "1:1") for p in platform_sizes.values() if isinstance(p, dict)
                    ))
                    supported_sizes["platform_sizes"] = platform_sizes
                    
                    ordered_sizes = []
                    seen_sizes = set()
                    for p in platform_sizes.values():
                        if not isinstance(p, dict): continue
                        size = p.get("closest_size")
                        if size and size not in seen_sizes:
                            ordered_sizes.append(size)
                            seen_sizes.add(size)
                    if ordered_sizes:
                        supported_sizes["sizes"] = ordered_sizes

            # Add notes if platform sizes exist (common for all)
            if "supported_sizes" in m or platform_sizes:
                 if platform_sizes:
                     # We can't access query_slug or use_case in this method scope easily unless passed.
                     # But previously I used use_case.get('name') which was ALSO undefined?
                     # Wait, use_case dict is NOT available here.
                     # So "Optimized for {use_case...}" will fail too!
                     # I should just leave the note logic that relies on local vars if possible or pass use_case name?
                     # Or rely on client side?
                     # The strict response has "Optimized for Instagram Post" etc.
                     # I'll use a generic fallback or the sample_prompt if needed?
                     # Actually, I can pass 'use_case_name' too? Or just skip the f-string if vars missing.
                     # Let's check stored supported_sizes notes. It has "Optimized for...".
                     # If I used strict extraction, the correct note is ALREADY in stored_supported_sizes!
                     # So I don't need to override it dynamically if I extracted it correctly!
                     # BUT for display-ads, I injected platform_sizes.
                     pass 
                
            # Format model response
            model_response = {
                "id": m.get("fal_endpoint"),
                "name": m.get("name", "").upper().replace(" ", "_").replace("-", "_"),
                "price_per_mp": m.get("price_per_mp", CREDIT_PRICE_MAP.get(credits, 0.03)),
                "description": m.get("description", f"{m.get('name')} - {'Image editing' if is_edit else 'Image generation'}"),
                "type": "editing" if is_edit else "generation",
                "supports_generation": not is_edit,
                "supports_editing": is_edit,
                "sample_prompt": sample_prompt or m.get("sample_prompt"), 
                "model": m.get("model_id"),
                "requires_image": is_edit,
                "supported_sizes": supported_sizes,
                "max_input_images": m.get("max_input_images", 0 if not is_edit else 1),
                "max_output_images": m.get("max_output_images", 1),
            }
            
            formatted.append(model_response)
        
        return formatted
    
    async def get_all_use_cases(self) -> List[Dict[str, Any]]:
        """Get all use cases from database"""
        if self.db is None:
            return []
        
        try:
            use_cases_collection = self.db["Use_Cases"]
            cursor = use_cases_collection.find({"enabled": True})
            use_cases = await cursor.to_list(length=100)
            for uc in use_cases:
                uc["_id"] = str(uc["_id"])
            return use_cases
        except Exception as e:
            print(f"❌ Failed to get use cases: {e}")
            return []
    
    async def get_use_case(self, use_case_id: str) -> Optional[Dict[str, Any]]:
        """Get a single use case by ID"""
        if self.db is None:
            return None
        
        try:
            use_cases_collection = self.db["Use_Cases"]
            use_case = await use_cases_collection.find_one({"use_case_id": use_case_id, "enabled": True})
            if use_case:
                use_case["_id"] = str(use_case["_id"])
            return use_case
        except Exception as e:
            print(f"❌ Failed to get use case {use_case_id}: {e}")
            return None


# =============================================================================
# GLOBAL INSTANCES
# =============================================================================

generator: Optional[GrovioImageGenerator] = None
s3_storage: Optional[S3ImageStorage] = None
db_storage: Optional[ImageMiniAppStorage] = None
prompt_enhancer: Optional[GPTPromptEnhancer] = None
models_storage: Optional[DynamicModelsStorage] = None


def get_generator() -> GrovioImageGenerator:
    """Get or create the Grovio AI generator instance"""
    global generator
    if generator is None:
        # Support both GROVIO_API_KEY and FAL_KEY for backward compatibility
        api_key = os.getenv("GROVIO_API_KEY") or os.getenv("FAL_KEY")
        if not api_key:
            raise HTTPException(
                status_code=500,
                detail="GROVIO_API_KEY environment variable not set. Please set your Grovio AI API key."
            )
        generator = GrovioImageGenerator(api_key=api_key)
    return generator


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/", tags=["Info"])
async def root():
    """API root - returns basic info"""
    return {
        "name": "Grovio AI Image Generation API",
        "version": "2.0.0",
        "docs": "/docs",
        "endpoints": {
            "generate": "POST /generate - Generate images with SSE streaming",
            "models": "GET /models - List 50 marketing-optimized models",
            "models_settings": "GET /models/{model_id}/settings - Get optimal settings",
            "marketing_prompts": "GET /marketing/prompts - Marketing prompt templates",
            "marketing_recommend": "GET /marketing/recommend - Get recommended model for use case",
            "marketing_enhance": "POST /marketing/enhance-prompt - GPT-5.1 powered prompt enhancement",
            "batch": "POST /batch - Batch generate multiple prompts",
            "history": "GET /history - User generation history",
            "health": "GET /health - Health check",
        },
        "features": {
            "auto_settings": "Automatic optimal guidance_scale and steps per model",
            "gpt_enhance": "GPT-5.1 powered prompt enhancement (enhance_prompt=true)",
            "50_models": "50 curated models for marketing creatives",
            "use_cases": ["social_media", "advertising", "e_commerce", "branding", "typography", "product_photography", "virtual_tryon"],
        }
    }


@app.get("/health", tags=["Info"])
async def health_check():
    """Health check endpoint"""
    api_key_configured = bool(os.getenv("GROVIO_API_KEY") or os.getenv("FAL_KEY"))
    
    # Check S3 configuration
    s3_ok = False
    if s3_storage is not None and s3_storage.s3_client is not None:
        s3_ok = True
    
    # Check MongoDB configuration
    mongo_ok = False
    if db_storage is not None and db_storage.collection is not None:
        mongo_ok = True
    
    # Check GPT enhancer
    gpt_ok = False
    if prompt_enhancer is not None and prompt_enhancer.client is not None:
        gpt_ok = True
    
    return {
        "service": "Grovio AI Image Generation",
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "grovio_api_configured": api_key_configured,
        "s3_configured": s3_ok,
        "mongodb_configured": mongo_ok,
        "gpt_enhancer_configured": gpt_ok,
        "s3_buckets": {
            "upload": s3_storage.upload_bucket if s3_storage else None,
            "generated": s3_storage.generated_bucket if s3_storage else None
        },
        "models_available": 50,
        "features": ["auto_settings", "marketing_prompts", "gpt_prompt_enhancement"]
    }


@app.get("/models", tags=["Models"])
async def list_models(
    category: Optional[str] = Query(default=None, description="Filter by category: flux2, gemini, gpt, reve, seedream, turbo, ideogram, upscaler, utility, style"),
    tier: Optional[str] = Query(default=None, description="Filter by tier: Starter, Pro, Creator, Team"),
    is_edit_model: Optional[bool] = Query(default=None, description="Filter: true for editing models, false for generation models"),
    query: Optional[str] = Query(default=None, description="Use-case filter: instagram-post, facebook-ads, linkedin-content, etc.")
):
    """
    List all available image models from database with credit costs.
    
    **Response includes:**
    - `model_id`: Short model identifier (e.g., "flux2-pro")
    - `name`: Display name
    - `credits`: Base credit cost (1-7)
    - `tier`: Minimum plan tier (Starter, Pro, Creator, Team)
    - `category`: Model category
    - `is_edit_model`: Whether this is an editing model
    - `model_size`: Size -> credit multiplier mapping
    
    **Use-case Query:**
    Use `?query=instagram-post` to get models recommended for that use case.
    
    **Tiers:**
    - Starter ($19/mo): 9,000 credits
    - Pro ($49/mo): 24,000 credits
    - Creator ($99/mo): 48,000 credits
    - Team ($199/mo): 97,000 credits
    """
    global models_storage
    
    if models_storage is None:
        models_storage = DynamicModelsStorage()
    
    try:
        # Get all models from database
        models = await models_storage.get_all_models_formatted()
        
        if not models:
            return {
                "success": False,
                "error": "No models found in database. Run seed_models.py to populate.",
                "models": [],
                "total": 0
            }
        
        # Handle use-case query filter (from DB)
        if query:
            query_slug = query.lower().strip()
            use_case = await models_storage.get_use_case(query_slug)
            
            # Get all available use cases for available_queries
            # Enforce exact order from query_responses.json
            available_queries = [
                "create-images",
                "edit-images",
                "instagram-post",
                "instagram-story",
                "facebook-ads",
                "linkedin-content",
                "tiktok-thumbnails",
                "pinterest-pins",
                "display-ads",
                "hero-banners",
                "product-listings",
                "logos-posters",
                "vector-icons",
                "headshots",
                "youtube-thumbnails",
                "quick-mockups",
                "print-billboard",
                "fashion-tryon",
                "image-editing",
                "image-upscale",
                "character-multiple-angles"
            ]
            
            if use_case:
                # Get model IDs for this use case
                recommended_model_ids = use_case.get("models", [])
                editing_model_ids = use_case.get("editing_models", [])
                all_recommended_ids = recommended_model_ids + editing_model_ids
                
                # Get platform sizes for this use case
                platform_sizes = use_case.get("platform_sizes", {})
                sample_prompt = use_case.get("sample_prompt", "")
                
                # Get formatted models with platform sizes and sample prompt
                all_models = await models_storage.get_all_models_formatted(platform_sizes, sample_prompt)
                
                # Filter models to those in the use case
                filtered_models = [m for m in all_models if m.get("model") in all_recommended_ids]
                
                # Add sample_prompt and notes to each model
                for m in filtered_models:
                    m["sample_prompt"] = sample_prompt
                    if "supported_sizes" in m and platform_sizes:
                        m["supported_sizes"]["notes"] = f"Optimized for {use_case.get('name', query_slug)}"
                
                # Sort by their order in all_recommended_ids
                filtered_models.sort(key=lambda m: all_recommended_ids.index(m.get("model")) if m.get("model") in all_recommended_ids else 999)
                
                response = {
                    "query": query_slug,
                    "description": f"Recommended models for {use_case.get('name', query_slug)}",
                    "total": len(filtered_models),
                    "available_queries": available_queries,
                    "models": filtered_models,
                }
                
                # Conditionally add fields if they exist (strict matching)
                if use_case.get("system_prompt"):
                    response["system_prompt"] = use_case.get("system_prompt")
                if sample_prompt:
                    response["sample_prompt"] = sample_prompt
                if use_case.get("prompt_tips"):
                    response["prompt_tips"] = use_case.get("prompt_tips")
                if editing_model_ids:
                    response["editing_models"] = editing_model_ids
                if platform_sizes:
                    response["platform_sizes"] = platform_sizes
                if use_case.get("recommended_size"):
                    response["recommended_size"] = use_case.get("recommended_size")
                if use_case.get("notes"):
                    response["notes"] = use_case.get("notes")
                    
                return response
            else:
                return {
                    "error": f"Unknown use-case: {query_slug}",
                    "available_queries": available_queries,
                    "models": [],
                    "total": 0
                }
        
        # Apply filters
        if category:
            models = [m for m in models if m.get("category", "").lower() == category.lower()]
        
        if tier:
            models = [m for m in models if m.get("tier", "").lower() == tier.lower()]
        
        if is_edit_model is not None:
            models = [m for m in models if m.get("is_edit_model") == is_edit_model]
        
        # Group by category
        by_category = {}
        for m in models:
            cat = m.get("category", "other")
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(m)
        
        return {
            "success": True,
            "total": len(models),
            "tiers": {
                "Starter": {"price": 19, "credits": 9000},
                "Pro": {"price": 49, "credits": 24000},
                "Creator": {"price": 99, "credits": 48000},
                "Team": {"price": 199, "credits": 97000},
            },
            "size_multipliers": {
                "512x512": 0.5,
                "768x1024": 0.8,
                "576x1024": 0.6,
                "1024x768": 0.8,
                "1024x576": 0.6,
                "1024x1024": 1.0,
                "1536x640": 1.2,
                "640x1536": 1.2,
                "2048x2048": 1.5,
                "4096x4096": 2.0,
            },
            "models": models,
            "by_category": by_category
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch models: {str(e)}")

















@app.get("/models/{model_id}/settings", tags=["Models"])
async def get_model_settings(model_id: str):
    """
    Get optimal settings and **image input capabilities** for a specific model.
    
    Returns:
    - `settings`: guidance_scale, num_inference_steps, best_for use cases
    - `image_capabilities`: What images the model accepts and how many
    
    **Image Capabilities Response:**
    ```json
    {
      "image_capabilities": {
        "accepts_images": true,
        "required_images": 1,
        "max_images": 4,
        "image_params": ["base_images"],
        "optional_params": ["mask_url"],
        "description": "Multi-image editing - can process 1-4 base images"
      }
    }
    ```
    """
    try:
        # Find model in MODEL_MAP
        model_enum = MODEL_MAP.get(model_id)
        if not model_enum:
            raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
        
        gen = get_generator()
        settings = gen.get_model_settings(model_enum)
        image_caps = get_model_image_capabilities(model_id)
        
        return {
            "model_id": model_id,
            "fal_model": model_enum.value,
            "settings": settings,
            "price_per_mp": gen.MODEL_PRICING.get(model_enum.value, 0.025),
            "is_edit_model": gen.is_edit_model(model_enum),
            "is_generation_model": gen.is_generation_model(model_enum),
            "image_capabilities": image_caps,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Model image input capabilities - tells frontend what images each model accepts
MODEL_IMAGE_CAPABILITIES = {
    # Text-to-image models - no image input required
    "text_to_image": {
        "models": [
            "flux2-pro", "flux2-dev", "flux2-flex", "flux2-lora",
            "flux-schnell", "flux-dev", "flux-pro", "flux-pro-v1.1", "flux-pro-ultra", "flux-lora", "flux-realism",
            "ideogram-v3", "ideogram-turbo",
            "recraft-v3",
            "gemini-3-pro", "nano-banana", "imagen4",
            "seedream-v4.5",
            "z-image-turbo"
        ],
        "accepts_images": False,
        "required_images": 0,
        "max_images": 0,
        "image_params": [],
        "description": "Text-to-image generation - no image input needed"
    },
    # Single image editing models
    "single_image_edit": {
        "models": [
            # Note: flux2-pro-edit, flux2-dev-edit, nano-banana-edit, seedream-v4.5-edit are in multi_image_edit
            "flux2-flex-edit",
            "flux-kontext-pro", "flux-kontext-dev", "flux-kontext-max", "flux-kontext-lora", "flux-kontext-inpaint",
            "ideogram-v3-edit", "ideogram-v3-remix",
            "recraft-v3-edit", "recraft-vectorize",
            "object-removal", "text-removal", "bria-eraser", "flux-inpainting",
            "creative-upscaler", "clarity-upscaler",
            "background-change", "relighting",
            "headshot-photo", "professional-photo", "portrait-enhance"
        ],
        "accepts_images": True,
        "required_images": 1,
        "max_images": 1,
        "image_params": ["base_images"],
        "description": "Single image editing - requires 1 base image"
    },
    # Multi-image editing models (can accept multiple base images)
    "multi_image_edit": {
        "models": [
            "flux2-dev-edit", "flux2-pro-edit",  # Can blend multiple images
            "nano-banana-edit",  # Supports image_urls array
            "seedream-v4.5-edit",  # Supports image_urls array
        ],
        "accepts_images": True,
        "required_images": 1,
        "max_images": 4,
        "image_params": ["base_images"],
        "description": "Multi-image editing - can process 1-4 base images"
    },
    # Style transfer models (content + style reference)
    "style_transfer": {
        "models": ["style-transfer", "recraft-v3-edit"],
        "accepts_images": True,
        "required_images": 1,
        "max_images": 2,
        "image_params": ["base_images", "style_image"],
        "description": "Style transfer - requires base image, optional style reference"
    },
    # Virtual try-on models (person + garment)
    "virtual_tryon": {
        "models": ["virtual-tryon", "fashn-tryon", "fashion-tryon", "kling-tryon"],
        "accepts_images": True,
        "required_images": 2,
        "max_images": 2,
        "image_params": ["base_images", "garment_image"],
        "description": "Virtual try-on - requires person image + garment image"
    },
    # Product photography (product image required)
    "product_photography": {
        "models": ["product-photography", "product-holding", "product-photoshoot", "bria-product-shot", "integrate-product"],
        "accepts_images": True,
        "required_images": 1,
        "max_images": 1,
        "image_params": ["base_images"],
        "description": "Product photography - requires product image"
    },
    # Inpainting models (image + mask)
    "inpainting": {
        "models": ["flux-kontext-inpaint", "flux-inpainting", "object-removal"],
        "accepts_images": True,
        "required_images": 1,
        "max_images": 1,
        "image_params": ["base_images"],
        "optional_params": ["mask_url"],
        "description": "Inpainting - requires base image, optional mask URL"
    }
}


def get_model_image_capabilities(model_id: str) -> dict:
    """Get image input capabilities for a specific model"""
    # Check each category for the model
    for category, info in MODEL_IMAGE_CAPABILITIES.items():
        if model_id in info["models"]:
            # For models in multiple categories, prefer the more specific one
            if category == "multi_image_edit" and model_id in MODEL_IMAGE_CAPABILITIES["single_image_edit"]["models"]:
                # Return multi-image capabilities for models that support both
                return {
                    "category": category,
                    "accepts_images": info["accepts_images"],
                    "required_images": info["required_images"],
                    "max_images": info["max_images"],
                    "image_params": info["image_params"],
                    "optional_params": info.get("optional_params", []),
                    "description": info["description"]
                }
            return {
                "category": category,
                "accepts_images": info["accepts_images"],
                "required_images": info["required_images"],
                "max_images": info["max_images"],
                "image_params": info["image_params"],
                "optional_params": info.get("optional_params", []),
                "description": info["description"]
            }
    
    # Default: assume text-to-image if not found
    return {
        "category": "text_to_image",
        "accepts_images": False,
        "required_images": 0,
        "max_images": 0,
        "image_params": [],
        "optional_params": [],
        "description": "Text-to-image generation - no image input needed"
    }


# Model-wise supported sizes based on fal.ai documentation
MODEL_SUPPORTED_SIZES = {
    # FLUX 2 Models - Support aspect ratios: 21:9, 16:9, 4:3, 3:2, 1:1, 2:3, 3:4, 9:16, 9:21
    "flux2": {
        "models": ["flux2-pro", "flux2-dev", "flux2-flex", "flux2-pro-edit", "flux2-dev-edit", "flux2-flex-edit"],
        "sizes": ["1024x1024", "768x1024", "576x1024", "1024x768", "1024x576", "1536x640", "640x1536"],
        "aspect_ratios": ["1:1", "3:4", "9:16", "4:3", "16:9", "21:9", "9:21"],
        "max_resolution": "1536x1536",
        "notes": "Supports custom dimensions up to 1536px"
    },
    # FLUX 1 Models - Support image_size enum
    "flux1": {
        "models": ["flux-schnell", "flux-dev", "flux-pro", "flux-pro-v1.1", "flux-pro-ultra", "flux-lora", "flux-realism"],
        "sizes": ["512x512", "1024x1024", "768x1024", "576x1024", "1024x768", "1024x576"],
        "aspect_ratios": ["1:1", "3:4", "9:16", "4:3", "16:9"],
        "max_resolution": "2048x2048 (Ultra)",
        "notes": "flux-pro-ultra supports up to 2K resolution"
    },
    # FLUX Kontext - Context-aware editing
    "kontext": {
        "models": ["flux-kontext-pro", "flux-kontext-dev", "flux-kontext-max", "flux-kontext-lora", "flux-kontext-inpaint"],
        "sizes": ["1024x1024", "768x1024", "576x1024", "1024x768", "1024x576"],
        "aspect_ratios": ["1:1", "3:4", "9:16", "4:3", "16:9"],
        "max_resolution": "1024x1024",
        "notes": "Inherits size from input image for editing"
    },
    # Ideogram - Typography & Logos
    "ideogram": {
        "models": ["ideogram-v3", "ideogram-v3-edit", "ideogram-v3-remix", "ideogram-turbo"],
        "sizes": ["1024x1024", "768x1024", "576x1024", "1024x768", "1024x576", "1024x1536", "1536x1024"],
        "aspect_ratios": ["1:1", "3:4", "9:16", "4:3", "16:9", "2:3", "3:2"],
        "max_resolution": "1536x1536",
        "notes": "Best for logos and text-heavy images"
    },
    # Recraft - Vector Art
    "recraft": {
        "models": ["recraft-v3", "recraft-v3-edit", "recraft-vectorize"],
        "sizes": ["1024x1024", "768x1024", "1024x768", "1536x1024", "1024x1536"],
        "aspect_ratios": ["1:1", "3:4", "4:3", "3:2", "2:3"],
        "max_resolution": "1536x1536",
        "notes": "Supports SVG output for vectorize"
    },
    # Google Nano Banana
    "nano-banana": {
        "models": ["gemini-3-pro", "nano-banana", "nano-banana-edit"],
        "sizes": ["1024x1024", "768x1024", "576x1024", "1024x768", "1024x576"],
        "aspect_ratios": ["1:1", "3:4", "9:16", "4:3", "16:9"],
        "max_resolution": "1024x1024",
        "notes": "Google's Gemini-based model"
    },
    # ByteDance Seedream
    "seedream": {
        "models": ["seedream-v4.5", "seedream-v4.5-edit"],
        "sizes": ["1024x1024", "768x1024", "576x1024", "1024x768", "1024x576"],
        "aspect_ratios": ["1:1", "3:4", "9:16", "4:3", "16:9"],
        "max_resolution": "1024x1024",
        "notes": "ByteDance's latest model"
    },
    # Z-Image Turbo
    "z-image": {
        "models": ["z-image-turbo"],
        "sizes": ["1024x1024", "768x1024", "576x1024", "1024x768", "1024x576"],
        "aspect_ratios": ["1:1", "3:4", "9:16", "4:3", "16:9"],
        "max_resolution": "1024x1024",
        "notes": "Ultra-fast 6B model from Tongyi-MAI"
    },
    # Google Imagen 4
    "imagen": {
        "models": ["imagen4"],
        "sizes": ["1024x1024", "768x1024", "576x1024", "1024x768", "1024x576"],
        "aspect_ratios": ["1:1", "3:4", "9:16", "4:3", "16:9"],
        "max_resolution": "2048x2048",
        "notes": "Google Imagen 4 premium quality"
    },
    # Bria Models
    "bria": {
        "models": ["bria-product-shot", "bria-eraser"],
        "sizes": ["1024x1024", "768x1024", "1024x768"],
        "aspect_ratios": ["1:1", "3:4", "4:3"],
        "max_resolution": "1024x1024",
        "notes": "Commercial safe AI models"
    },
    # Product Photography - Usually inherits from input
    "product": {
        "models": ["product-photography", "product-holding", "product-photoshoot", "bria-product-shot", "integrate-product"],
        "sizes": ["1024x1024", "768x1024", "1024x768"],
        "aspect_ratios": ["1:1", "3:4", "4:3"],
        "max_resolution": "1024x1024",
        "notes": "Size depends on input product image"
    },
    # Virtual Try-On
    "tryon": {
        "models": ["virtual-tryon", "fashn-tryon", "fashion-tryon", "kling-tryon"],
        "sizes": ["768x1024", "576x1024"],
        "aspect_ratios": ["3:4", "9:16"],
        "max_resolution": "1024x1024",
        "notes": "Portrait orientation recommended for fashion"
    },
    # Upscaling - Output depends on input and scale
    "upscale": {
        "models": ["creative-upscaler", "clarity-upscaler", "recraft-upscale"],
        "sizes": ["Up to 4x input size"],
        "aspect_ratios": ["Preserves input aspect ratio"],
        "max_resolution": "4096x4096",
        "notes": "Output size = input size × scale factor (1-4x)"
    },
}


def get_model_supported_sizes(model_id: str) -> dict:
    """Get supported sizes for a model based on its family"""
    
    # If it's an edit model, it doesn't support custom output sizes (inherits from input)
    if model_id.endswith("-edit"):
        return {
            "family": "editing",
            "sizes": [],
            "aspect_ratios": [],
            "max_resolution": "Same as input",
            "notes": "Inherits size and aspect ratio from input image"
        }

    # Check each family for the model
    for family_name, family_info in MODEL_SUPPORTED_SIZES.items():
        if model_id in family_info["models"]:
            return {
                "family": family_name,
                "sizes": family_info["sizes"],
                "aspect_ratios": family_info["aspect_ratios"],
                "max_resolution": family_info["max_resolution"],
                "notes": family_info["notes"]
            }
    
    # Default for unknown models
    return {
        "family": "general",
        "sizes": ["1024x1024", "768x1024", "1024x768"],
        "aspect_ratios": ["1:1", "3:4", "4:3"],
        "max_resolution": "1024x1024",
        "notes": "Size typically inherited from input image"
    }


@app.get("/models/image-capabilities", tags=["Models"])
async def get_all_image_capabilities(
    model_id: Optional[str] = Query(default=None, description="Filter by specific model ID")
):
    """
    Get image input capabilities for all models or a specific model.
    
    **Frontend developers use this to know:**
    - Which models accept image uploads
    - How many images each model can process
    - What image parameters to use (base_images, style_image, garment_image)
    
    **Response for a specific model:**
    ```json
    {
      "model_id": "flux2-dev-edit",
      "accepts_images": true,
      "required_images": 1,
      "max_images": 4,
      "image_params": ["base_images"],
      "optional_params": [],
      "description": "Multi-image editing - can process 1-4 base images"
    }
    ```
    
    **Categories:**
    - `text_to_image`: No images needed (prompt only)
    - `single_image_edit`: 1 image required
    - `multi_image_edit`: 1-4 images supported
    - `style_transfer`: base_images + optional style_image
    - `virtual_tryon`: base_images (person) + garment_image
    - `product_photography`: 1 product image
    - `inpainting`: base_images + optional mask_url
    """
    if model_id:
        caps = get_model_image_capabilities(model_id)
        return {
            "model_id": model_id,
            **caps
        }
    
    # Return all categories with their models
    return {
        "categories": MODEL_IMAGE_CAPABILITIES,
        "image_params_reference": {
            "base_images": "Primary image(s) for editing - supports single or multiple files",
            "style_image": "Style reference image for style transfer",
            "garment_image": "Garment/clothing image for virtual try-on",
            "mask_url": "URL to mask image for inpainting (white=edit, black=keep)"
        }
    }


@app.get("/sizes", tags=["Models"])
async def get_supported_sizes(
    model_id: Optional[str] = Query(default=None, description="Filter by specific model ID")
):
    """
    Get supported image sizes for each model family.
    
    Returns aspect ratios and dimensions supported by different model families.
    Optionally filter by a specific model ID.
    """
    if model_id:
        # Find which family this model belongs to
        for family, info in MODEL_SUPPORTED_SIZES.items():
            if model_id in info["models"]:
                return {
                    "model_id": model_id,
                    "family": family,
                    "supported_sizes": info["sizes"],
                    "aspect_ratios": info["aspect_ratios"],
                    "max_resolution": info["max_resolution"],
                    "notes": info["notes"],
                    "size_aliases": {
                        "square": "512x512",
                        "square_hd": "1024x1024",
                        "portrait": "768x1024",
                        "portrait_9_16": "576x1024",
                        "story": "576x1024",
                        "reels": "576x1024",
                        "landscape": "1024x768",
                        "landscape_16_9": "1024x576",
                        "youtube": "1024x576",
                        "widescreen": "1024x576",
                    }
                }
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    
    # Return all model families with their supported sizes
    return {
        "model_families": MODEL_SUPPORTED_SIZES,
        "size_aliases": {
            "square": "512x512",
            "square_hd": "1024x1024", 
            "portrait": "768x1024 (3:4)",
            "portrait_9_16": "576x1024 (9:16)",
            "story": "576x1024 (Instagram Story)",
            "reels": "576x1024 (Instagram Reels)",
            "landscape": "1024x768 (4:3)",
            "landscape_16_9": "1024x576 (16:9)",
            "youtube": "1024x576 (YouTube thumbnail)",
            "widescreen": "1024x576 (16:9)",
        },
        "total_families": len(MODEL_SUPPORTED_SIZES),
    }


@app.get("/marketing/prompts", tags=["Marketing"])
async def get_marketing_prompts():
    """
    Get all available marketing prompt templates organized by use case.
    
    Categories: social_media_post, advertising, e_commerce, branding, seasonal, content_marketing
    """
    return {
        "prompts": MARKETING_PROMPTS,
        "categories": list(MARKETING_PROMPTS.keys()),
    }


@app.get("/marketing/recommend", tags=["Marketing"])
async def recommend_model(
    use_case: str = Query(..., description="Use case: instagram, facebook, product_listing, logo, etc.")
):
    """
    Get recommended model for a specific marketing use case.
    
    **Use Cases:**
    - Social: instagram, facebook, twitter, linkedin, tiktok, pinterest
    - Advertising: display_ad, hero_banner, product_ad, video_thumbnail
    - E-commerce: product_listing, lifestyle_shot, product_photoshoot
    - Typography: logo, poster, banner, text_heavy
    - Vector: vector, icon, infographic
    - Fashion: virtual_tryon, fashion
    - Professional: headshot, professional
    - Speed: quick, brainstorm, mockup
    - Quality: premium, print, billboard
    """
    try:
        gen = get_generator()
        recommended = gen.get_recommended_model(use_case)
        settings = gen.get_model_settings(recommended)
        
        # Find API model_id from MODEL_MAP
        api_model_id = None
        for key, value in MODEL_MAP.items():
            if value == recommended:
                api_model_id = key
                break
        
        return {
            "use_case": use_case,
            "recommended_model": api_model_id,
            "fal_model": recommended.value,
            "settings": settings,
            "price_per_mp": gen.MODEL_PRICING.get(recommended.value, 0.025),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/marketing/enhance-prompt", tags=["Marketing"])
async def enhance_prompt(
    prompt: str = Form(..., description="Original prompt"),
    model: str = Form(default="flux2-dev", description="Model to use"),
    use_case: Optional[str] = Form(default=None, description="Marketing use case"),
    platform: Optional[str] = Form(default=None, description="Target platform"),
    style: Optional[str] = Form(default=None, description="Desired style (photorealistic, illustration, minimalist, etc.)"),
    use_gpt: bool = Form(default=True, description="Use GPT-4o for enhancement (True) or fallback mode (False)")
):
    """
    Enhance a prompt with GPT-4o powered marketing optimization.
    
    **Features:**
    - GPT-4o transforms basic prompts into detailed, marketing-optimized descriptions
    - Falls back to predefined templates if GPT is unavailable
    - Frontend can control enhancement via `use_gpt` parameter
    
    **Use Cases:** social_media_post, advertising, e_commerce, branding, seasonal, content_marketing
    **Platforms:** instagram, facebook, twitter, linkedin, display_ad, product_listing, etc.
    **Styles:** photorealistic, illustration, minimalist, vintage, modern, corporate, etc.
    """
    try:
        gen = get_generator()
        model_enum = MODEL_MAP.get(model, ImageModel.FLUX2_DEV)
        settings = gen.get_model_settings(model_enum)
        
        # Use GPT enhancement if enabled and available
        if use_gpt and prompt_enhancer and prompt_enhancer.client:
            result = await prompt_enhancer.enhance_prompt(
                prompt=prompt,
                model=model,
                use_case=use_case,
                platform=platform,
                style=style
            )
            return {
                **result,
                "recommended_settings": {
                    "guidance_scale": settings.get("guidance_scale", 3.5),
                    "num_inference_steps": settings.get("num_inference_steps", 28),
                },
                "gpt_enabled": True
            }
        else:
            # Fallback to basic enhancement
            enhanced = gen.enhance_prompt_for_marketing(
                prompt=prompt,
                model=model_enum,
                use_case=use_case,
                platform=platform
            )
            
            return {
                "original_prompt": prompt,
                "enhanced_prompt": enhanced,
                "model": model,
                "use_case": use_case,
                "platform": platform,
                "style": style,
                "enhancement_method": "fallback",
                "tokens_used": None,
                "recommended_settings": {
                    "guidance_scale": settings.get("guidance_scale", 3.5),
                    "num_inference_steps": settings.get("num_inference_steps", 28),
                },
                "gpt_enabled": False
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate", tags=["Generation"])
async def stream_generate(
    prompt: str = Form(..., description="Text description of the image"),
    user_id: str = Form(..., description="User ID for tracking and storage"),
    org_id: Optional[str] = Form(default=None, description="Organization ID for multi-tenant support"),
    model: str = Form(default="flux2-dev", description="Model to use"),
    size: str = Form(default="1024x1024", description="Image size"),
    width: Optional[int] = Form(default=None, description="Custom width"),
    height: Optional[int] = Form(default=None, description="Custom height"),
    num_images: int = Form(default=1, ge=1, le=4, description="Number of images"),
    guidance_scale: Optional[float] = Form(default=None, description="Guidance scale (auto if not set)"),
    num_inference_steps: Optional[int] = Form(default=None, description="Inference steps (auto if not set)"),
    seed: Optional[int] = Form(default=None, description="Random seed"),
    enable_safety_checker: bool = Form(default=True, description="Safety checker"),
    output_format: str = Form(default="jpeg", description="Output format"),
    scale: Optional[int] = Form(default=None, description="Upscale factor (1, 2, or 4) - only for upscaler models"),
    lora_url: Optional[str] = Form(default=None, description="LoRA URL"),
    lora_scale: float = Form(default=1.0, description="LoRA scale"),
    base_images: Optional[List[UploadFile]] = File(default=None, description="Base image(s) for editing - supports single or multiple images"),
    style_image: Optional[UploadFile] = File(default=None, description="Style reference image for style transfer"),
    garment_image: Optional[UploadFile] = File(default=None, description="Garment image for virtual try-on"),
    strength: float = Form(default=0.85, ge=0.0, le=1.0, description="Edit strength (0.0-1.0, higher = more change)"),
    mask_url: Optional[str] = Form(default=None, description="Mask URL for inpainting (white=edit, black=keep)"),
    # Marketing enhancement options
    use_case: Optional[str] = Form(default=None, description="Marketing use case: social_media_post, advertising, etc."),
    platform: Optional[str] = Form(default=None, description="Target platform: instagram, facebook, etc."),
    style: Optional[str] = Form(default=None, description="Desired style: photorealistic, illustration, minimalist, etc."),
    enhance_prompt: bool = Form(default=False, description="Use GPT-4o to enhance prompt (True/False)"),
    auto_settings: bool = Form(default=True, description="Use optimal model settings automatically"),
    brand_details: bool = Form(default=False, description="Use brand guidance (colors, style, logo for edit models) - requires org_id"),
):
    """
    Generate images with Server-Sent Events (SSE) streaming
    
    Provides real-time progress updates during generation.
    Supports **single or multiple image uploads** for editing scenarios.
    
    **Image Upload Options:**
    - `base_images`: Base image(s) for editing - works with single or multiple files
    - `style_image`: Style reference image for style transfer models
    - `garment_image`: Garment image for virtual try-on models
    
    **Event Types:**
    - `session_start`: Generation started
    - `processing`: Progress update
    - `image_generated`: Image completed with URL
    - `complete`: All images generated
    - `error`: Error occurred
    
    **Example: Single Image Edit**
    ```bash
    curl -X POST "/generate" \\
      -F "prompt=Change background to beach" \\
      -F "user_id=user_123" -F "org_id=org_456" \\
      -F "model=flux2-dev-edit" \\
      -F "base_images=@photo.jpg"
    ```
    
    **Example: Multiple Images**
    ```bash
    curl -X POST "/generate" \\
      -F "prompt=Combine these images" \\
      -F "user_id=user_123" -F "org_id=org_456" \\
      -F "model=flux2-dev-edit" \\
      -F "base_images=@image1.jpg" \\
      -F "base_images=@image2.jpg"
    ```
    
    **Example: Style Transfer**
    ```bash
    curl -X POST "/generate" \\
      -F "prompt=Apply this artistic style" \\
      -F "user_id=user_123" -F "org_id=org_456" \\
      -F "model=style-transfer" \\
      -F "base_images=@content.jpg" \\
      -F "style_image=@style_reference.jpg"
    ```
    
    **Example: Virtual Try-On**
    ```bash
    curl -X POST "/generate" \\
      -F "prompt=Try on this garment" \\
      -F "user_id=user_123" -F "org_id=org_456" \\
      -F "model=virtual-tryon" \\
      -F "base_images=@person.jpg" \\
      -F "garment_image=@shirt.jpg"
    ```
    """
    # Helper function to upload image to S3
    async def upload_image_to_s3(image_file: UploadFile, label: str) -> Optional[str]:
        """Upload a single image file to S3 and return the URL"""
        if not image_file or not s3_storage or not s3_storage.s3_client:
            return None
        try:
            image_bytes = await image_file.read()
            s3_url = await s3_storage.upload_user_image(
                image_bytes,
                image_file.content_type or 'image/jpeg',
                user_id,
                org_id or ''
            )
            print(f"📁 {label} uploaded to S3: {s3_url}")
            return s3_url
        except Exception as e:
            print(f"⚠️ Failed to upload {label}: {e}")
            return None
    
    # Process all image uploads
    base_image_s3_url = None  # Primary image (first one)
    base_image_s3_urls = []   # All images list
    style_image_s3_url = None
    garment_image_s3_url = None
    
    # Upload base images (single or multiple)
    if base_images:
        for i, img in enumerate(base_images):
            if img and img.filename:  # Check if file was actually uploaded
                url = await upload_image_to_s3(img, f"Base image {i+1}")
                if url:
                    base_image_s3_urls.append(url)
        # First image is the primary one
        if base_image_s3_urls:
            base_image_s3_url = base_image_s3_urls[0]
            print(f"📁 Uploaded {len(base_image_s3_urls)} base image(s)")
    
    # Upload style reference image (for style transfer)
    if style_image:
        style_image_s3_url = await upload_image_to_s3(style_image, "Style image")
    
    # Upload garment image (for virtual try-on)
    if garment_image:
        garment_image_s3_url = await upload_image_to_s3(garment_image, "Garment image")
    
    async def generate_stream():
        """SSE stream generator"""
        start_time = time.time()
        session_id = f"gen_{int(time.time() * 1000)}"
        
        # Send session start with user context
        yield f"data: {json.dumps({'type': 'session_start', 'session_id': session_id, 'user_id': user_id, 'org_id': org_id, 'timestamp': datetime.now(timezone.utc).isoformat()})}\n\n"
        
        try:
            # Declare nonlocal for variables we may modify (brand logo)
            nonlocal base_image_s3_url, base_image_s3_urls
            
            gen = get_generator()
            
            # Map model and size
            model_enum = MODEL_MAP.get(model, ImageModel.FLUX2_DEV)
            size_enum = SIZE_MAP.get(size, ImageSize.SQUARE_1024)
            
            # Get model settings for auto-configuration
            model_settings = gen.get_model_settings(model_enum)
            
            # Apply auto-settings if enabled
            actual_guidance = guidance_scale
            actual_steps = num_inference_steps
            if auto_settings:
                if actual_guidance is None:
                    actual_guidance = model_settings.get("guidance_scale", 3.5)
                if actual_steps is None:
                    actual_steps = model_settings.get("num_inference_steps", 28)
            else:
                actual_guidance = actual_guidance or 3.5
                actual_steps = actual_steps or 28
            
            # Brand Memory: Fetch brand context if brand_details=True
            brand_context = None
            brand_logo_url = None
            brand_name = None
            if brand_details and org_id and BRAND_MEMORY_AVAILABLE and brand_memory:
                print(f"\n{'='*60}")
                print(f"🏢 [/generate] BRAND DETAILS REQUEST")
                print(f"{'='*60}")
                print(f"   org_id: {org_id}")
                print(f"   brand_details: {brand_details}")
                
                yield f"data: {json.dumps({'type': 'processing', 'message': 'Fetching brand guidelines...', 'brand_details': True})}\n\n"
                brand_data = await brand_memory.get_brand_context(org_id)
                if brand_data:
                    brand_context = brand_memory.format_for_prompt(brand_data)
                    brand_name = brand_data.get('organization_name', 'Unknown')
                    brand_colors = brand_memory.get_brand_colors_hex(brand_data)
                    brand_logo_url = brand_data.get('logo_url', '')
                    
                    # Log all brand details
                    print(f"   ✅ Brand Found: {brand_name}")
                    print(f"   Colors: {brand_colors}")
                    print(f"   Logo URL: {brand_logo_url}")
                    print(f"   Brand Context for Prompt:")
                    for line in brand_context.split('\n'):
                        print(f"      {line}")
                    print(f"{'='*60}\n")
                    
                    yield f"data: {json.dumps({'type': 'brand_context', 'brand_name': brand_name, 'brand_colors': brand_colors, 'brand_guidelines': brand_context, 'logo_url': brand_logo_url})}\n\n"
                    
                    # Auto-use logo for edit models when brand_details=True
                    is_edit_model = gen.is_edit_model(model_enum)
                    if brand_logo_url and is_edit_model and not base_image_s3_url:
                        base_image_s3_url = brand_logo_url
                        base_image_s3_urls = [brand_logo_url]
                        print(f"🖼️ [/generate] Auto-using brand logo as base image for edit model")
                        yield f"data: {json.dumps({'type': 'brand_logo', 'message': 'Using brand logo as base image (edit model)', 'logo_url': brand_logo_url})}\n\n"
                else:
                    print(f"   ⚠️ Brand NOT found for org_id: {org_id}")
                    print(f"{'='*60}\n")
                    yield f"data: {json.dumps({'type': 'warning', 'message': f'Brand not found for org_id: {org_id}'})}\n\n"
            
            # Enhance prompt for marketing if enabled
            final_prompt = prompt
            enhancement_method = None
            if enhance_prompt or brand_context:
                # Use GPT enhancement if available
                if prompt_enhancer and prompt_enhancer.client:
                    brand_system = ""
                    if brand_context:
                        brand_system = f"\n\nIMPORTANT - Apply these brand guidelines:\n{brand_context}\n\nUse the EXACT hex color codes provided."
                    
                    print(f"\n✨ [/generate] PROMPT ENHANCEMENT")
                    print(f"   Original: {prompt[:80]}...")
                    print(f"   Method: GPT{'+brand' if brand_context else ''}")
                    
                    yield f"data: {json.dumps({'type': 'processing', 'message': 'Enhancing prompt with GPT' + (' + brand' if brand_context else '') + '...'})}\n\n"
                    enhancement_result = await prompt_enhancer.enhance_prompt(
                        prompt=prompt,
                        model=model,
                        use_case=use_case,
                        platform=platform,
                        style=style,
                        brand_context=brand_system if brand_context else None
                    )
                    final_prompt = enhancement_result["enhanced_prompt"]
                    enhancement_method = "gpt-5.1+brand" if brand_context else enhancement_result.get("enhancement_method", "gpt-5.1")
                    tokens_used = enhancement_result.get("tokens_used")
                    
                    print(f"   Enhanced: {final_prompt[:80]}...")
                    print(f"   Tokens Used: {tokens_used}")
                    
                    yield f"data: {json.dumps({'type': 'prompt_enhanced', 'original_prompt': prompt, 'enhanced_prompt': final_prompt, 'method': enhancement_method, 'tokens_used': tokens_used, 'brand_applied': bool(brand_context)})}\n\n"
                else:
                    # Fallback to basic enhancement
                    final_prompt = gen.enhance_prompt_for_marketing(
                        prompt=prompt,
                        model=model_enum,
                        use_case=use_case,
                        platform=platform
                    )
                    enhancement_method = "fallback"
                    yield f"data: {json.dumps({'type': 'prompt_enhanced', 'original_prompt': prompt, 'enhanced_prompt': final_prompt, 'method': 'fallback', 'tokens_used': None})}\n\n"
            
            # Handle custom dimensions
            custom_width = width
            custom_height = height
            if size == "custom" and (not width or not height):
                yield f"data: {json.dumps({'type': 'error', 'message': 'width and height required for custom size'})}\n\n"
                return
            
            # Send processing start with settings info
            settings_info = {
                'model': model,
                'num_images': num_images,
                'guidance_scale': actual_guidance,
                'num_inference_steps': actual_steps,
                'auto_settings': auto_settings,
                'enhance_prompt': enhance_prompt,
                'enhancement_method': enhancement_method,
                'best_for': model_settings.get('best_for', [])
            }
            yield f"data: {json.dumps({'type': 'processing', 'message': f'Starting generation with {model}...', **settings_info})}\n\n"
            
            # Estimate cost
            if custom_width and custom_height:
                w, h = custom_width, custom_height
            else:
                dims = SIZE_DIMENSIONS.get(size_enum.value, {"width": 1024, "height": 1024})
                w, h = dims["width"], dims["height"]
            
            cost_estimate = gen._estimate_cost(model_enum.value, w, h, num_images)
            yield f"data: {json.dumps({'type': 'processing', 'message': f'Estimated cost: ${cost_estimate:.4f}', 'cost_estimate': cost_estimate})}\n\n"
            
            # Determine if this is an editing operation
            # Editing is triggered by: base_image, base_images, style_image, or garment_image
            has_any_image = base_image_s3_url or base_image_s3_urls or style_image_s3_url or garment_image_s3_url
            is_editing = has_any_image and gen.is_edit_model(model_enum)
            
            # Build image URLs list for models that support multiple inputs
            image_urls = []
            if base_image_s3_urls:
                image_urls.extend(base_image_s3_urls)
            elif base_image_s3_url:
                image_urls.append(base_image_s3_url)
            
            # Log the operation mode
            if is_editing:
                edit_info = {
                    'base_image_url': base_image_s3_url,
                    'base_image_urls': base_image_s3_urls if base_image_s3_urls else None,
                    'style_image_url': style_image_s3_url,
                    'garment_image_url': garment_image_s3_url,
                    'total_images': len(image_urls) + (1 if style_image_s3_url else 0) + (1 if garment_image_s3_url else 0)
                }
                yield f"data: {json.dumps({'type': 'processing', 'message': f'Editing with {model}...', 'mode': 'edit', **{k:v for k,v in edit_info.items() if v}})}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'processing', 'message': f'Generating images... This may take 5-30 seconds.', 'mode': 'generate'})}\n\n"
            
            # =====================================================================
            # SPECIAL HANDLING: Multiple Angles (ALWAYS 8 parallel API calls)
            # =====================================================================
            if model == "multiple-angles" and base_image_s3_url:
                yield f"data: {json.dumps({'type': 'processing', 'message': 'Generating 8 camera angles in parallel...', 'mode': 'multiple_angles'})}\n\n"

                # Define 8 angle presets (ALWAYS generate all 8 angles)
                angle_presets = [
                    {"name": "front", "rotate_right_left": 0, "vertical_angle": 0, "move_forward": 0},
                    {"name": "right_45", "rotate_right_left": -45, "vertical_angle": 0, "move_forward": 0},
                    {"name": "left_45", "rotate_right_left": 45, "vertical_angle": 0, "move_forward": 0},
                    {"name": "right_profile", "rotate_right_left": -90, "vertical_angle": 0, "move_forward": 0},
                    {"name": "left_profile", "rotate_right_left": 90, "vertical_angle": 0, "move_forward": 0},
                    {"name": "high_angle", "rotate_right_left": 0, "vertical_angle": -0.5, "move_forward": 0},
                    {"name": "low_angle", "rotate_right_left": 0, "vertical_angle": 0.5, "move_forward": 0},
                    {"name": "closeup", "rotate_right_left": 0, "vertical_angle": 0, "move_forward": 5},
                ]

                # Make 8 parallel API calls using asyncio.gather
                async def generate_single_angle(angle_config):
                    """Generate a single angle using fal.ai API directly"""
                    import fal_client
                    try:
                        result = await fal_client.run_async(
                            "fal-ai/qwen-image-edit-2509-lora-gallery/multiple-angles",
                            arguments={
                                "image_urls": [base_image_s3_url],
                                "rotate_right_left": angle_config["rotate_right_left"],
                                "vertical_angle": angle_config["vertical_angle"],
                                "move_forward": angle_config["move_forward"],
                                "guidance_scale": 1,
                                "num_inference_steps": 6,
                                "lora_scale": 1.25,
                                "num_images": 1,
                                "output_format": output_format,
                                "enable_safety_checker": enable_safety_checker,
                            }
                        )
                        return {"success": True, "angle": angle_config["name"], "images": result.get("images", [])}
                    except Exception as e:
                        return {"success": False, "angle": angle_config["name"], "error": str(e)}
                
                # Run all 8 API calls in parallel
                import asyncio
                yield f"data: {json.dumps({'type': 'processing', 'message': 'Making 8 parallel API calls...', 'angles': [a['name'] for a in angle_presets]})}\n\n"
                
                angle_results = await asyncio.gather(*[generate_single_angle(a) for a in angle_presets])
                
                # Collect successful images
                all_images = []
                for i, angle_result in enumerate(angle_results):
                    if angle_result["success"] and angle_result.get("images"):
                        for img in angle_result["images"]:
                            angle_name = angle_result["angle"]
                            img["angle"] = angle_name
                            all_images.append(img)
                            yield f"data: {json.dumps({'type': 'processing', 'message': f'Angle {i+1}/8 ({angle_name}) complete'})}\n\n"
                    else:
                        angle_name = angle_result["angle"]
                        error_msg = angle_result.get("error", "Unknown error")
                        yield f"data: {json.dumps({'type': 'warning', 'message': f'Angle {angle_name} failed: {error_msg}'})}\n\n"
                
                if not all_images:
                    yield f"data: {json.dumps({'type': 'error', 'message': 'All angle generations failed'})}\n\n"
                    return
                
                # Save all images to S3
                s3_urls = []
                yield f"data: {json.dumps({'type': 'processing', 'message': f'Saving {len(all_images)} images to storage...'})}\n\n"
                
                for i, img in enumerate(all_images):
                    s3_url = await s3_storage.save_generated_image(
                        img.get('url', ''),
                        user_id,
                        org_id or '',
                        f"{prompt} - {img.get('angle', 'angle')}"
                    )
                    s3_urls.append(s3_url)
                    img['s3_url'] = s3_url
                    yield f"data: {json.dumps({'type': 'image_generated', 'index': i + 1, 'total': len(all_images), 'image': img})}\n\n"
                
                generation_time = time.time() - start_time
                cost_estimate = len(all_images) * 0.03  # $0.03 per image

                # Note: Generation logging is now handled by frontend

                yield f"data: {json.dumps({'type': 'complete', 'success': True, 'images': all_images, 's3_urls': s3_urls, 'angles_generated': len(all_images), 'prompt': prompt, 'model': model, 'cost_estimate': cost_estimate, 'generation_time': generation_time})}\n\n"
                return  # Exit early - don't run normal generation
            
            # =====================================================================
            # STANDARD GENERATION (for all other models)
            # =====================================================================
            
            # Use generate_or_edit which handles both generation and editing
            result = await gen.generate_or_edit(
                prompt=final_prompt,
                model=model_enum,
                image_url=base_image_s3_url,  # Primary base image for editing
                image_urls=image_urls if len(image_urls) > 1 else None,  # Multiple images
                style_image_url=style_image_s3_url,  # Style reference for style transfer
                garment_image_url=garment_image_s3_url,  # Garment for virtual try-on
                mask_url=mask_url,  # For inpainting models
                size=size_enum,
                num_images=num_images,
                guidance_scale=actual_guidance,
                num_inference_steps=actual_steps,
                seed=seed,
                strength=strength,  # Edit strength (0.0-1.0)
                enable_safety_checker=enable_safety_checker,
                output_format=output_format,
                lora_url=lora_url,
                lora_scale=lora_scale,
                custom_width=custom_width,
                custom_height=custom_height,
                scale=scale,  # Upscale factor (1, 2, or 4) for upscaler models
            )
            
            if not result.success:
                print(f"❌ [/generate] Image generation FAILED: {result.error}")
                yield f"data: {json.dumps({'type': 'error', 'message': result.error})}\n\n"
                return
            
            # Log image generation success
            print(f"\n🎨 [/generate] IMAGE GENERATION COMPLETE")
            print(f"   Model: {model}")
            print(f"   Images Generated: {len(result.images)}")
            print(f"   Cost Estimate: ${result.cost_estimate:.4f}" if result.cost_estimate else "   Cost: N/A")
            print(f"   Seed: {result.seed}")
            
            # Save generated images to S3 and collect URLs
            s3_urls = []
            print(f"\n📤 [/generate] UPLOADING TO S3...")
            yield f"data: {json.dumps({'type': 'processing', 'message': 'Saving images to storage...'})}\n\n"
            
            for i, img in enumerate(result.images):
                # Save to S3
                s3_url = await s3_storage.save_generated_image(
                    img.get('url', ''),
                    user_id,
                    org_id or '',
                    prompt
                )
                s3_urls.append(s3_url)
                print(f"   ✅ Image {i+1}: {s3_url[:60]}...")
                
                # Send image with both original and S3 URL
                img_with_s3 = {**img, 's3_url': s3_url}
                yield f"data: {json.dumps({'type': 'image_generated', 'index': i + 1, 'total': len(result.images), 'image': img_with_s3})}\n\n"
            
            print(f"   All {len(s3_urls)} images uploaded to S3")
            
            generation_time = time.time() - start_time

            # Note: Generation logging is now handled by frontend
            record_id = None

            # Send completion with S3 URLs
            yield f"data: {json.dumps({'type': 'complete', 'success': True, 'images': result.images, 's3_urls': s3_urls, 'record_id': record_id, 'seed': result.seed, 'prompt': final_prompt, 'original_prompt': prompt, 'model': model, 'cost_estimate': result.cost_estimate, 'generation_time': generation_time, 'brand_enabled': brand_context is not None})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


# =============================================================================
# BRAND-AWARE GENERATION ENDPOINT
# =============================================================================

@app.post("/brand-generate", tags=["Brand Generation"])
async def brand_generate(
    prompt: str = Form(..., description="Text description of the image"),
    org_id: str = Form(..., description="Organization ID (REQUIRED) - fetches brand guidelines"),
    user_id: str = Form(..., description="User ID for tracking and storage"),
    model: str = Form(default="flux2-dev", description="Model to use"),
    size: str = Form(default="1024x1024", description="Image size"),
    width: Optional[int] = Form(default=None, description="Custom width"),
    height: Optional[int] = Form(default=None, description="Custom height"),
    num_images: int = Form(default=1, ge=1, le=4, description="Number of images"),
    guidance_scale: Optional[float] = Form(default=None, description="Guidance scale (auto if not set)"),
    num_inference_steps: Optional[int] = Form(default=None, description="Inference steps (auto if not set)"),
    seed: Optional[int] = Form(default=None, description="Random seed"),
    enable_safety_checker: bool = Form(default=True, description="Safety checker"),
    output_format: str = Form(default="jpeg", description="Output format"),
    lora_url: Optional[str] = Form(default=None, description="LoRA URL"),
    lora_scale: float = Form(default=1.0, description="LoRA scale"),
    base_images: Optional[List[UploadFile]] = File(default=None, description="Base image(s) for editing"),
    style_image: Optional[UploadFile] = File(default=None, description="Style reference image"),
    garment_image: Optional[UploadFile] = File(default=None, description="Garment image for try-on"),
    strength: float = Form(default=0.85, ge=0.0, le=1.0, description="Edit strength"),
    mask_url: Optional[str] = Form(default=None, description="Mask URL for inpainting"),
    use_case: Optional[str] = Form(default=None, description="Marketing use case"),
    platform: Optional[str] = Form(default=None, description="Target platform"),
    style: Optional[str] = Form(default=None, description="Desired style"),
    auto_settings: bool = Form(default=True, description="Use optimal model settings"),
    use_logo: bool = Form(default=False, description="Use brand logo as base image (for edit models)"),
):
    """
    Brand-Aware Image Generation with SSE Streaming
    
    **Requires org_id** - Fetches brand guidelines (colors, visual style, personality)
    from the brand_summaries collection and injects them into GPT prompt enhancement.
    
    **How it works:**
    1. Fetches brand data: colors, visual style, brand personality
    2. Formats brand guidelines for GPT
    3. Enhances prompt with brand context using GPT-5.1
    4. Generates on-brand images
    
    **Example:**
    ```bash
    curl -X POST "/brand-generate" \\
      -F "prompt=Create a car leasing advertisement" \\
      -F "org_id=ORG_J64NKRUF_Y3ACAQH1C5ACDXB7" \\
      -F "user_id=user_123" \\
      -F "model=flux2-pro"
    ```
    
    **SSE Events:**
    - `brand_context`: Brand guidelines loaded
    - `prompt_enhanced`: GPT-enhanced prompt with brand
    - `processing`: Generation progress
    - `image_generated`: Image completed
    - `complete`: All images generated
    """
    # Validate brand memory is available
    if not BRAND_MEMORY_AVAILABLE or not brand_memory:
        raise HTTPException(
            status_code=503,
            detail="Brand Memory module not available. Check MongoDB connection."
        )
    
    # Helper function to upload image to S3
    async def upload_image_to_s3(image_file: UploadFile, label: str) -> Optional[str]:
        if not image_file or not s3_storage or not s3_storage.s3_client:
            return None
        try:
            image_bytes = await image_file.read()
            s3_url = await s3_storage.upload_user_image(
                image_bytes,
                image_file.content_type or 'image/jpeg',
                user_id,
                org_id
            )
            print(f"📁 {label} uploaded to S3: {s3_url}")
            return s3_url
        except Exception as e:
            print(f"⚠️ Failed to upload {label}: {e}")
            return None
    
    # Process image uploads
    base_image_s3_url = None
    base_image_s3_urls = []
    style_image_s3_url = None
    garment_image_s3_url = None
    
    if base_images:
        for i, img in enumerate(base_images):
            if img and img.filename:
                url = await upload_image_to_s3(img, f"Base image {i+1}")
                if url:
                    base_image_s3_urls.append(url)
        if base_image_s3_urls:
            base_image_s3_url = base_image_s3_urls[0]
    
    if style_image:
        style_image_s3_url = await upload_image_to_s3(style_image, "Style image")
    
    if garment_image:
        garment_image_s3_url = await upload_image_to_s3(garment_image, "Garment image")
    
    async def generate_stream():
        start_time = time.time()
        session_id = f"brand_gen_{int(time.time() * 1000)}"
        
        yield f"data: {json.dumps({'type': 'session_start', 'session_id': session_id, 'user_id': user_id, 'org_id': org_id, 'brand_enabled': True})}\n\n"
        
        try:
            # Declare nonlocal for variables we'll modify
            nonlocal base_image_s3_url, base_image_s3_urls
            
            # Step 1: Fetch brand context
            yield f"data: {json.dumps({'type': 'processing', 'message': 'Fetching brand guidelines...'})}\n\n"
            
            brand_data = await brand_memory.get_brand_context(org_id)
            
            if not brand_data:
                yield f"data: {json.dumps({'type': 'warning', 'message': f'Brand not found for org_id: {org_id}. Proceeding without brand context.'})}\n\n"
                brand_context = None
                brand_logo_url = None
            else:
                brand_context = brand_memory.format_for_prompt(brand_data)
                brand_name = brand_data.get('organization_name', 'Unknown')
                brand_colors = brand_memory.get_brand_colors_hex(brand_data)
                brand_logo_url = brand_data.get('logo_url', '')
                
                yield f"data: {json.dumps({'type': 'brand_context', 'brand_name': brand_name, 'brand_colors': brand_colors, 'brand_guidelines': brand_context, 'logo_url': brand_logo_url})}\n\n"
            
            gen = get_generator()
            model_enum = MODEL_MAP.get(model, ImageModel.FLUX2_DEV)
            is_edit_model = gen.is_edit_model(model_enum)
            
            # Auto-use brand logo for edit models (or if use_logo=True explicitly)
            logo_used = False
            if brand_logo_url and not base_image_s3_url:
                # Auto-use logo if: user requested use_logo OR it's an edit model
                should_use_logo = use_logo or is_edit_model
                if should_use_logo:
                    base_image_s3_url = brand_logo_url
                    base_image_s3_urls = [brand_logo_url]
                    logo_used = True
                    reason = "edit model selected" if is_edit_model else "use_logo=true"
                    yield f"data: {json.dumps({'type': 'brand_logo', 'message': f'Using brand logo as base image ({reason})', 'logo_url': brand_logo_url})}\n\n"
            
            size_enum = SIZE_MAP.get(size, ImageSize.SQUARE_1024)
            model_settings = gen.get_model_settings(model_enum)
            
            # Auto-settings
            actual_guidance = guidance_scale if guidance_scale else model_settings.get("guidance_scale", 3.5)
            actual_steps = num_inference_steps if num_inference_steps else model_settings.get("num_inference_steps", 28)
            
            # Step 2: Enhance prompt with brand context
            final_prompt = prompt
            enhancement_method = None
            
            if prompt_enhancer and prompt_enhancer.client:
                yield f"data: {json.dumps({'type': 'processing', 'message': 'Enhancing prompt with brand guidelines...'})}\n\n"
                
                # Build enhanced system prompt with brand context
                brand_system_addition = ""
                if brand_context:
                    brand_system_addition = f"""

BRAND GUIDELINES (MUST FOLLOW):
{brand_context}

Apply these brand guidelines to:
- Color choices and color palette
- Visual style and aesthetics
- Overall mood and tone
- Professional quality matching brand personality"""
                
                # Call GPT with brand context in system prompt
                try:
                    system_prompt = f"""You are an expert marketing creative director and AI image prompt engineer.
Your task is to transform basic image descriptions into detailed, professional prompts optimized for AI image generation.

Guidelines:
1. Keep the core subject/concept from the original prompt
2. Add specific visual details: lighting, composition, colors, textures
3. Include marketing-relevant elements: professional quality, brand-appropriate aesthetics
4. Add technical quality keywords: high resolution, sharp focus, professional photography
5. Keep the enhanced prompt concise but detailed (50-150 words)
6. Focus on visual elements that convert well for marketing
{brand_system_addition}

Output ONLY the enhanced prompt, nothing else."""

                    context_parts = []
                    if use_case:
                        context_parts.append(f"Use case: {use_case}")
                    if platform:
                        context_parts.append(f"Platform: {platform}")
                    if style:
                        context_parts.append(f"Style: {style}")
                    context = ". ".join(context_parts) if context_parts else "Marketing creative"
                    
                    user_message = f'Original prompt: "{prompt}"\n\nContext: {context}\n\nTransform this into a detailed, brand-aligned image generation prompt.'
                    
                    api_params = {
                        "model": prompt_enhancer.model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_message}
                        ],
                        "temperature": 0.7
                    }
                    if "gpt-5" in prompt_enhancer.model or "o1" in prompt_enhancer.model:
                        api_params["max_completion_tokens"] = 300
                    else:
                        api_params["max_tokens"] = 300
                    
                    response = prompt_enhancer.client.chat.completions.create(**api_params)
                    final_prompt = response.choices[0].message.content.strip()
                    if final_prompt.startswith('"') and final_prompt.endswith('"'):
                        final_prompt = final_prompt[1:-1]
                    
                    enhancement_method = f"{prompt_enhancer.model}+brand"
                    tokens_used = response.usage.total_tokens if response.usage else None
                    
                    yield f"data: {json.dumps({'type': 'prompt_enhanced', 'original_prompt': prompt, 'enhanced_prompt': final_prompt, 'method': enhancement_method, 'tokens_used': tokens_used, 'brand_applied': True})}\n\n"
                    
                except Exception as e:
                    print(f"⚠️ Brand prompt enhancement failed: {e}")
                    yield f"data: {json.dumps({'type': 'warning', 'message': f'Prompt enhancement failed: {str(e)}. Using original prompt.'})}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'warning', 'message': 'GPT not available. Using original prompt without enhancement.'})}\n\n"
            
            # Estimate cost
            if width and height:
                w, h = width, height
            else:
                dims = SIZE_DIMENSIONS.get(size_enum.value, {"width": 1024, "height": 1024})
                w, h = dims["width"], dims["height"]
            
            cost_estimate = gen._estimate_cost(model_enum.value, w, h, num_images)
            yield f"data: {json.dumps({'type': 'processing', 'message': f'Estimated cost: ${cost_estimate:.4f}', 'cost_estimate': cost_estimate})}\n\n"
            
            # Determine editing mode
            image_urls = base_image_s3_urls if base_image_s3_urls else ([base_image_s3_url] if base_image_s3_url else [])
            has_any_image = base_image_s3_url or style_image_s3_url or garment_image_s3_url
            is_editing = has_any_image and gen.is_edit_model(model_enum)
            
            yield f"data: {json.dumps({'type': 'processing', 'message': f'Generating brand-aligned images with {model}...', 'mode': 'edit' if is_editing else 'generate'})}\n\n"
            
            # Generate images
            result = await gen.generate_or_edit(
                prompt=final_prompt,
                model=model_enum,
                image_url=base_image_s3_url,
                image_urls=image_urls if len(image_urls) > 1 else None,
                style_image_url=style_image_s3_url,
                garment_image_url=garment_image_s3_url,
                mask_url=mask_url,
                size=size_enum,
                num_images=num_images,
                guidance_scale=actual_guidance,
                num_inference_steps=actual_steps,
                seed=seed,
                strength=strength,
                enable_safety_checker=enable_safety_checker,
                output_format=output_format,
                lora_url=lora_url,
                lora_scale=lora_scale,
                custom_width=width,
                custom_height=height,
            )
            
            if not result.success:
                yield f"data: {json.dumps({'type': 'error', 'message': result.error})}\n\n"
                return
            
            # Save to S3
            s3_urls = []
            yield f"data: {json.dumps({'type': 'processing', 'message': 'Saving images to storage...'})}\n\n"
            
            for i, img in enumerate(result.images):
                s3_url = await s3_storage.save_generated_image(
                    img.get('url', ''),
                    user_id,
                    org_id,
                    prompt
                )
                s3_urls.append(s3_url)
                img_with_s3 = {**img, 's3_url': s3_url}
                yield f"data: {json.dumps({'type': 'image_generated', 'index': i + 1, 'total': len(result.images), 'image': img_with_s3})}\n\n"
            
            generation_time = time.time() - start_time
            
            # Note: Generation logging is now handled by frontend

            yield f"data: {json.dumps({'type': 'complete', 'success': True, 'images': result.images, 's3_urls': s3_urls, 'seed': result.seed, 'prompt': final_prompt, 'original_prompt': prompt, 'model': model, 'cost_estimate': result.cost_estimate, 'generation_time': generation_time, 'brand_enabled': True, 'brand_name': brand_data.get('organization_name') if brand_data else None})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


# =============================================================================
# BRAND REST API ENDPOINTS (For use by other miniapps)
# =============================================================================

@app.get("/brand/{org_id}", tags=["Brand API"])
async def get_brand_data(org_id: str):
    """
    Get complete brand data for an organization.
    
    **For use by other miniapps** (Tweet Creator, etc.)
    
    Returns brand core, creative guidelines, and voice data formatted for prompt injection.
    
    **Example:**
    ```
    GET http://localhost:5005/brand/ORG_J64NKRUF_Y3ACAQH1C5ACDXB7
    ```
    """
    if not BRAND_MEMORY_AVAILABLE or not brand_memory:
        raise HTTPException(status_code=503, detail="Brand Memory not available")
    
    try:
        brand_data = await brand_memory.get_brand_context(org_id)
        
        if not brand_data:
            raise HTTPException(status_code=404, detail=f"Brand not found for org_id: {org_id}")
        
        return {
            "status": "success",
            "org_id": org_id,
            "organization_name": brand_data.get("organization_name"),
            "brand_core": brand_data.get("brand_core", {}),
            "creative_guidelines": brand_data.get("creative_guidelines", {}),
            "tone_voice": brand_data.get("tone_voice", {}),
            "prompt_context": brand_memory.format_for_prompt(brand_data),
            "brand_colors": brand_memory.get_brand_colors_hex(brand_data),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/brand/{org_id}/creative", tags=["Brand API"])
async def get_brand_creative(org_id: str):
    """
    Get brand creative guidelines (colors, typography, visual style).
    
    **For use by design/image generation miniapps.**
    """
    if not BRAND_MEMORY_AVAILABLE or not brand_memory:
        raise HTTPException(status_code=503, detail="Brand Memory not available")
    
    try:
        brand_data = await brand_memory.get_brand_context(org_id)
        
        if not brand_data:
            raise HTTPException(status_code=404, detail=f"Brand not found for org_id: {org_id}")
        
        creative = brand_data.get("creative_guidelines", {})
        
        return {
            "status": "success",
            "org_id": org_id,
            "organization_name": brand_data.get("organization_name"),
            "color_palette": creative.get("color_palette", []),
            "typography": creative.get("typography", {}),
            "visual_style": creative.get("visual_style", {}),
            "logo_usage": creative.get("logo_usage", {}),
            "brand_colors_hex": brand_memory.get_brand_colors_hex(brand_data),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/brand/{org_id}/voice", tags=["Brand API"])
async def get_brand_voice(org_id: str):
    """
    Get brand tone and voice guidelines.
    
    **For use by content creation miniapps** (Tweet Creator, etc.)
    """
    if not BRAND_MEMORY_AVAILABLE or not brand_memory:
        raise HTTPException(status_code=503, detail="Brand Memory not available")
    
    try:
        brand_data = await brand_memory.get_brand_context(org_id)
        
        if not brand_data:
            raise HTTPException(status_code=404, detail=f"Brand not found for org_id: {org_id}")
        
        tone_voice = brand_data.get("tone_voice", {})
        brand_core = brand_data.get("brand_core", {})
        
        return {
            "status": "success",
            "org_id": org_id,
            "organization_name": brand_data.get("organization_name"),
            "voice_guidelines": tone_voice.get("voice_guidelines", {}),
            "vocabulary": tone_voice.get("vocabulary", {}),
            "writing_examples": tone_voice.get("writing_examples", {}),
            "brand_personality": brand_core.get("brand_personality", {}),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/brand/{org_id}/prompt-context", tags=["Brand API"])
async def get_brand_prompt_context(org_id: str):
    """
    Get pre-formatted brand context for prompt injection.
    
    **Returns a ready-to-use string** for LLM system prompts.
    
    **Example usage in other miniapps:**
    ```python
    response = requests.get(f"http://images-api:5005/brand/{org_id}/prompt-context")
    brand_context = response.json()["prompt_context"]
    
    # Inject into your LLM prompt:
    system_prompt = f"You are a content writer. {brand_context}"
    ```
    """
    if not BRAND_MEMORY_AVAILABLE or not brand_memory:
        raise HTTPException(status_code=503, detail="Brand Memory not available")
    
    try:
        brand_data = await brand_memory.get_brand_context(org_id)
        
        if not brand_data:
            raise HTTPException(status_code=404, detail=f"Brand not found for org_id: {org_id}")
        
        return {
            "status": "success",
            "org_id": org_id,
            "organization_name": brand_data.get("organization_name"),
            "prompt_context": brand_memory.format_for_prompt(brand_data),
            "brand_colors": brand_memory.get_brand_colors_hex(brand_data),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/brands", tags=["Brand API"])
async def list_all_brands(
    limit: int = Query(default=50, ge=1, le=100, description="Max brands to return"),
    skip: int = Query(default=0, ge=0, description="Number to skip for pagination")
):
    """
    List all available brands.
    
    Returns basic info: org_id, name, and whether brand data exists.
    """
    if not BRAND_MEMORY_AVAILABLE or brand_memory is None or brand_memory.collection is None:
        raise HTTPException(status_code=503, detail="Brand Memory not available")
    
    try:
        cursor = brand_memory.collection.find(
            {},
            {
                "organization_id": 1,
                "organization_name": 1,
                "metadata.website_url": 1,
            }
        ).skip(skip).limit(limit).sort("organization_name", 1)
        
        brands = []
        for brand in cursor:
            brands.append({
                "org_id": brand.get("organization_id"),
                "organization_name": brand.get("organization_name"),
                "website_url": brand.get("metadata", {}).get("website_url"),
            })
        
        total = brand_memory.collection.count_documents({})
        
        return {
            "status": "success",
            "total_count": total,
            "returned_count": len(brands),
            "skip": skip,
            "limit": limit,
            "brands": brands,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history", tags=["History"])
async def get_generation_history(
    user_id: str = Query(..., description="User ID to get history for"),
    org_id: str = Query(..., description="Organization ID (required)"),
    limit: int = Query(default=50, ge=1, le=200, description="Number of records to return"),
    offset: int = Query(default=0, ge=0, description="Number of records to skip")
):
    """
    Get user's image generation history from MongoDB
    
    Both user_id and org_id are required.
    Returns list of generation records with S3 URLs.
    """
    try:
        if db_storage is None or db_storage.collection is None:
            raise HTTPException(status_code=503, detail="Database not available")
        
        records = await db_storage.get_user_generations(
            user_id=user_id,
            org_id=org_id,
            limit=limit,
            offset=offset
        )
        
        return {
            "success": True,
            "user_id": user_id,
            "org_id": org_id,
            "total_returned": len(records),
            "limit": limit,
            "offset": offset,
            "records": records
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch", tags=["Generation"])
async def batch_generate(
    prompts: List[str] = Form(..., description="List of prompts"),
    model: str = Form(default="flux2-dev", description="Model to use"),
    size: str = Form(default="1024x1024", description="Image size"),
):
    """
    Generate images for multiple prompts in batch
    
    Processes all prompts concurrently for faster results.
    """
    start_time = time.time()
    
    try:
        gen = get_generator()
        model_enum = MODEL_MAP.get(model, ImageModel.FLUX2_DEV)
        size_enum = SIZE_MAP.get(size, ImageSize.SQUARE_1024)
        
        results = await gen.generate_batch(
            prompts=prompts,
            model=model_enum,
            size=size_enum,
        )
        
        generation_time = time.time() - start_time
        
        return {
            "success": True,
            "results": [
                {
                    "prompt": prompts[i],
                    "success": r.success,
                    "images": r.images,
                    "seed": r.seed,
                    "error": r.error,
                    "cost_estimate": r.cost_estimate,
                }
                for i, r in enumerate(results)
            ],
            "total_prompts": len(prompts),
            "successful": sum(1 for r in results if r.success),
            "generation_time": generation_time,
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# ENHANCE ENDPOINTS - POST-PROCESSING OPERATIONS
# =============================================================================

@app.post("/enhance/upscale", tags=["Enhance"])
async def upscale_image(
    image_url: str = Form(..., description="URL of image to upscale (S3 or fal.ai URL)"),
    user_id: str = Form(..., description="User ID for tracking"),
    org_id: str = Form(default=None, description="Organization ID"),
    scale: float = Form(default=2.0, ge=1.0, le=4.0, description="Upscale factor (1-4x)"),
    model: str = Form(default="creative-upscaler", description="Upscaler model: creative-upscaler, clarity-upscaler, recraft-upscale"),
):
    """
    Upscale an existing image to higher resolution.
    
    Models:
    - creative-upscaler: Best for artistic/creative upscaling
    - clarity-upscaler: Best for clarity and sharpness
    - recraft-upscale: Recraft's creative upscaler
    """
    start_time = time.time()
    
    try:
        gen = get_generator()
        
        # Map model string to enum
        upscale_models = {
            "creative-upscaler": ImageModel.CREATIVE_UPSCALER,
            "clarity-upscaler": ImageModel.CLARITY_UPSCALER,
            "recraft-upscale": ImageModel.RECRAFT_UPSCALE,
        }
        model_enum = upscale_models.get(model, ImageModel.CREATIVE_UPSCALER)
        
        # Call fal.ai upscaler
        result = await gen.edit_image(
            prompt=f"Upscale image by {scale}x",
            image_url=image_url,
            model=model_enum,
        )
        
        generation_time = time.time() - start_time
        
        # Save to S3 if available
        s3_urls = []
        if s3_storage and s3_storage.s3_client and result.success:
            for img in result.images:
                s3_url = await s3_storage.save_generated_image(
                    img.get("url"), user_id, org_id, f"Upscaled {scale}x"
                )
                s3_urls.append(s3_url)
        
        return {
            "success": result.success,
            "original_url": image_url,
            "upscaled_images": result.images,
            "s3_urls": s3_urls,
            "scale": scale,
            "model": model,
            "generation_time": generation_time,
            "error": result.error
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/enhance/remove-object", tags=["Enhance"])
async def remove_object(
    image_url: str = Form(..., description="URL of image"),
    mask_url: str = Form(..., description="Mask URL (white=remove, black=keep)"),
    user_id: str = Form(..., description="User ID for tracking"),
    org_id: str = Form(default=None, description="Organization ID"),
    model: str = Form(default="object-removal", description="Model: object-removal, bria-eraser, flux-inpainting"),
):
    """
    Remove objects from an image using a mask.
    
    The mask should be:
    - White areas = regions to remove/inpaint
    - Black areas = regions to keep
    """
    start_time = time.time()
    
    try:
        gen = get_generator()
        
        removal_models = {
            "object-removal": ImageModel.OBJECT_REMOVAL,
            "bria-eraser": ImageModel.BRIA_ERASER,
        }
        model_enum = removal_models.get(model, ImageModel.OBJECT_REMOVAL)
        
        result = await gen.edit_image(
            prompt="Remove object from image",
            image_url=image_url,
            mask_url=mask_url,
            model=model_enum,
        )
        
        generation_time = time.time() - start_time
        
        s3_urls = []
        if s3_storage and s3_storage.s3_client and result.success:
            for img in result.images:
                s3_url = await s3_storage.save_generated_image(
                    img.get("url"), user_id, org_id, "Object removed"
                )
                s3_urls.append(s3_url)
        
        return {
            "success": result.success,
            "original_url": image_url,
            "processed_images": result.images,
            "s3_urls": s3_urls,
            "model": model,
            "generation_time": generation_time,
            "error": result.error
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/enhance/remove-text", tags=["Enhance"])
async def remove_text(
    image_url: str = Form(..., description="URL of image with text to remove"),
    user_id: str = Form(..., description="User ID for tracking"),
    org_id: str = Form(default=None, description="Organization ID"),
):
    """
    Automatically detect and remove text from an image.
    Uses AI to identify and cleanly remove text overlays.
    """
    start_time = time.time()
    
    try:
        gen = get_generator()
        
        result = await gen.edit_image(
            prompt="Remove all text from image",
            image_url=image_url,
            model=ImageModel.TEXT_REMOVAL,
        )
        
        generation_time = time.time() - start_time
        
        s3_urls = []
        if s3_storage and s3_storage.s3_client and result.success:
            for img in result.images:
                s3_url = await s3_storage.save_generated_image(
                    img.get("url"), user_id, org_id, "Text removed"
                )
                s3_urls.append(s3_url)
        
        return {
            "success": result.success,
            "original_url": image_url,
            "processed_images": result.images,
            "s3_urls": s3_urls,
            "model": "text-removal",
            "generation_time": generation_time,
            "error": result.error
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/enhance/style", tags=["Enhance"])
async def apply_style(
    image_url: str = Form(..., description="URL of image to style"),
    style_image_url: str = Form(default=None, description="URL of style reference image"),
    style_prompt: str = Form(default=None, description="Style description (e.g., 'oil painting', 'watercolor')"),
    user_id: str = Form(..., description="User ID for tracking"),
    org_id: str = Form(default=None, description="Organization ID"),
    strength: float = Form(default=0.7, ge=0.0, le=1.0, description="Style strength (0-1)"),
):
    """
    Apply artistic style to an image.
    
    Provide either:
    - style_image_url: Reference image for style transfer
    - style_prompt: Text description of desired style
    """
    start_time = time.time()
    
    if not style_image_url and not style_prompt:
        raise HTTPException(status_code=400, detail="Provide either style_image_url or style_prompt")
    
    try:
        gen = get_generator()
        
        prompt = style_prompt or "Apply artistic style from reference"
        
        result = await gen.edit_image(
            prompt=prompt,
            image_url=image_url,
            model=ImageModel.STYLE_TRANSFER,
            strength=strength,
        )
        
        generation_time = time.time() - start_time
        
        s3_urls = []
        if s3_storage and s3_storage.s3_client and result.success:
            for img in result.images:
                s3_url = await s3_storage.save_generated_image(
                    img.get("url"), user_id, org_id, f"Style: {style_prompt or 'transferred'}"
                )
                s3_urls.append(s3_url)
        
        return {
            "success": result.success,
            "original_url": image_url,
            "styled_images": result.images,
            "s3_urls": s3_urls,
            "style_prompt": style_prompt,
            "strength": strength,
            "generation_time": generation_time,
            "error": result.error
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/enhance/background", tags=["Enhance"])
async def change_background(
    image_url: str = Form(..., description="URL of image (subject will be extracted)"),
    background_prompt: str = Form(..., description="Description of new background"),
    user_id: str = Form(..., description="User ID for tracking"),
    org_id: str = Form(default=None, description="Organization ID"),
    model: str = Form(default="background-change", description="Model: background-change, add-background"),
):
    """
    Change the background of an image.
    
    The subject is automatically extracted and placed on a new AI-generated background.
    
    Examples:
    - "white studio background"
    - "tropical beach at sunset"
    - "modern office interior"
    """
    start_time = time.time()
    
    try:
        gen = get_generator()
        
        bg_models = {
            "background-change": ImageModel.BACKGROUND_CHANGE,
            "add-background": ImageModel.ADD_BACKGROUND,
        }
        model_enum = bg_models.get(model, ImageModel.BACKGROUND_CHANGE)
        
        result = await gen.edit_image(
            prompt=background_prompt,
            image_url=image_url,
            model=model_enum,
        )
        
        generation_time = time.time() - start_time
        
        s3_urls = []
        if s3_storage and s3_storage.s3_client and result.success:
            for img in result.images:
                s3_url = await s3_storage.save_generated_image(
                    img.get("url"), user_id, org_id, f"Background: {background_prompt[:50]}"
                )
                s3_urls.append(s3_url)
        
        return {
            "success": result.success,
            "original_url": image_url,
            "processed_images": result.images,
            "s3_urls": s3_urls,
            "background_prompt": background_prompt,
            "model": model,
            "generation_time": generation_time,
            "error": result.error
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/enhance/relight", tags=["Enhance"])
async def relight_image(
    image_url: str = Form(..., description="URL of image to relight"),
    lighting_prompt: str = Form(default="natural daylight", description="Lighting description"),
    user_id: str = Form(..., description="User ID for tracking"),
    org_id: str = Form(default=None, description="Organization ID"),
):
    """
    Adjust the lighting of an image.
    
    Examples:
    - "warm golden hour sunlight"
    - "cool blue studio lighting"
    - "dramatic side lighting"
    - "soft diffused natural light"
    """
    start_time = time.time()
    
    try:
        gen = get_generator()
        
        result = await gen.edit_image(
            prompt=lighting_prompt,
            image_url=image_url,
            model=ImageModel.RELIGHTING,
        )
        
        generation_time = time.time() - start_time
        
        s3_urls = []
        if s3_storage and s3_storage.s3_client and result.success:
            for img in result.images:
                s3_url = await s3_storage.save_generated_image(
                    img.get("url"), user_id, org_id, f"Relight: {lighting_prompt[:50]}"
                )
                s3_urls.append(s3_url)
        
        return {
            "success": result.success,
            "original_url": image_url,
            "relit_images": result.images,
            "s3_urls": s3_urls,
            "lighting_prompt": lighting_prompt,
            "generation_time": generation_time,
            "error": result.error
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# ADMIN ENDPOINTS - DYNAMIC MODEL MANAGEMENT
# =============================================================================

class AddModelRequest(BaseModel):
    """Request body for adding a new model to Image_Models collection"""
    model_id: str = Field(..., description="Unique model ID (e.g., 'flux2-pro')")
    name: str = Field(..., description="Display name (e.g., 'FLUX 2 Pro')")
    fal_endpoint: str = Field(..., description="fal.ai endpoint (e.g., 'fal-ai/flux-2-pro')")
    category: str = Field(default="general", description="Category: flux2, ideogram, recraft, google, etc.")
    credits: int = Field(default=1, description="Credits required per generation")
    tier: str = Field(default="Standard", description="Tier: Free, Standard, Pro, Premium")
    is_edit_model: bool = Field(default=False, description="Is this an editing model?")
    enabled: bool = Field(default=True, description="Is model enabled?")
    best_for: List[str] = Field(default=[], description="Best use cases list")
    model_size: Dict[str, float] = Field(
        default={
            "512x512": 0.5,
            "768x1024": 0.8,
            "576x1024": 0.6,
            "1024x768": 0.8,
            "1024x576": 0.6,
            "1024x1024": 1.0,
            "1536x640": 1.2,
            "640x1536": 1.2
        },
        description="Size to credit multiplier mapping"
    )


class UpdateModelRequest(BaseModel):
    """Request body for updating a model in Image_Models collection"""
    name: Optional[str] = None
    fal_endpoint: Optional[str] = None
    category: Optional[str] = None
    credits: Optional[int] = None
    tier: Optional[str] = None
    is_edit_model: Optional[bool] = None
    enabled: Optional[bool] = None
    best_for: Optional[List[str]] = None
    model_size: Optional[Dict[str, float]] = None


@app.get("/admin/models", tags=["Admin"])
async def list_all_models(include_disabled: bool = Query(default=False, description="Include disabled models")):
    """
    List all dynamic models from database.
    Returns both enabled and disabled models if include_disabled=true.
    """
    if models_storage is None or models_storage.collection is None:
        return {
            "success": True,
            "source": "hardcoded",
            "message": "Dynamic models not configured. Using hardcoded MODEL_MAP.",
            "models": [{"model_id": k, "fal_endpoint": v.value} for k, v in MODEL_MAP.items()],
            "total": len(MODEL_MAP)
        }
    
    models = await models_storage.get_all_models(include_disabled=include_disabled)
    return {
        "success": True,
        "source": "database",
        "models": models,
        "total": len(models)
    }


@app.get("/admin/models/{model_id}", tags=["Admin"])
async def get_model_details(model_id: str):
    """Get details of a specific model"""
    if models_storage is None or models_storage.collection is None:
        # Check hardcoded models
        if model_id in MODEL_MAP:
            return {
                "success": True,
                "source": "hardcoded",
                "model": {
                    "model_id": model_id,
                    "fal_endpoint": MODEL_MAP[model_id].value
                }
            }
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    
    model = await models_storage.get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    
    return {"success": True, "model": model}


@app.post("/admin/models", tags=["Admin"])
async def add_new_model(request: AddModelRequest):
    """
    Add a new model to the database.
    This allows adding fal.ai models without code changes.
    """
    if models_storage is None or models_storage.collection is None:
        raise HTTPException(status_code=503, detail="Dynamic models storage not configured")
    
    try:
        model_data = request.model_dump()
        inserted_id = await models_storage.add_model(model_data)
        
        return {
            "success": True,
            "message": f"Model '{request.model_id}' added successfully",
            "id": inserted_id,
            "model_id": request.model_id,
            "name": request.name,
            "fal_endpoint": request.fal_endpoint,
            "category": request.category,
            "credits": request.credits,
            "tier": request.tier
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/admin/models/{model_id}", tags=["Admin"])
async def update_model(model_id: str, request: UpdateModelRequest):
    """Update an existing model's settings"""
    if models_storage is None or models_storage.collection is None:
        raise HTTPException(status_code=503, detail="Dynamic models storage not configured")
    
    # Filter out None values
    updates = {k: v for k, v in request.model_dump().items() if v is not None}
    
    if not updates:
        raise HTTPException(status_code=400, detail="No updates provided")
    
    success = await models_storage.update_model(model_id, updates)
    
    if not success:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found or no changes made")
    
    return {
        "success": True,
        "message": f"Model '{model_id}' updated successfully",
        "updates": updates
    }


@app.delete("/admin/models/{model_id}", tags=["Admin"])
async def disable_model(model_id: str, hard_delete: bool = Query(default=False, description="Permanently delete")):
    """
    Disable or permanently delete a model.
    By default, models are soft-deleted (disabled).
    Use hard_delete=true to permanently remove.
    """
    if models_storage is None or models_storage.collection is None:
        raise HTTPException(status_code=503, detail="Dynamic models storage not configured")
    
    if hard_delete:
        success = await models_storage.hard_delete_model(model_id)
        action = "permanently deleted"
    else:
        success = await models_storage.delete_model(model_id)
        action = "disabled"
    
    if not success:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    
    return {
        "success": True,
        "message": f"Model '{model_id}' {action} successfully"
    }


@app.post("/admin/models/{model_id}/enable", tags=["Admin"])
async def enable_model(model_id: str):
    """Re-enable a disabled model"""
    if models_storage is None or models_storage.collection is None:
        raise HTTPException(status_code=503, detail="Dynamic models storage not configured")
    
    success = await models_storage.update_model(model_id, {"enabled": True})
    
    if not success:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    
    return {
        "success": True,
        "message": f"Model '{model_id}' enabled successfully"
    }


@app.post("/admin/models/sync-hardcoded", tags=["Admin"])
async def sync_hardcoded_models():
    """
    Sync all hardcoded models to the database.
    This is useful for initial setup or after adding new models to code.
    """
    if models_storage is None or models_storage.collection is None:
        raise HTTPException(status_code=503, detail="Dynamic models storage not configured")
    
    added = 0
    skipped = 0
    
    for model_id, model_enum in MODEL_MAP.items():
        try:
            existing = await models_storage.get_model(model_id)
            if existing:
                skipped += 1
                continue
            
            # Get specifications from MODEL_SPECIFICATIONS if available
            spec = MODEL_SPECIFICATIONS.get(model_id, {})
            
            model_data = {
                "model_id": model_id,
                "fal_endpoint": model_enum.value,
                "name": spec.get("name", model_id.replace("-", " ").title()),
                "category": model_id.split("-")[0] if "-" in model_id else "general",
                "description": f"{spec.get('name', model_id)} - Image generation",
                "price_per_mp": spec.get("price_per_image", 0.025),
                "best_for": spec.get("best_for", []),
                "is_edit_model": spec.get("requires_image", "edit" in model_id.lower()),
                "default_guidance_scale": 3.5,
                "default_steps": 28,
                "enabled": True,
                # Include model specification fields
                "type": spec.get("type", "text_to_image"),
                "requires_image": spec.get("requires_image", False),
                "max_input_images": spec.get("max_input_images", 0),
                "max_output_images": spec.get("max_output_images", 4),  # Use spec value, not default
            }
            await models_storage.add_model(model_data)
            added += 1
        except Exception as e:
            print(f"Failed to sync {model_id}: {e}")
    
    return {
        "success": True,
        "message": f"Synced {added} models, skipped {skipped} existing",
        "added": added,
        "skipped": skipped,
        "total_hardcoded": len(MODEL_MAP)
    }


# =============================================================================
# STARTUP & SHUTDOWN
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    global s3_storage, db_storage, prompt_enhancer, models_storage
    
    print("🚀 Grovio AI Image Generation API starting...")
    print("   Marketing-focused image generation with 50+ optimized models")
    
    # Initialize S3 storage
    s3_storage = S3ImageStorage()
    print(f"📦 S3 Storage: {'✅ Connected' if s3_storage.s3_client else '❌ Not configured'}")
    
    # Initialize MongoDB storage
    db_storage = ImageMiniAppStorage()
    print(f"🗄️ MongoDB Storage: {'✅ Connected' if db_storage.collection is not None else '❌ Not configured'}")
    
    # Initialize Dynamic Models Storage
    models_storage = DynamicModelsStorage()
    print(f"🔧 Dynamic Models: {'✅ Ready' if models_storage.collection is not None else '❌ Not configured (using hardcoded)'}")
    
    # Initialize GPT Prompt Enhancer
    prompt_enhancer = GPTPromptEnhancer()
    print(f"🤖 GPT Enhancer: {'✅ Ready' if prompt_enhancer.client else '❌ Not configured (fallback mode)'}")
    
    # Check for API key (support both GROVIO_API_KEY and FAL_KEY)
    api_key = os.getenv("GROVIO_API_KEY") or os.getenv("FAL_KEY")
    if api_key:
        print("✅ GROVIO_API_KEY configured")
        try:
            get_generator()
            print("✅ Grovio AI Generator initialized successfully")
        except Exception as e:
            print(f"⚠️ Generator initialization failed: {e}")
    else:
        print("⚠️ GROVIO_API_KEY not set - set environment variable before making requests")
    
    print("📖 API docs available at /docs")
    print("🎯 Marketing endpoints at /marketing/*")
    print("🔧 Admin endpoints at /admin/*")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("👋 Grovio AI Image Generation API shutting down...")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    print(f"🚀 Starting server on {host}:{port}")
    uvicorn.run(
        "api_server:app",
        host=host,
        port=port,
        reload=True,
        log_level="info",
    )

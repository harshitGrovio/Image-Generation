"""
Bytedance Seedream Client for Grovio AI
Uses BytePlus ModelArk API (OpenAI-compatible) for Seedream image generation

This client provides an interface to Bytedance's Seedream models through their
BytePlus ModelArk API, which is OpenAI-compatible.
"""

import os
import asyncio
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

# Load .env file for environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not available, rely on system environment variables


# BytePlus ModelArk is OpenAI-compatible
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️ OpenAI SDK not installed. Run: pip install openai")


@dataclass
class BytedanceGenerationResult:
    """Result of Bytedance image generation"""
    success: bool
    images: List[Dict[str, Any]]  # List of {url, width, height, content_type}
    seed: Optional[int] = None
    prompt: Optional[str] = None
    model: Optional[str] = None
    error: Optional[str] = None
    revised_prompt: Optional[str] = None


class BytedanceSeedreamClient:
    """
    Bytedance Seedream Image Generation Client
    
    Uses BytePlus ModelArk API which is OpenAI-compatible.
    Supports Seedream 4.0, 4.5 for text-to-image generation.
    
    Usage:
        client = BytedanceSeedreamClient(api_key="your-api-key")
        result = await client.generate_image(
            prompt="A beautiful sunset",
            model="seedream-v4.5",
            size="1024x1024"
        )
    """
    
    # BytePlus ModelArk API base URL (Singapore region)
    BASE_URL = "https://ark.ap-southeast.bytepluses.com/api/v3"
    
    # Map our model IDs to BytePlus model names
    # Based on BytePlus ModelArk documentation
    # Model IDs include version date suffix: seedream-4-5-YYMMDD
    MODEL_MAPPING = {
        # Text-to-image models
        "seedream-v4.5": "seedream-4-5-251128",      # Seedream 4.5 (latest)
        "seedream-v4": "seedream-4-0-250828",        # Seedream 4.0
        # Edit models (image-to-image) - using generation models for now
        # Edit-specific models may need to be activated separately
        "seedream-v4.5-edit": "seedream-4-5-251128",
        "seedream-v4-edit": "seedream-4-0-250828",
    }
    
    # fal.ai model IDs to our short names
    FAL_TO_SHORT = {
        "fal-ai/bytedance/seedream/v4.5/text-to-image": "seedream-v4.5",
        "fal-ai/bytedance/seedream/v4.5/edit": "seedream-v4.5-edit",
        "fal-ai/bytedance/seedream/v4/text-to-image": "seedream-v4",
        "fal-ai/bytedance/seedream/v4/edit": "seedream-v4-edit",
    }
    
    # Supported sizes for Seedream models
    # Note: Seedream 4.5 requires at least 3,686,400 pixels (1920x1920)
    SUPPORTED_SIZES = [
        "1920x1920",  # Square 1:1 (3.7MP)
        "1664x2432",  # Portrait 3:4ish (4.0MP)
        "1472x2560",  # Portrait 9:16 (3.8MP)
        "2432x1664",  # Landscape 4:3ish (4.0MP)
        "2560x1472",  # Landscape 16:9 (3.8MP)
    ]
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Bytedance Seedream Client
        
        Args:
            api_key: Bytedance API key. If not provided, uses BYTEDANCE_API_KEY env variable
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI SDK not installed. Run: pip install openai")
        
        self.api_key = api_key or os.getenv("BYTEDANCE_API_KEY")
        if not self.api_key:
            raise ValueError("BYTEDANCE_API_KEY not provided. Set BYTEDANCE_API_KEY environment variable or pass api_key parameter")
        
        # Initialize OpenAI client with BytePlus base URL
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.BASE_URL
        )
        
        print("✅ Bytedance Seedream Client initialized")
    
    def is_seedream_model(self, model_id: str) -> bool:
        """Check if a model ID is a Seedream model"""
        # Check short name
        if model_id in self.MODEL_MAPPING:
            return True
        # Check fal.ai format
        if model_id in self.FAL_TO_SHORT:
            return True
        # Check if it contains "seedream"
        return "seedream" in model_id.lower()
    
    def is_edit_model(self, model_id: str) -> bool:
        """Check if a model is an edit (image-to-image) model"""
        return "edit" in model_id.lower()
    
    def _get_bytedance_model(self, model_id: str) -> str:
        """Convert our model ID to BytePlus model name"""
        # Handle fal.ai format
        if model_id in self.FAL_TO_SHORT:
            model_id = self.FAL_TO_SHORT[model_id]
        
        # Get BytePlus model name
        return self.MODEL_MAPPING.get(model_id, "seedream-4-5-251128")
    
    def _validate_size(self, width: int, height: int) -> str:
        """Validate and return size string"""
        size_str = f"{width}x{height}"
        if size_str in self.SUPPORTED_SIZES:
            return size_str
        
        # Find closest supported size based on aspect ratio
        target_ratio = width / height
        best_size = "1920x1920"
        best_diff = float('inf')
        
        for size in self.SUPPORTED_SIZES:
            w, h = map(int, size.split('x'))
            ratio = w / h
            diff = abs(ratio - target_ratio)
            if diff < best_diff:
                best_diff = diff
                best_size = size
        
        print(f"⚠️ Size {size_str} not supported, using {best_size}")
        return best_size
    
    async def generate_image(
        self,
        prompt: str,
        model: str = "seedream-v4.5",
        width: int = 1024,
        height: int = 1024,
        num_images: int = 1,
        seed: Optional[int] = None,
        **kwargs
    ) -> BytedanceGenerationResult:
        """
        Generate images using Bytedance Seedream model
        
        Args:
            prompt: Text description of the image to generate
            model: Model to use (seedream-v4.5, seedream-v4, etc.)
            width: Image width
            height: Image height
            num_images: Number of images to generate (1-4)
            seed: Random seed for reproducibility
            **kwargs: Additional arguments (ignored for compatibility)
        
        Returns:
            BytedanceGenerationResult with generated images
        """
        try:
            # Get BytePlus model name
            bytedance_model = self._get_bytedance_model(model)
            
            # Validate size
            size = self._validate_size(width, height)
            
            print(f"🎨 Generating image with Bytedance {bytedance_model}")
            print(f"   Prompt: {prompt[:50]}...")
            print(f"   Size: {size}")
            
            # Call BytePlus API (OpenAI-compatible)
            # Run synchronous call in thread to make it async
            response = await asyncio.to_thread(
                self.client.images.generate,
                model=bytedance_model,
                prompt=prompt,
                size=size,
                n=min(num_images, 4),
            )
            
            # Parse response
            images = []
            w, h = map(int, size.split('x'))
            
            for img_data in response.data:
                images.append({
                    "url": img_data.url,
                    "width": w,
                    "height": h,
                    "content_type": "image/png",
                })
            
            revised_prompt = None
            if hasattr(response.data[0], 'revised_prompt'):
                revised_prompt = response.data[0].revised_prompt
            
            print(f"✅ Generated {len(images)} image(s) successfully")
            
            return BytedanceGenerationResult(
                success=True,
                images=images,
                prompt=prompt,
                model=bytedance_model,
                revised_prompt=revised_prompt,
            )
            
        except Exception as e:
            print(f"❌ Bytedance generation failed: {e}")
            return BytedanceGenerationResult(
                success=False,
                images=[],
                error=str(e),
                model=model,
            )
    
    async def edit_image(
        self,
        image_url: str,
        prompt: str,
        model: str = "seedream-v4.5-edit",
        image_urls: Optional[List[str]] = None,
        width: int = 1024,
        height: int = 1024,
        **kwargs
    ) -> BytedanceGenerationResult:
        """
        Edit images using Bytedance Seededit model
        
        Note: BytePlus ModelArk image edit API may have different requirements.
        This implementation follows the OpenAI-compatible format.
        
        Args:
            image_url: Primary URL of the image to edit
            prompt: Description of the edit to make
            model: Model to use (seedream-v4.5-edit, seedream-v4-edit)
            image_urls: List of image URLs (for multi-image editing)
            width: Output width
            height: Output height
            **kwargs: Additional arguments
        
        Returns:
            BytedanceGenerationResult with edited images
        """
        try:
            # Get BytePlus model name
            bytedance_model = self._get_bytedance_model(model)
            
            # Validate size
            size = self._validate_size(width, height)
            
            # Collect all image URLs
            all_urls = []
            if image_urls:
                all_urls.extend(image_urls)
            elif image_url:
                all_urls.append(image_url)
            
            print(f"✏️ Editing image with Bytedance {bytedance_model}")
            print(f"   Prompt: {prompt[:50]}...")
            print(f"   Input images: {len(all_urls)}")
            
            # For edit operations, we use the images.edit endpoint
            # Note: This may need adjustment based on exact BytePlus API requirements
            # The OpenAI-compatible API typically expects a file, but some providers
            # accept URLs directly
            
            # Try using the generate endpoint with image reference
            # (BytePlus may support this in their Seedream edit model)
            response = await asyncio.to_thread(
                self.client.images.generate,
                model=bytedance_model,
                prompt=f"{prompt} [Reference image: {all_urls[0]}]" if all_urls else prompt,
                size=size,
                n=1,
            )
            
            # Parse response
            images = []
            w, h = map(int, size.split('x'))
            
            for img_data in response.data:
                images.append({
                    "url": img_data.url,
                    "width": w,
                    "height": h,
                    "content_type": "image/png",
                })
            
            print(f"✅ Edited image successfully")
            
            return BytedanceGenerationResult(
                success=True,
                images=images,
                prompt=prompt,
                model=bytedance_model,
            )
            
        except Exception as e:
            print(f"❌ Bytedance edit failed: {e}")
            return BytedanceGenerationResult(
                success=False,
                images=[],
                error=str(e),
                model=model,
            )


# Singleton instance for easy access
_bytedance_client: Optional[BytedanceSeedreamClient] = None


def get_bytedance_client() -> Optional[BytedanceSeedreamClient]:
    """Get or create the singleton Bytedance client"""
    global _bytedance_client
    
    if _bytedance_client is None:
        try:
            _bytedance_client = BytedanceSeedreamClient()
        except (ImportError, ValueError) as e:
            print(f"⚠️ Bytedance client not available: {e}")
            return None
    
    return _bytedance_client


# For quick testing
if __name__ == "__main__":
    import asyncio
    
    async def test():
        client = BytedanceSeedreamClient()
        result = await client.generate_image(
            prompt="A colorful sunset over ocean waves, vibrant colors, professional photography",
            model="seedream-v4.5",
            width=1920,
            height=1920,
        )
        print(f"\nResult: {result}")
        if result.success and result.images:
            print(f"Image URL: {result.images[0]['url']}")
    
    asyncio.run(test())

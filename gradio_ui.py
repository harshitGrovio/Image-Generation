"""
Grovio AI Image Generation UI
A simple web interface to test marketing creative generation

Usage:
    python gradio_ui.py
    
Then open http://localhost:7860 in your browser
"""

import gradio as gr
import asyncio
import os
from typing import Optional, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import Grovio AI generator
from fal_image_generator import (
    GrovioImageGenerator,
    ImageModel,  # 50 marketing-optimized models
    ImageSize,
    SIZE_DIMENSIONS,
)

# =============================================================================
# GLOBAL GENERATOR
# =============================================================================

generator: Optional[GrovioImageGenerator] = None


def get_generator() -> GrovioImageGenerator:
    """Get or create Grovio AI generator instance"""
    global generator
    if generator is None:
        api_key = os.getenv("GROVIO_API_KEY") or os.getenv("FAL_KEY")
        if not api_key:
            raise ValueError("GROVIO_API_KEY not set. Please set your Grovio AI API key in .env file")
        generator = GrovioImageGenerator(api_key=api_key)
    return generator


# =============================================================================
# MODEL & SIZE OPTIONS
# =============================================================================

MODEL_OPTIONS = {
    # FLUX 2 Models (Latest)
    "⭐ FLUX.2 Pro (Best Quality, $0.05/MP)": "fal-ai/flux-2-pro",
    "⭐ FLUX.2 Dev ($0.025/MP)": "fal-ai/flux-2-dev",
    "⭐ FLUX.2 Flex (Adjustable, $0.06/MP)": "fal-ai/flux-2-flex",
    # Google Nano Banana
    "🍌 Nano Banana Pro (Google, $0.02/MP)": "fal-ai/nano-banana-pro",
    "🍌 Nano Banana (Gemini Flash, $0.01/MP)": "fal-ai/nano-banana",
    # FLUX 1 Models
    "FLUX Schnell (Fast, $0.003/MP)": "fal-ai/flux/schnell",
    "FLUX Dev (High Quality, $0.025/MP)": "fal-ai/flux/dev",
    "FLUX Pro ($0.05/MP)": "fal-ai/flux-pro",
    "FLUX Pro v1.1 ($0.04/MP)": "fal-ai/flux-pro/v1.1",
    "FLUX Pro Ultra ($0.06/MP)": "fal-ai/flux-pro/v1.1-ultra",
    "FLUX LoRA ($0.025/MP)": "fal-ai/flux-lora",
    "FLUX Realism ($0.025/MP)": "fal-ai/flux-realism",
    # Other Models
    "Seedream V4 ($0.03/img)": "fal-ai/bytedance/seedream/v4/text-to-image",
    "Stable Diffusion XL ($0.01/MP)": "fal-ai/stable-diffusion-xl",
    "AuraFlow ($0.01/MP)": "fal-ai/aura-flow",
}

# Image Editing Models (subset of unified ImageModel)
EDIT_MODEL_OPTIONS = {
    "⭐ FLUX Kontext Pro (Best, $0.05/MP)": "fal-ai/flux-pro/kontext",
    "FLUX Kontext Dev (Fast, $0.025/MP)": "fal-ai/flux-kontext/dev",
    "FLUX Kontext Max (Premium, $0.08/MP)": "fal-ai/flux-kontext/max",
    "FLUX Kontext LoRA ($0.025/MP)": "fal-ai/flux-kontext-lora",
    "🍌 Nano Banana Pro Edit ($0.02/MP)": "fal-ai/nano-banana-pro/edit",
    "FLUX Inpainting ($0.025/MP)": "fal-ai/flux-lora/inpainting",
    "FLUX General Inpaint ($0.025/MP)": "fal-ai/flux-general/inpainting",
}

SIZE_OPTIONS = {
    "512x512 (Square)": "square",
    "1024x1024 (Square HD)": "square_hd",
    "768x1024 (Portrait 3:4)": "portrait_3_4",
    "576x1024 (Portrait 9:16)": "portrait_9_16",
    "1024x768 (Landscape 4:3)": "landscape_4_3",
    "1024x576 (Landscape 16:9)": "landscape_16_9",
}


# =============================================================================
# GENERATION FUNCTION
# =============================================================================

def generate_images(
    prompt: str,
    model_name: str,
    size_name: str,
    num_images: int,
    guidance_scale: float,
    num_steps: int,
    seed: Optional[int],
    negative_prompt: str,
    output_format: str,
    safety_checker: bool,
):
    """Generate images with the given parameters"""
    
    if not prompt.strip():
        return None, "❌ Please enter a prompt"
    
    try:
        gen = get_generator()
        
        # Get model and size
        model_id = MODEL_OPTIONS.get(model_name, "fal-ai/flux/dev")
        size_key = SIZE_OPTIONS.get(size_name, "square_hd")
        
        # Find matching enums
        model_enum = ImageModel.FLUX_DEV
        for m in ImageModel:
            if m.value == model_id:
                model_enum = m
                break
        
        size_enum = ImageSize.SQUARE_1024
        for s in ImageSize:
            if s.value == size_key:
                size_enum = s
                break
        
        # Handle seed
        actual_seed = seed if seed and seed > 0 else None
        
        # Handle negative prompt
        actual_negative = negative_prompt.strip() if negative_prompt.strip() else None
        
        # Generate
        result = asyncio.run(gen.generate(
            prompt=prompt,
            model=model_enum,
            size=size_enum,
            num_images=int(num_images),
            guidance_scale=guidance_scale,
            num_inference_steps=int(num_steps),
            seed=actual_seed,
            negative_prompt=actual_negative,
            enable_safety_checker=safety_checker,
            output_format=output_format,
        ))
        
        if not result.success:
            return None, f"❌ Generation failed: {result.error}"
        
        # Get image URLs
        image_urls = [img["url"] for img in result.images]
        
        # Build status message
        dims = SIZE_DIMENSIONS.get(size_key, {"width": 1024, "height": 1024})
        status = f"""✅ Generated {len(result.images)} image(s)
        
**Model:** {model_id}
**Size:** {dims['width']}x{dims['height']}
**Seed:** {result.seed}
**Estimated Cost:** ${result.cost_estimate:.4f}
"""
        
        return image_urls, status
        
    except Exception as e:
        return None, f"❌ Error: {str(e)}"


def edit_image(
    image_file,
    prompt: str,
    model_name: str,
    strength: float,
    guidance_scale: float,
    num_steps: int,
    seed: Optional[int],
):
    """Edit an image with the given parameters"""
    import base64
    import fal_client
    
    if image_file is None:
        return None, "❌ Please upload an image"
    
    if not prompt.strip():
        return None, "❌ Please enter an edit prompt"
    
    try:
        gen = get_generator()
        
        # Get model
        model_id = EDIT_MODEL_OPTIONS.get(model_name, "fal-ai/flux-kontext/dev")
        
        # Find matching enum (using unified ImageModel)
        model_enum = ImageModel.FLUX_KONTEXT_DEV
        for m in ImageModel:
            if m.value == model_id:
                model_enum = m
                break
        
        # Handle seed
        actual_seed = seed if seed and seed > 0 else None
        
        # Upload image to Grovio AI storage
        print(f"📤 Uploading image to Grovio AI storage...")
        image_url = fal_client.upload_file(image_file)
        print(f"✅ Image uploaded: {image_url}")
        
        # Edit image
        result = asyncio.run(gen.edit_image(
            image_url=image_url,
            prompt=prompt,
            model=model_enum,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=int(num_steps),
            seed=actual_seed,
        ))
        
        if not result.success:
            return None, f"❌ Edit failed: {result.error}"
        
        # Get image URLs
        image_urls = [img["url"] for img in result.images]
        
        status = f"""✅ Image edited successfully!

**Model:** {model_id}
**Seed:** {result.seed}
**Estimated Cost:** ${result.cost_estimate:.4f}
"""
        
        return image_urls, status
        
    except Exception as e:
        return None, f"❌ Error: {str(e)}"


# =============================================================================
# GRADIO INTERFACE
# =============================================================================

def create_ui():
    """Create the Gradio interface"""
    
    with gr.Blocks() as demo:
        
        gr.Markdown("""
        # 🎨 Grovio AI Image Generator
        
        Generate marketing creatives using 50+ optimized AI models. Features auto-settings and marketing prompt enhancement.
        
        **Best for:** Social media, advertising, e-commerce, branding, logos, product photography
        """)
        
        with gr.Tabs():
            # =================================================================
            # TAB 1: Text to Image
            # =================================================================
            with gr.TabItem("📝 Text to Image"):
                with gr.Row():
                    # Left column - Input
                    with gr.Column(scale=1):
                        t2i_prompt = gr.Textbox(
                            label="Prompt",
                            placeholder="A beautiful sunset over mountains, photorealistic, 8k resolution...",
                            lines=3,
                        )
                        
                        t2i_negative = gr.Textbox(
                            label="Negative Prompt (optional)",
                            placeholder="blurry, low quality, distorted...",
                            lines=2,
                        )
                        
                        with gr.Row():
                            t2i_model = gr.Dropdown(
                                choices=list(MODEL_OPTIONS.keys()),
                                value="⭐ FLUX.2 Dev ($0.025/MP)",
                                label="Model",
                            )
                            
                            t2i_size = gr.Dropdown(
                                choices=list(SIZE_OPTIONS.keys()),
                                value="1024x1024 (Square HD)",
                                label="Size",
                            )
                        
                        with gr.Row():
                            t2i_num_images = gr.Slider(
                                minimum=1,
                                maximum=4,
                                value=1,
                                step=1,
                                label="Number of Images",
                            )
                            
                            t2i_format = gr.Radio(
                                choices=["jpeg", "png"],
                                value="jpeg",
                                label="Format",
                            )
                        
                        with gr.Accordion("Advanced Settings", open=False):
                            t2i_guidance = gr.Slider(
                                minimum=1.0,
                                maximum=20.0,
                                value=3.5,
                                step=0.5,
                                label="Guidance Scale",
                            )
                            
                            t2i_steps = gr.Slider(
                                minimum=1,
                                maximum=50,
                                value=28,
                                step=1,
                                label="Inference Steps",
                            )
                            
                            t2i_seed = gr.Number(
                                label="Seed (optional)",
                                value=0,
                                precision=0,
                            )
                            
                            t2i_safety = gr.Checkbox(
                                label="Enable Safety Checker",
                                value=True,
                            )
                        
                        t2i_btn = gr.Button("🎨 Generate Images", variant="primary", size="lg")
                    
                    # Right column - Output
                    with gr.Column(scale=1):
                        t2i_gallery = gr.Gallery(
                            label="Generated Images",
                            columns=2,
                            rows=2,
                            height=500,
                        )
                        
                        t2i_status = gr.Markdown()
                
                # Example prompts
                gr.Examples(
                    examples=[
                        ["A majestic lion in the African savanna at golden hour, photorealistic, 8k"],
                        ["A futuristic cyberpunk city at night with neon lights and flying cars"],
                        ["A serene Japanese garden with cherry blossoms and a koi pond"],
                        ["An astronaut riding a horse on Mars, digital art, highly detailed"],
                        ["A cozy cabin in the mountains during winter, snow falling, warm light"],
                    ],
                    inputs=[t2i_prompt],
                    label="Example Prompts",
                )
                
                # Connect button
                t2i_btn.click(
                    fn=generate_images,
                    inputs=[
                        t2i_prompt,
                        t2i_model,
                        t2i_size,
                        t2i_num_images,
                        t2i_guidance,
                        t2i_steps,
                        t2i_seed,
                        t2i_negative,
                        t2i_format,
                        t2i_safety,
                    ],
                    outputs=[t2i_gallery, t2i_status],
                )
            
            # =================================================================
            # TAB 2: Image Editing
            # =================================================================
            with gr.TabItem("✏️ Image Editing"):
                with gr.Row():
                    # Left column - Input
                    with gr.Column(scale=1):
                        edit_image_input = gr.Image(
                            label="Upload Image to Edit",
                            type="filepath",
                            height=300,
                        )
                        
                        edit_prompt = gr.Textbox(
                            label="Edit Prompt",
                            placeholder="Change the background to a beach sunset...",
                            lines=3,
                        )
                        
                        edit_model = gr.Dropdown(
                            choices=list(EDIT_MODEL_OPTIONS.keys()),
                            value="FLUX Kontext Dev (Fast, $0.025/MP)",
                            label="Edit Model",
                        )
                        
                        with gr.Accordion("Advanced Settings", open=False):
                            edit_strength = gr.Slider(
                                minimum=0.1,
                                maximum=1.0,
                                value=0.85,
                                step=0.05,
                                label="Edit Strength",
                                info="Higher = more change",
                            )
                            
                            edit_guidance = gr.Slider(
                                minimum=1.0,
                                maximum=20.0,
                                value=3.5,
                                step=0.5,
                                label="Guidance Scale",
                            )
                            
                            edit_steps = gr.Slider(
                                minimum=10,
                                maximum=50,
                                value=28,
                                step=1,
                                label="Inference Steps",
                            )
                            
                            edit_seed = gr.Number(
                                label="Seed (optional)",
                                value=0,
                                precision=0,
                            )
                        
                        edit_btn = gr.Button("✏️ Edit Image", variant="primary", size="lg")
                    
                    # Right column - Output
                    with gr.Column(scale=1):
                        edit_gallery = gr.Gallery(
                            label="Edited Images",
                            columns=2,
                            rows=2,
                            height=500,
                        )
                        
                        edit_status = gr.Markdown()
                
                gr.Markdown("""
                **Kontext Edit Tips:**
                - Describe what you want to change: "Change the shirt color to blue"
                - Add elements: "Add sunglasses to the person"
                - Change style: "Make it look like a watercolor painting"
                - Change background: "Replace background with a tropical beach"
                """)
                
                # Connect button
                edit_btn.click(
                    fn=edit_image,
                    inputs=[
                        edit_image_input,
                        edit_prompt,
                        edit_model,
                        edit_strength,
                        edit_guidance,
                        edit_steps,
                        edit_seed,
                    ],
                    outputs=[edit_gallery, edit_status],
                )
        
        # Footer
        gr.Markdown("""
        ---
        **Models:**
        - ⭐ **FLUX.2**: Latest flagship models with best quality
        - 🍌 **Nano Banana**: Google's Gemini-based image generation
        - **Kontext**: Intelligent image editing that understands context
        """)
    
    return demo


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Check for API key (support both GROVIO_API_KEY and FAL_KEY)
    api_key = os.getenv("GROVIO_API_KEY") or os.getenv("FAL_KEY")
    if not api_key:
        print("⚠️  GROVIO_API_KEY not set!")
        print("   Please create a .env file with: GROVIO_API_KEY=your-api-key")
        exit(1)
    
    print("🎨 Starting Grovio AI Image Generator UI...")
    print("   50+ marketing-optimized models available")
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )

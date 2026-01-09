# Grovio AI Image Generation API

A FastAPI-based image generation & editing service using **32 production-ready models** with GPT-5.1 prompt enhancement.

> **Powered by Grovio AI** - Your marketing creative generation platform

## 🚀 Key Features

- 🎨 **50+ Models**: Text-to-image, editing, upscaling, and utility models
- 🛠️ **Dynamic Model Management**: Add/Update models at runtime via Admin APIs
- 🤖 **GPT-5.1 Prompt Enhancement**: AI-powered prompt optimization
- 🧠 **Brand Memory**: Inject brand guidelines (colors, voice, visual style) into prompts
- ✏️ **Image Editing**: Edit images with up to 6-10 input images
- 📐 **Multiple Angles**: Generate 8 camera angles from single image
- ⚙️ **Auto-Settings**: Optimal settings per model
- 📡 **SSE Streaming**: Real-time progress updates
- ☁️ **S3 Storage**: Automatic upload to AWS S3
- 🗄️ **MongoDB Integration**: Generation history tracking
- 💰 **Cost Estimation**: Know the cost before generating
- 🔗 **Brand API**: REST endpoints for other miniapps to access brand data

## 📦 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables

```bash
cp .env.example .env
# Edit .env with your credentials:
# - GROVIO_API_KEY (fal.ai API key)
# - OPENAI_API_KEY (for GPT-5.1 prompt enhancement)
# - AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
# - MONGODB_CONNECTION_STRING
```

### 3. Run Server

```bash
# Local
uvicorn api_server:app --reload --port 5005

# Docker
docker-compose up -d
```

### 4. Access API

- **API Docs**: http://localhost:5005/docs
- **Health Check**: http://localhost:5005/health
- **Models List**: http://localhost:5005/models

### 5. Initialize Models (Optional)

If running for the first time with a fresh database:

```bash
curl -X POST http://localhost:5005/admin/models/sync-hardcoded
```

---

## 📊 Available Models (50+ Total)

*(List expands dynamically via Admin API)*

### Text-to-Image (Featured)

| Model | API Value | Max Output | Price | Notes |
|-------|-----------|------------|-------|-------|
| **FLUX 2 Pro** | `flux2-pro` | 1 | $0.03 | Premium quality |
| **FLUX 2 Dev** | `flux2-dev` | 4 | $0.012 | Balanced |
| **Nano Banana Pro** | `gemini-3-pro` | 4 | $0.15 | Typography |
| **Seedream V4.5** | `seedream-v4.5` | 4 | $0.04 | Stylized |
| **Z-Image Turbo** | `z-image-turbo` | 4 | $0.005 | Fastest |
| **Reve** | `reve` | 4 | $0.04 | State-of-the-art |
| **Seedream V4** | `seedream-v4` | 6 | $0.03 | 4K support |
| **Dreamina V3.1** | `dreamina-v3.1` | 4 | $0.027 | Portraits |
| **Ideogram V3** | `ideogram-v3` | 8 | $0.08 | Typography + Styles |
| **GPT Image 1.5** | `gpt-image-1.5` | 4 | $0.05 | Transparent backgrounds |

### Image Editing (11 Models)

| Model | API Value | Max Input | Price | Notes |
|-------|-----------|-----------|-------|-------|
| **FLUX 2 Pro Edit** | `flux2-pro-edit` | 3 | $0.03 | Premium |
| **FLUX 2 Dev Edit** | `flux2-dev-edit` | 3 | $0.024 | Standard |
| **Nano Banana Pro Edit** | `gemini-3-pro-edit` | 10 | $0.15 | Multi-image |
| **Seedream V4.5 Edit** | `seedream-v4.5-edit` | 3 | $0.04 | Style transfer |
| **Seedream V4 Edit** | `seedream-v4-edit` | 10 | $0.03 | High-res |
| **Reve Edit** | `reve-edit` | 1 | $0.04 | Standard edit |
| **Reve Fast Edit** | `reve-fast-edit` | 1 | $0.04 | Fast editing |
| **Reve Remix** | `reve-remix` | 6 | $0.04 | XML tag refs |
| **Reve Fast Remix** | `reve-fast-remix` | 6 | $0.04 | Fast remix |
| **Ideogram V3 Reframe** | `ideogram-v3-reframe` | 1 | $0.08 | Aspect ratio change |
| **GPT Image 1.5 Edit** | `gpt-image-1.5-edit` | 10 | $0.05 | Multi-image editing |

### Upscaling (3 Models)

| Model | API Value | Price | Notes |
|-------|-----------|-------|-------|
| **Creative Upscaler** | `creative-upscaler` | $0.02 | Artistic detail |
| **Clarity Upscaler** | `clarity-upscaler` | $0.02 | Sharpness |
| **Recraft Upscale** | `recraft-upscale` | $0.02 | Maximum quality |

> **Scale Parameter**: Use `scale=1` (enhancement), `scale=2` (2x), or `scale=4` (4x) for upscalers.

### Utility Models (8 Models)

| Model | API Value | Price | Notes |
|-------|-----------|-------|-------|
| **Object Removal** | `object-removal` | $0.02 | Remove objects |
| **Bria Eraser** | `bria-eraser` | $0.02 | Commercial-safe |
| **Text Removal** | `text-removal` | $0.02 | Remove text |
| **Style Transfer** | `style-transfer` | $0.02 | Apply styles |
| **Background Change** | `background-change` | $0.02 | Change background |
| **Add Background** | `add-background` | $0.02 | Add background |
| **Relighting** | `relighting` | $0.02 | Adjust lighting |
| **Multiple Angles** | `multiple-angles` | $0.03 | 8 camera angles |

---

## 🧠 Brand Memory - NEW!

Generate **on-brand images** by automatically injecting brand guidelines into AI prompts.

### What Brand Guidance Includes

| Brand Element | Injected Into Prompt |
|---------------|---------------------|
| **Colors** | Hex codes (#C084FC, #A855F7, etc.) |
| **Visual Style** | Border radius, shadows, spacing |
| **Brand Personality** | Attributes (innovative, expert, etc.) |
| **Tone & Voice** | "We are" / "We are NOT" guidelines |
| **Logo** | Optional: Use as base image with edit models |

### POST /brand-generate - Brand-Aware Generation

Fetches complete brand guidance from `brand_summaries` collection and injects into GPT prompt enhancement.

**Example - GPT transforms this:**
```
Input:  "Create a car leasing advertisement"
Output: "Car leasing advertisement with sleek dark background (#000000), 
         gradient glows in brand purples (#C084FC, #A855F7), 
         expert and innovative mood, professional marketing aesthetic..."
```

**Usage:**
```bash
curl -X POST "http://localhost:5005/brand-generate" \
  -F "prompt=Create a marketing banner for our brand" \
  -F "org_id=ORG_J64NKRUF_Y3ACAQH1C5ACDXB7" \
  -F "user_id=user_123" \
  -F "model=flux2-pro"
```

**Optional - Use Brand Logo (edit models only):**
```bash
curl -X POST "http://localhost:5005/brand-generate" \
  ... \
  -F "model=gemini-3-pro-edit" \
  -F "use_logo=true"
```

### Brand REST API (For Other Miniapps)

| Endpoint | Returns |
|----------|---------|
| `GET /brand/{org_id}` | Complete brand data (colors, style, voice, logo) |
| `GET /brand/{org_id}/creative` | Colors, typography, visual style |
| `GET /brand/{org_id}/voice` | Tone, vocabulary, writing examples |
| `GET /brand/{org_id}/prompt-context` | **Ready-to-use prompt string** |
| `GET /brands` | List all available brands |

**Usage from other miniapps (e.g., Tweet Creator):**

```python
import requests
response = requests.get("http://images-api:5005/brand/ORG_123/prompt-context")
brand_context = response.json()["prompt_context"]
# Inject into your LLM: "You are a social media manager. {brand_context}"
```

---

## 🖼️ POST /generate - Main Endpoint

Unified endpoint for **image generation** and **image editing**.

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `prompt` | string | **Yes** | - | Image description |
| `user_id` | string | **Yes** | - | User ID |
| `org_id` | string | No | - | Organization ID |
| `model` | string | No | `flux2-dev` | Model to use |
| `enhance_prompt` | bool | No | `false` | GPT-5.1 enhancement |
| `auto_settings` | bool | No | `true` | Use optimal settings |
| `brand_details` | bool | No | `false` | **Brand guidance** (requires org_id) |
| `base_images` | File(s) | No | - | Image(s) to edit |
| `size` | string | No | `1024x1024` | Image size |
| `num_images` | int | No | 1 | Output images (1-4) |

> **`brand_details=true`**: Injects brand colors, visual style, and personality into the prompt. For edit models, auto-uses brand logo as base image.

### Generate Image

```bash
curl -X POST "http://localhost:5005/generate" \
  -F "prompt=A beautiful sunset over mountains" \
  -F "user_id=user_123" \
  -F "model=flux2-pro"
```

### Edit Image

```bash
curl -X POST "http://localhost:5005/generate" \
  -F "prompt=Change background to black" \
  -F "user_id=user_123" \
  -F "model=reve-edit" \
  -F "base_images=@my_image.jpg"
```

### Generate 8 Camera Angles (NEW!)

```bash
curl -X POST "http://localhost:5005/generate" \
  -F "prompt=Character turnaround" \
  -F "user_id=user_123" \
  -F "model=multiple-angles" \
  -F "base_images=@character.png"
```

> Generates 8 angles: front, right 45°, left 45°, right profile, left profile, high angle, low angle, close-up

### Professional Headshot

```bash
curl -X POST "http://localhost:5005/generate" \
  -F "prompt=Professional corporate headshot" \
  -F "user_id=user_123" \
  -F "model=reve-edit" \
  -F "base_images=@selfie.jpg"
```

### Upscale Image

```bash
curl -X POST "http://localhost:5005/generate" \
  -F "prompt=Enhance and upscale" \
  -F "user_id=user_123" \
  -F "model=creative-upscaler" \
  -F "base_images=@low_res.jpg"
```

---

## 📋 GET /models - List All Models

### Get All Models

```bash
curl "http://localhost:5005/models"
```

### Filter by Use-Case

```bash
curl "http://localhost:5005/models?query=create-images"              # All 9 text-to-image
curl "http://localhost:5005/models?query=edit-images"                # All 10 editing models
curl "http://localhost:5005/models?query=image-upscale"              # All 3 upscalers
curl "http://localhost:5005/models?query=character-multiple-angles"  # Multi-angle generation
curl "http://localhost:5005/models?query=headshots"                  # Professional headshots
curl "http://localhost:5005/models?query=quick-mockups"              # 3D mockups
```

**20 Use-Cases Available:** instagram-post, instagram-story, facebook-ads, linkedin-content, tiktok-thumbnails, pinterest-pins, youtube-thumbnails, display-ads, hero-banners, product-listings, logos-posters, vector-icons, headshots, quick-mockups, print-billboard, fashion-tryon, image-editing, image-upscale, character-multiple-angles, **create-images**

### Response Fields

Each use-case query returns:

| Field | Description |
|-------|-------------|
| `models` | List of recommended models |
| `system_prompt` | AI guidance for generating images |
| `prompt_tips` | Tips for better prompts |
| `editing_models` | Available editing models (when applicable) |
| `platform_sizes` | Recommended sizes for the platform |

---

## 📊 Available Sizes

| Size Alias | Dimensions | Best For |
|------------|------------|----------|
| `1024x1024` | 1024×1024 | Instagram posts |
| `768x1024` | 768×1024 | Pinterest, portraits |
| `576x1024` | 576×1024 | Instagram Stories |
| `1024x768` | 1024×768 | Blog headers |
| `1024x576` | 1024×576 | YouTube thumbnails |

---

## 🔧 Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROVIO_API_KEY` | Yes | fal.ai API key |
| `OPENAI_API_KEY` | No | OpenAI key for GPT-5.1 |
| `AWS_ACCESS_KEY_ID` | No | AWS access key for S3 |
| `AWS_SECRET_ACCESS_KEY` | No | AWS secret key |
| `AWS_REGION` | No | AWS region (default: ap-south-1) |
| `MONGODB_CONNECTION_STRING` | No | MongoDB URI |
| `PORT` | No | Server port (default: 5005) |

---

## 🐳 Docker Deployment

```bash
# Build and run
docker-compose up -d

# View logs
docker-compose logs -f grovio-images-api

# Rebuild
docker-compose down && docker-compose up --build
```

---

**Last Updated:** December 2025
**Version:** 5.3 (50+ Models + Brand Memory)
"# Image-Generation" 

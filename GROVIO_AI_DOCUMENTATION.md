# Grovio AI - Complete API Documentation

## Table of Contents
1. [Overview](#overview)
2. [Available Models (30)](#available-models)
3. [Marketing Use-Cases (20)](#marketing-use-cases)
4. [API Endpoints](#api-endpoints)
5. [Brand Memory API](#brand-memory-api)
6. [Admin Management](#admin-management)
7. [Usage Examples](#usage-examples)
8. [Pricing](#pricing)

---

## Overview

**Grovio AI** is a marketing-focused image generation & editing platform providing access to **50+ production-ready models** optimized for social media, advertising, and content creation. Built on enterprise-grade infrastructure with GPT-5.1 prompt enhancement and dynamic model management.

### Platform Highlights
- **Company**: Grovio AI
- **Focus**: Marketing & Growth Creatives
- **Models**: 50+ production-ready models (expanding dynamically)
- **New in v5.3**: Brand Memory API, Brand-aware image generation
- **Features**: GPT-5.1 prompt enhancement, auto-settings, S3 storage, MongoDB tracking, Admin APIs, Brand Memory
- **Infrastructure**: Powered by fal.ai
- **API Port**: 5005 (default)

---

## Available Models

### Text-to-Image (10 Models)

| Model | API ID | fal.ai Endpoint | Max Output | Price |
|-------|--------|-----------------|------------|-------|| **FLUX 2 Pro** | `flux2-pro` | `fal-ai/flux-2-pro` | 1 | $0.03 |
| **FLUX 2 Dev** | `flux2-dev` | `fal-ai/flux-2` | 4 | $0.012 |
| **Gemini 3 Pro** | `gemini-3-pro` | `fal-ai/gemini-3-pro-image-preview` | 4 | $0.15 |
| **Seedream V4.5** | `seedream-v4.5` | `fal-ai/bytedance/seedream/v4.5/text-to-image` | 4 | $0.04 |
| **Z-Image Turbo** | `z-image-turbo` | `fal-ai/z-image/turbo` | 4 | $0.005 |
| **Reve** | `reve` | `fal-ai/reve/text-to-image` | 4 | $0.04 |
| **Seedream V4** | `seedream-v4` | `fal-ai/bytedance/seedream/v4/text-to-image` | 6 | $0.03 |
| **Dreamina V3.1** | `dreamina-v3.1` | `fal-ai/bytedance/dreamina/v3.1/text-to-image` | 4 | $0.027 |
| **Ideogram V3** | `ideogram-v3` | `fal-ai/ideogram/v3` | 8 | $0.08 |
| **GPT Image 1.5** | `gpt-image-1.5` | `fal-ai/gpt-image-1.5` | 4 | $0.05 |

### Image Editing (11 Models)

| Model | API ID | fal.ai Endpoint | Max Input | Max Output | Price |
|-------|--------|-----------------|-----------|------------|-------|
| **FLUX 2 Pro Edit** | `flux2-pro-edit` | `fal-ai/flux-2-pro/edit` | 3 | 4 | $0.03 |
| **FLUX 2 Dev Edit** | `flux2-dev-edit` | `fal-ai/flux-2/edit` | 3 | 4 | $0.024 |
| **Gemini 3 Pro Edit** | `gemini-3-pro-edit` | `fal-ai/gemini-3-pro-image-preview/edit` | 10 | 4 | $0.15 |
| **Seedream V4.5 Edit** | `seedream-v4.5-edit` | `fal-ai/bytedance/seedream/v4.5/edit` | 3 | 4 | $0.04 |
| **Seedream V4 Edit** | `seedream-v4-edit` | `fal-ai/bytedance/seedream/v4/edit` | 10 | 6 | $0.03 |
| **Reve Edit** | `reve-edit` | `fal-ai/reve/edit` | 1 | 4 | $0.04 |
| **Reve Fast Edit** | `reve-fast-edit` | `fal-ai/reve/fast/edit` | 1 | 4 | $0.04 |
| **Reve Remix** | `reve-remix` | `fal-ai/reve/remix` | 6 | 4 | $0.04 |
| **Reve Fast Remix** | `reve-fast-remix` | `fal-ai/reve/fast/remix` | 6 | 4 | $0.04 |
| **Ideogram V3 Reframe** | `ideogram-v3-reframe` | `fal-ai/ideogram/v3/reframe` | 1 | 4 | $0.08 |
| **GPT Image 1.5 Edit** | `gpt-image-1.5-edit` | `fal-ai/gpt-image-1.5/edit` | 10 | 4 | $0.05 |

### Upscaling (3 Models)

| Model | API ID | fal.ai Endpoint | Price | Notes |
|-------|--------|-----------------|-------|-------|
| **Creative Upscaler** | `creative-upscaler` | `fal-ai/creative-upscaler` | $0.02 | Artistic detail |
| **Clarity Upscaler** | `clarity-upscaler` | `fal-ai/clarity-upscaler` | $0.02 | Sharpness focus |
| **Recraft Upscale** | `recraft-upscale` | `fal-ai/recraft/upscale/creative` | $0.02 | Maximum quality |

> **Upscaler Scale Parameter**: Use `scale=1` (enhancement only), `scale=2` (2x), or `scale=4` (4x) when calling `/generate`.

### Utility Models (8 Models)

| Model | API ID | fal.ai Endpoint | Price | Notes |
|-------|--------|-----------------|-------|-------|
| **Object Removal** | `object-removal` | `fal-ai/object-removal` | $0.02 | Remove objects |
| **Bria Eraser** | `bria-eraser` | `fal-ai/bria/eraser` | $0.02 | Commercial-safe |
| **Text Removal** | `text-removal` | `fal-ai/text-removal` | $0.02 | Remove text/watermarks |
| **Style Transfer** | `style-transfer` | `fal-ai/style-transfer` | $0.02 | Apply artistic styles |
| **Background Change** | `background-change` | `fal-ai/background-change` | $0.02 | Change background |
| **Add Background** | `add-background` | `fal-ai/add-background` | $0.02 | Add background to subject |
| **Relighting** | `relighting` | `fal-ai/relighting` | $0.02 | Adjust lighting |
| **Multiple Angles** | `multiple-angles` | `fal-ai/qwen-image-edit-2509-lora-gallery/multiple-angles` | $0.03 | 8 camera angles |

> **Reve Remix Note**: Use `<img>0</img>`, `<img>1</img>` in prompts to reference specific input images.

---

## Marketing Use-Cases

Each use-case includes a **system prompt** and **prompt tips** for optimal results.

### Social Media (6)

| Use-Case | Query | Models | System Prompt Focus |
|----------|-------|--------|---------------------|
| **Instagram Post** | `instagram-post` | 9 text-to-image | Scroll-stopping, vibrant colors |
| **Instagram Story** | `instagram-story` | 9 text-to-image | Full-screen immersive |
| **Facebook Ads** | `facebook-ads` | 9 text-to-image | High-converting, minimal text |
| **LinkedIn Content** | `linkedin-content` | 9 text-to-image | Professional B2B |
| **TikTok Thumbnails** | `tiktok-thumbnails` | 9 text-to-image | Gen-Z trendy |
| **Pinterest Pins** | `pinterest-pins` | 9 text-to-image | Aspirational lifestyle |

### Advertising (2)

| Use-Case | Query | Models | System Prompt Focus |
|----------|-------|--------|---------------------|
| **Display Ads** | `display-ads` | 9 text-to-image | IAB standard formats |
| **Hero Banners** | `hero-banners` | 9 text-to-image | Full-width cinematic |

### E-Commerce (1)

| Use-Case | Query | Models | System Prompt Focus |
|----------|-------|--------|---------------------|
| **Product Listings** | `product-listings` | 8 editing models | Clean white background (requires image) |

### Typography (2)

| Use-Case | Query | Models | System Prompt Focus |
|----------|-------|--------|---------------------|
| **Logos & Posters** | `logos-posters` | 9 text-to-image | Accurate text rendering |
| **Vector Icons** | `vector-icons` | 9 text-to-image | Clean, scalable designs |

### People & Fashion (1)

| Use-Case | Query | Models | System Prompt Focus |
|----------|-------|--------|---------------------|
| **Headshots** | `headshots` | 7 editing models | Professional corporate headshot |

### Video (1)

| Use-Case | Query | Models | System Prompt Focus |
|----------|-------|--------|---------------------|
| **YouTube Thumbnails** | `youtube-thumbnails` | 9 text-to-image | Click-worthy, high CTR |

### Special Use-Cases (5)

| Use-Case | Query | Models | System Prompt Focus |
|----------|-------|--------|---------------------|
| **Quick Mockups** | `quick-mockups` | 8 editing models | 3D product mockups |
| **Image Upscale** | `image-upscale` | 3 upscalers | 1x/2x/4x resolution |
| **Fashion Try-On** | `fashion-tryon` | 3 editing models | Virtual clothing try-on |
| **Character Multiple Angles** | `character-multiple-angles` | 1 model | 8 camera angles |
| **Image Editing** | `image-editing` | 10 editing models | General editing |

---

## API Endpoints

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | API info |
| `GET` | `/health` | Health check |
| `GET` | `/models` | List all 30 models |
| `GET` | `/models?query=create-images` | All 10 text-to-image models |
| `GET` | `/models?query=edit-images` | All 11 editing models |
| `GET` | `/models?query=image-upscale` | All 3 upscalers (use scale=1/2/4) |
| `POST` | `/generate` | Generate/Edit with SSE streaming (supports `brand_details=true`) |
| `GET` | `/history` | User generation history |

### Marketing Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/marketing/prompts` | Marketing prompt templates |
| `GET` | `/marketing/recommend` | Get recommended model |
| `POST` | `/marketing/enhance-prompt` | GPT-5.1 enhancement |

### Brand Memory Endpoints (NEW!)

**Brand Guidance** automatically injects brand colors, visual style, personality, and voice into prompts.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/brand-generate` | **Brand-aware generation** - injects brand guidance into prompt |
| `GET` | `/brand/{org_id}` | Complete brand data (colors, style, voice, logo) |
| `GET` | `/brand/{org_id}/creative` | Colors, typography, visual style |
| `GET` | `/brand/{org_id}/voice` | Tone, vocabulary, personality |
| `GET` | `/brand/{org_id}/prompt-context` | **Ready-to-use prompt string** for any LLM |
| `GET` | `/brands` | List all available brands |

**Optional**: `use_logo=true` - Uses brand logo as base image with edit models.


### Admin Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/admin/models` | List all dynamic models |
| `POST` | `/admin/models` | Add a new model configuration |
| `PUT` | `/admin/models/{id}` | Update an existing model |
| `DELETE` | `/admin/models/{id}` | Disable/Delete a model |
| `POST` | `/admin/models/sync-hardcoded` | Sync hardcoded models to DB |

### /models Response Fields

When querying by use-case (`?query=instagram-post`), the response includes:

| Field | Description |
|-------|-------------|
| `models` | List of recommended models for the use-case |
| `system_prompt` | AI guidance for generating images (what style/quality to aim for) |
| `prompt_tips` | Array of tips for writing better prompts |
| `editing_models` | List of editing models available for this use-case |
| `platform_sizes` | Recommended dimensions for the platform |
| `notes` | Additional usage notes |

---

## POST /generate Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `prompt` | string | **Yes** | - | Image description |
| `user_id` | string | **Yes** | - | User ID |
| `org_id` | string | No | - | Organization ID |
| `model` | string | No | `flux2-dev` | Model to use |
| `enhance_prompt` | bool | No | `false` | GPT-5.1 enhancement |
| `auto_settings` | bool | No | `true` | Use optimal settings |
| `base_images` | File(s) | No | - | Image(s) to edit |
| `size` | string | No | `1024x1024` | Image size |
| `num_images` | int | No | 1 | Images (1-8) |

### Available Sizes

| Size | Dimensions | Best For |
|------|------------|----------|
| `1024x1024` | 1024×1024 | Instagram, profile |
| `768x1024` | 768×1024 | Pinterest, portraits |
| `576x1024` | 576×1024 | Stories, TikTok |
| `1024x768` | 1024×768 | Blog headers |
| `1024x576` | 1024×576 | YouTube thumbnails |

---

## Usage Examples

### Generate Single Image

```bash
curl -X POST "http://localhost:5005/generate" \
  -F "prompt=A beautiful sunset" \
  -F "user_id=user_123" \
  -F "model=flux2-pro"
```

### Generate with Typography

```bash
curl -X POST "http://localhost:5005/generate" \
  -F "prompt=Logo with text 'GROVIO AI' in modern font" \
  -F "user_id=user_123" \
  -F "model=ideogram-v3"
```

### Edit Image

```bash
curl -X POST "http://localhost:5005/generate" \
  -F "prompt=Change background to beach at sunset" \
  -F "user_id=user_123" \
  -F "model=reve-edit" \
  -F "base_images=@my_image.jpg"
```

### Professional Headshot

```bash
curl -X POST "http://localhost:5005/generate" \
  -F "prompt=Professional corporate headshot, studio lighting" \
  -F "user_id=user_123" \
  -F "model=reve-edit" \
  -F "base_images=@selfie.jpg"
```

### Generate 8 Camera Angles (NEW!)

```bash
curl -X POST "http://localhost:5005/generate" \
  -F "prompt=Character turnaround" \
  -F "user_id=user_123" \
  -F "model=multiple-angles" \
  -F "base_images=@character.png"
```

> **Output**: 8 images with angles: front, right 45°, left 45°, right profile, left profile, high angle, low angle, close-up

### 3D Product Mockup

```bash
curl -X POST "http://localhost:5005/generate" \
  -F "prompt=Place logo on floating MacBook screen with realistic shadows" \
  -F "user_id=user_123" \
  -F "model=seedream-v4.5-edit" \
  -F "base_images=@logo.png"
```

### Upscale Image

```bash
curl -X POST "http://localhost:5005/generate" \
  -F "prompt=Enhance and upscale 4x" \
  -F "user_id=user_123" \
  -F "model=creative-upscaler" \
  -F "base_images=@low_res.jpg"
```

### Reve Remix (Multi-Image)

```bash
curl -X POST "http://localhost:5005/generate" \
  -F "prompt=Dress the model in <img>1</img> clothes" \
  -F "user_id=user_123" \
  -F "model=reve-remix" \
  -F "base_images=@person.jpg" \
  -F "base_images=@outfit.jpg"
```

---

## Pricing (Per Image at 1024×1024)

| Model | Price | Best For |
|-------|-------|----------|
| Z-Image Turbo | **$0.005** | Fast prototyping |
| FLUX 2 Dev | **$0.012** | Social media |
| Utility Models | **$0.02** | Background, upscaling |
| Dreamina V3.1 | **$0.027** | Portraits |
| FLUX 2 Pro | **$0.03** | Premium quality |
| Seedream V4 | **$0.03** | 4K content |
| Multiple Angles | **$0.03** | 8 angles = $0.24 total |
| Reve (all) | **$0.04** | State-of-the-art |
| Seedream V4.5 | **$0.04** | Stylized |
| Ideogram V3 | **$0.08** | Typography |
| Gemini 3 Pro | **$0.15** | Multi-image editing |

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROVIO_API_KEY` | Yes | fal.ai API key |
| `OPENAI_API_KEY` | No | GPT-5.1 enhancement |
| `AWS_ACCESS_KEY_ID` | No | S3 storage |
| `AWS_SECRET_ACCESS_KEY` | No | S3 storage |
| `MONGODB_CONNECTION_STRING` | No | History tracking |

---

## Docker Deployment

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
**Version:** 5.3 (50+ Models, Admin APIs, Brand Memory)

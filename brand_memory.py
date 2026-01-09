"""
Brand Memory Module for GrovioAI Images Workflows

Fetches brand guidelines from MongoDB brand_summaries collection
and formats them for GPT prompt enhancement.
"""

import os
from typing import Optional, Dict, Any
from datetime import datetime, timezone

# MongoDB imports
try:
    from pymongo import MongoClient
    from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False
    print("⚠️ MongoDB dependencies not available for BrandMemory")


class BrandMemory:
    """
    Fetch brand guidelines for on-brand image generation.
    
    Queries brand_summaries collection directly using the same MongoDB
    connection string as other components.
    
    Usage:
        brand_memory = BrandMemory()
        brand_data = await brand_memory.get_brand_context("ORG_J64NKRUF_Y3ACAQH1C5ACDXB7")
        prompt_context = brand_memory.format_for_prompt(brand_data)
    """
    
    def __init__(self):
        self.client = None
        self.db = None
        self.collection = None
        self.database_name = os.getenv("MONGODB_DB", "Grovio_MAIN_DB")
        self.collection_name = "brand_summaries"
        self._connect()
    
    def _get_connection_string(self) -> str:
        """Get MongoDB connection string from environment"""
        return (
            os.getenv("MONGODB_CONNECTION_STRING") or 
            os.getenv("MONGO_URI") or 
            ""
        )
    
    def _connect(self):
        """Establish MongoDB connection"""
        if not MONGODB_AVAILABLE:
            print("⚠️ BrandMemory: MongoDB not available")
            return
        
        connection_string = self._get_connection_string()
        if not connection_string:
            print("⚠️ BrandMemory: No MongoDB connection string")
            return
        
        try:
            self.client = MongoClient(connection_string, serverSelectionTimeoutMS=5000)
            # Test connection
            self.client.admin.command('ping')
            self.db = self.client[self.database_name]
            self.collection = self.db[self.collection_name]
            print(f"✅ BrandMemory connected to {self.database_name}.{self.collection_name}")
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            print(f"⚠️ BrandMemory: MongoDB connection failed: {e}")
            self.client = None
            self.db = None
            self.collection = None
    
    async def get_brand_context(self, org_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch brand data from brand_summaries collection.
        
        Args:
            org_id: Organization ID (e.g., "ORG_J64NKRUF_Y3ACAQH1C5ACDXB7")
            
        Returns:
            Dict with brand data or None if not found
        """
        if self.collection is None:
            return None

        
        try:
            # Try by organization_id first
            brand = self.collection.find_one(
                {"organization_id": org_id},
                {
                    "organization_id": 1,
                    "organization_name": 1,
                    "summary.brand_core": 1,
                    "summary.creative_guidelines": 1,
                    "summary.tone_voice": 1,
                }
            )
            
            if not brand:
                # Try by organization_name as fallback
                brand = self.collection.find_one(
                    {"organization_name": {"$regex": org_id, "$options": "i"}},
                    {
                        "organization_id": 1,
                        "organization_name": 1,
                        "summary.brand_core": 1,
                        "summary.creative_guidelines": 1,
                        "summary.tone_voice": 1,
                    }
                )
            
            if brand:
                creative_guidelines = brand.get("summary", {}).get("creative_guidelines", {})
                logo_usage = creative_guidelines.get("logo_usage", {})
                logo_url = logo_usage.get("logo_url", "") if isinstance(logo_usage, dict) else ""
                
                return {
                    "organization_id": brand.get("organization_id"),
                    "organization_name": brand.get("organization_name"),
                    "brand_core": brand.get("summary", {}).get("brand_core", {}),
                    "creative_guidelines": creative_guidelines,
                    "tone_voice": brand.get("summary", {}).get("tone_voice", {}),
                    "logo_url": logo_url,
                }
            
            return None
            
        except Exception as e:
            print(f"⚠️ BrandMemory: Error fetching brand: {e}")
            return None
    
    def format_for_prompt(self, brand_data: Dict[str, Any]) -> str:
        """
        Format brand data into a string suitable for GPT prompt injection.
        
        Args:
            brand_data: Dict from get_brand_context()
            
        Returns:
            Formatted string with brand guidelines for GPT
        """
        if not brand_data:
            return ""
        
        def safe_str(val):
            """Convert any value to string safely"""
            if isinstance(val, list):
                return ', '.join(str(v) for v in val if v)
            return str(val) if val else ""
        
        def safe_join(items, limit=5):
            """Safely join a list of items that might contain non-strings"""
            result = []
            for item in items[:limit]:
                if isinstance(item, (str, int, float)):
                    result.append(str(item))
                elif isinstance(item, dict):
                    # If it's a dict, try to get a 'name' or 'value' key
                    result.append(str(item.get('name', item.get('value', str(item)))))
                elif isinstance(item, list):
                    result.append(', '.join(str(x) for x in item))
                else:
                    result.append(str(item))
            return result
        
        parts = []
        org_name = brand_data.get("organization_name", "Unknown Brand")
        parts.append(f"Brand: {org_name}")
        
        # Extract color palette
        creative = brand_data.get("creative_guidelines", {})
        color_palette = creative.get("color_palette", [])
        if color_palette and isinstance(color_palette, list):
            colors = []
            for color in color_palette[:4]:  # Limit to 4 main colors
                if isinstance(color, dict):
                    name = color.get("name", "")
                    hex_code = color.get("hex", "")
                    if hex_code:
                        colors.append(f"{name}: {hex_code}" if name else hex_code)
                elif isinstance(color, str):
                    colors.append(color)
            if colors:
                parts.append(f"Brand Colors: {', '.join(colors)}")
        
        # Extract visual style
        visual_style = creative.get("visual_style", {})
        if visual_style and isinstance(visual_style, dict):
            style_attrs = []
            border_radius = visual_style.get("border_radius")
            if border_radius:
                style_attrs.append(f"rounded corners ({safe_str(border_radius)})")
            shadows = visual_style.get("shadows")
            if shadows:
                style_attrs.append(safe_str(shadows))
            if style_attrs:
                parts.append(f"Visual Style: {', '.join(style_attrs)}")
        
        # Extract typography
        typography = creative.get("typography", {})
        if isinstance(typography, dict) and typography.get("font_family"):
            parts.append(f"Typography: {safe_str(typography['font_family'])}")
        
        # Extract brand personality
        brand_core = brand_data.get("brand_core", {})
        personality = brand_core.get("brand_personality", {}) if isinstance(brand_core, dict) else {}
        if personality and isinstance(personality, dict):
            attributes = personality.get("attributes", [])
            if attributes and isinstance(attributes, list):
                attr_strs = safe_join(attributes, 5)
                if attr_strs:
                    parts.append(f"Brand Personality: {', '.join(attr_strs)}")
            tone = personality.get("tone", "")
            if tone:
                parts.append(f"Tone: {safe_str(tone)}")
        
        # Extract voice guidelines
        tone_voice = brand_data.get("tone_voice", {})
        voice_guidelines = tone_voice.get("voice_guidelines", {}) if isinstance(tone_voice, dict) else {}
        if voice_guidelines and isinstance(voice_guidelines, dict):
            we_are = voice_guidelines.get("we_are", [])
            if we_are and isinstance(we_are, list):
                we_are_strs = safe_join(we_are, 5)
                if we_are_strs:
                    parts.append(f"We are: {', '.join(we_are_strs)}")
            we_are_not = voice_guidelines.get("we_are_not", [])
            if we_are_not and isinstance(we_are_not, list):
                we_are_not_strs = safe_join(we_are_not, 3)
                if we_are_not_strs:
                    parts.append(f"We are NOT: {', '.join(we_are_not_strs)}")
        
        return "\n".join(parts)
    
    def get_brand_colors_hex(self, brand_data: Dict[str, Any]) -> list:
        """
        Extract just the hex color codes from brand data.
        
        Returns:
            List of hex color codes, e.g., ["#0066CC", "#00CC66"]
        """
        if not brand_data:
            return []
        
        creative = brand_data.get("creative_guidelines", {})
        color_palette = creative.get("color_palette", [])
        
        return [c.get("hex") for c in color_palette if c.get("hex")]


# Global instance
brand_memory = BrandMemory()

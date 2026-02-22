"""Image generation tool using Gemini API."""

import base64
import json
import logging
import os
from pathlib import Path
from typing import Any

from nanobot.agent.tools.base import Tool

logger = logging.getLogger(__name__)


class ImageGenTool(Tool):
    """Generate images using Gemini AI models."""

    @property
    def name(self) -> str:
        return "image_gen"

    @property
    def description(self) -> str:
        return (
            "Generate an image using AI (Gemini). Returns the file path of the generated image. "
            "The image will be automatically attached to your reply."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Text prompt describing the image to generate.",
                },
                "model": {
                    "type": "string",
                    "description": "Model to use. Default: cloudsway-gemini-2.5-flash-image",
                    "enum": [
                        "cloudsway-gemini-2.5-flash-image",
                        "cloudsway-gemini-3-pro-image",
                    ],
                },
            },
            "required": ["prompt"],
        }

    async def execute(self, **kwargs: Any) -> str:
        prompt = kwargs.get("prompt", "")
        model = kwargs.get("model", "cloudsway-gemini-2.5-flash-image")

        if not prompt:
            return "Error: prompt is required"

        try:
            import requests

            # Read API config
            cfg_path = Path.home() / ".nanobot" / "config.json"
            cfg = json.loads(cfg_path.read_text())
            custom = cfg["providers"]["custom"]
            api_base = custom["apiBase"]
            api_key = custom["apiKey"]

            proxy = os.environ.get("http_proxy") or os.environ.get("https_proxy")
            proxies = {"http": proxy, "https": proxy} if proxy else None

            resp = requests.post(
                f"{api_base}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                },
                proxies=proxies,
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()

            msg = data["choices"][0]["message"]
            images = msg.get("images", [])

            if not images:
                text = msg.get("content", "")[:300]
                return f"No image returned. Model response: {text}"

            img_url = images[0]["image_url"]["url"]
            _, b64data = img_url.split(",", 1)
            img_bytes = base64.b64decode(b64data)

            # Save to temp file (and clean up old generated images)
            import tempfile
            import time

            tmp_dir = tempfile.gettempdir()
            cutoff = time.time() - 24 * 3600  # 24 hours ago
            for old_file in Path(tmp_dir).glob("image_gen_*.png"):
                try:
                    if old_file.stat().st_mtime < cutoff:
                        old_file.unlink()
                        logger.debug(f"Cleaned up old image: {old_file}")
                except OSError:
                    pass

            output_path = os.path.join(
                tmp_dir, f"image_gen_{os.getpid()}_{id(img_bytes)}.png"
            )
            with open(output_path, "wb") as f:
                f.write(img_bytes)

            logger.info(f"Image generated: {output_path} ({len(img_bytes)} bytes)")
            return f"Image saved to {output_path}"

        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            return f"Error generating image: {e}"

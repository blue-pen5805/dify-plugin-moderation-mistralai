from collections.abc import Generator
from typing import Any

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage
from dify_plugin.file.file import File, FileType

import httpx
import json

class MistralAIModerationTool(Tool):
    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage]:
        text: str = tool_parameters.get("text")
        model = "mistral-moderation-latest"

        base_url = self.runtime.credentials.get('api_base') or "https://api.mistral.ai/v1"
        url = f"{base_url}/moderations"
        headers = {
            "Authorization": f"Bearer {self.runtime.credentials['api_key']}",
            "Content-Type": "application/json",
        }
        data = {
            "input": text,
            "model": model,
        }
        response = httpx.post(url, headers=headers, json=data)

        if response.is_success:
            json_response = response.json()
            result = json_response["results"][0]

            categories = result["categories"]
            category_scores = result["category_scores"]

            unsafe_score = max(category_scores.values())
            flagged_categories = [category for category, is_flagged in categories.items() if is_flagged]
            flagged = len(flagged_categories) > 0

            return [
                self.create_json_message(json_response),
                self.create_text_message("true" if flagged else "false"),
                self.create_variable_message("flagged", flagged),
                self.create_variable_message("unsafe_score", unsafe_score),
                self.create_variable_message("flagged_categories", flagged_categories),
                self.create_variable_message("category_scores", category_scores),
            ]
        else:
            raise ValueError(response.text)

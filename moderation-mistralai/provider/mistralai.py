from typing import Any

from dify_plugin import ToolProvider
from dify_plugin.errors.tool import ToolProviderCredentialValidationError

import httpx

class MistralAIProvider(ToolProvider):
    def _validate_credentials(self, credentials: dict[str, Any]) -> None:
        try:
            base_url = credentials.get('api_base') or "https://api.mistral.ai/v1"
            url = f"{base_url}/moderations"
            headers = {
                "Authorization": f"Bearer {credentials['api_key']}",
                "Content-Type": "application/json",
            }
            response = httpx.post(url, headers=headers, json={
                "input": "validate credentials",
                "model": "mistral-moderation-latest",
            })

            if response.is_success:
                pass
            else:
                raise ToolProviderCredentialValidationError(response.text)
        except Exception as e:
            raise ToolProviderCredentialValidationError(str(e))

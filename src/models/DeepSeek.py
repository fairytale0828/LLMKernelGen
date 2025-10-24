import os
from typing import List, Optional
import openai
from tenacity import retry, stop_after_attempt, wait_random_exponential

from models.Base import BaseModel


class DeepSeekModel(BaseModel):
    def __init__(
        self,
        model_id: str = "deepseek-chat",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        request_timeout: int = 60,
    ):
        api_key = api_key or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
        assert api_key and api_key.strip(), "no api key is provided."

        ds_url = (
            base_url
            or os.environ.get("DEEPSEEK_API_URL")
            or "https://api.deepseek.com/v1"   # DeepSeek 的 OpenAI 兼容端
        )

        self.model_id = model_id
        self.request_timeout = request_timeout

        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=ds_url, 
            timeout=request_timeout,
        )

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
    def generate(self, 
                 messages: List, 
                 temperature=0, 
                 presence_penalty=0, 
                 frequency_penalty=0, 
                 max_tokens=4096) -> str:
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=temperature,
            n=1,
            stream=False,
            stop=None,
            max_tokens=max_tokens,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            logit_bias=None,
            user=None
        )
        
        if not response or not hasattr(response, 'choices') or len(response.choices) == 0:
            raise ValueError("No response choices returned from the API.")

        return response.choices[0].message.content

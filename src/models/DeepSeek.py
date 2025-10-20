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
                 max_tokens=2048) -> str:
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




# from typing import List, Dict, Any, Optional
# from openai import APIStatusError, OpenAIError
# from loguru import logger
# from tenacity import retry, wait_random_exponential, stop_after_attempt
# import os
# from typing import List, Optional
# import openai
# from tenacity import retry, stop_after_attempt, wait_random_exponential

# from models.Base import BaseModel

# class DeepSeekModel(BaseModel):
#     def __init__(
#         self,
#         model_id: str = "deepseek-chat",
#         api_key: Optional[str] = None,
#         base_url: Optional[str] = None,
#         request_timeout: int = 60,
#     ):
#         assert api_key and api_key.strip(), "no api key is provided."

#         ds_url = (
#             base_url
#             or os.environ.get("DEEPSEEK_API_URL")
#             or "https://api.deepseek.com/v1"   # DeepSeek 的 OpenAI 兼容端
#         )

#         self.model_id = model_id
#         self.request_timeout = request_timeout

#         self.client = openai.OpenAI(
#             api_key=api_key,
#             base_url=ds_url, 
#             timeout=request_timeout,
#         )
#     @retry(wait=wait_random_exponential(min=1, max=30), stop=stop_after_attempt(2), reraise=True)
#     def generate(
#         self,
#         messages: List[Dict[str, Any]],
#         temperature: float = 0.2,
#         presence_penalty: float = 0.0,
#         frequency_penalty: float = 0.0,
#         max_tokens: int = 1024,  # 先保守
#         stop: Optional[List[str]] = None,
#         user: Optional[str] = None,
#         logit_bias: Optional[Dict[str, float]] = None,
#         stream: bool = False,
#     ) -> str:
#         # 只构造非 None 的参数，避免兼容端因 None 报 400
#         payload = {
#             "model": self.model_id,
#             "messages": messages,
#             "temperature": temperature,
#             "max_tokens": max_tokens,
#             "n": 1,
#             "stream": stream,
#         }
#         if stop is not None:
#             payload["stop"] = stop
#         # 如需这些再传；不需要就别带
#         if presence_penalty:
#             payload["presence_penalty"] = presence_penalty
#         if frequency_penalty:
#             payload["frequency_penalty"] = frequency_penalty
#         if logit_bias:
#             payload["logit_bias"] = logit_bias
#         if user:
#             payload["user"] = user

#         try:
#             resp = self.client.chat.completions.create(**payload)
#         except APIStatusError as e:
#             # 关键：把状态码 + 返回体打印出来
#             logger.error(
#                 "[DeepSeek] HTTP {} | {}",
#                 e.status_code,
#                 getattr(e.response, "text", ""),
#             )
#             raise
#         except OpenAIError as e:
#             logger.exception("[DeepSeek] OpenAIError: {}", e)
#             raise
#         except Exception as e:
#             logger.exception("[DeepSeek] General error: {}", e)
#             raise

#         if stream:
#             chunks = []
#             for chunk in resp:
#                 delta = chunk.choices[0].delta
#                 if delta and getattr(delta, "content", None):
#                     chunks.append(delta.content)
#             text = "".join(chunks).strip()
#             if not text:
#                 raise ValueError("Empty streamed response from DeepSeek API.")
#             return text

#         if not resp or not getattr(resp, "choices", None):
#             raise ValueError("No response choices returned from the API.")
#         return resp.choices[0].message.content or ""

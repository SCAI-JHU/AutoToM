import os
import requests
from openai.types.chat.chat_completion import ChatCompletion

HEADERS = {
    "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
    "Content-Type": "application/json",
}


def openrouter_request(prompt, model, seed):
    for ith_retry in range(3):
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=HEADERS,
                json={
                    "messages": [{"role": "user", "content": prompt}],
                    "model": model,
                    "logprobs": True,
                    "top_p": 0.01,
                    "top_logprobs": 5,
                    "temperature": 0.01,
                    "seed": seed,
                    "max_tokens": 1,
                    # https://openrouter.ai/docs/features/provider-routing#requiring-providers-to-support-all-parameters
                    "provider": {"require_parameters": True},
                },
            )
            response = ChatCompletion(**response.json())
            response.choices[0].logprobs.content[0].top_logprobs
            return response
        except TypeError:
            continue
    raise Exception("Failed to get logprobs")

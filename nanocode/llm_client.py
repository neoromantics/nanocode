import httpx
from typing import Optional, Callable, Dict, Any

class LLMResponse:
    def __init__(self, raw_json: Dict[str, Any]):
        self.raw_json = raw_json

async def send_request(
    api_url: str,
    payload: Dict[str, Any],
    setup_headers: Callable[[Dict[str, str]], None],
    on_chunk: Optional[Callable[[str], None]] = None,
    on_line: Optional[Callable[[str], None]] = None,
    finalize: Optional[Callable[[], LLMResponse]] = None
) -> LLMResponse:
    headers = {"Content-Type": "application/json"}
    setup_headers(headers)
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        if on_chunk or on_line:
            async with client.stream("POST", api_url, json=payload, headers=headers) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        if on_line:
                            on_line(data_str)
                if finalize:
                    return finalize()
                return LLMResponse({})
        else:
            response = await client.post(api_url, json=payload, headers=headers)
            response.raise_for_status()
            return LLMResponse(response.json())

async def fetch_json(
    url: str,
    setup_headers: Callable[[Dict[str, str]], None]
) -> Dict[str, Any]:
    headers = {}
    setup_headers(headers)
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, headers=headers)
        resp.raise_for_status()
        return resp.json()

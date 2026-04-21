# src/llm/openrouter_client.py

import requests
import json
import time
from enum import Enum
from typing import Optional

class TaskType(Enum):
    SCREENING = "screening"       # Alto volumen, rapidez
    SCENARIO = "scenario"         # Razonamiento numérico
    THESIS = "thesis"             # Máxima calidad
    MACRO = "macro"               # Síntesis

class OpenRouterClient:
    """
    Cliente OpenRouter que rota modelos gratuitos
    para mantenerse dentro de rate limits
    """
    
    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
    
    # Modelos gratuitos por tarea (en orden de preferencia)
    MODELS = {
        TaskType.SCREENING: [
            "meta-llama/llama-3.3-70b-instruct:free",
            "qwen/qwen-2.5-72b-instruct:free",
            "google/gemma-2-9b-it:free",
        ],
        TaskType.SCENARIO: [
            "deepseek/deepseek-r1:free",
            "meta-llama/llama-3.3-70b-instruct:free",
            "qwen/qwen-2.5-72b-instruct:free",
        ],
        TaskType.THESIS: [
            "qwen/qwen-2.5-72b-instruct:free",
            "meta-llama/llama-3.3-70b-instruct:free",
            "deepseek/deepseek-r1:free",
        ],
        TaskType.MACRO: [
            "meta-llama/llama-3.3-70b-instruct:free",
            "mistralai/mistral-7b-instruct:free",
        ]
    }
    
    def __init__(self, api_key: str, site_url: str = "", app_name: str = ""):
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": site_url,
            "X-Title": app_name
        }
        # Tracking de uso por modelo para evitar rate limits
        self.request_counts = {}
        self.last_request_time = {}
    
    def complete(
        self,
        prompt: str,
        task: TaskType,
        system: str = "",
        max_tokens: int = 1000,
        temperature: float = 0.1,
        require_json: bool = False
    ) -> str:
        """
        Completa con rotación automática si hay rate limit
        """
        models = self.MODELS[task]
        
        for model in models:
            try:
                result = self._call_model(
                    model=model,
                    prompt=prompt,
                    system=system,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    require_json=require_json
                )
                return result
                
            except RateLimitError:
                print(f"Rate limit en {model}, rotando...")
                time.sleep(2)
                continue
                
            except Exception as e:
                print(f"Error en {model}: {e}, rotando...")
                continue
        
        raise Exception("Todos los modelos han fallado")
    
    def _call_model(
        self,
        model: str,
        prompt: str,
        system: str,
        max_tokens: int,
        temperature: float,
        require_json: bool
    ) -> str:
        
        # Rate limiting propio: mínimo 3s entre requests al mismo modelo
        now = time.time()
        last = self.last_request_time.get(model, 0)
        if now - last < 3:
            time.sleep(3 - (now - last))
        
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        
        # Algunos modelos soportan JSON mode
        if require_json:
            payload["response_format"] = {"type": "json_object"}
        
        response = requests.post(
            self.BASE_URL,
            headers=self.headers,
            json=payload,
            timeout=60
        )
        
        self.last_request_time[model] = time.time()
        self.request_counts[model] = self.request_counts.get(model, 0) + 1
        
        if response.status_code == 429:
            raise RateLimitError(f"Rate limit: {model}")
        
        if response.status_code != 200:
            raise Exception(f"API Error {response.status_code}: {response.text}")
        
        data = response.json()
        
        # Limpiar JSON si require_json pero el modelo no lo soporta nativamente
        content = data["choices"][0]["message"]["content"]
        
        if require_json:
            content = self._extract_json(content)
        
        return content
    
    def _extract_json(self, text: str) -> str:
        """Extrae JSON aunque el modelo añada texto alrededor"""
        # Buscar el primer { y el último }
        start = text.find("{")
        end = text.rfind("}") + 1
        
        if start != -1 and end > start:
            json_str = text[start:end]
            # Validar que es JSON válido
            json.loads(json_str)
            return json_str
        
        raise Exception(f"No se encontró JSON válido en: {text[:200]}")
    
    def usage_report(self) -> dict:
        """Cuántas requests has usado por modelo"""
        return self.request_counts

class RateLimitError(Exception):
    pass

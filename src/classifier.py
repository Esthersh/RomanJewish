import os
from typing import List, Dict, Any, Tuple
import google.generativeai as genai
from openai import OpenAI
from src.data_loader import Keyword

class LLMProvider:
    def generate(self, prompt: str) -> str:
        raise NotImplementedError

class GeminiProvider(LLMProvider):
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash", temperature: float = 0.7, top_p: float = 0.95):
        genai.configure(api_key=api_key)
        self.generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            top_p=top_p
        )
        self.model = genai.GenerativeModel(model_name)

    def generate(self, prompt: str) -> str:
        try:
            response = self.model.generate_content(prompt, generation_config=self.generation_config)
            return response.text
        except Exception as e:
            print(f"Gemini Error: {e}")
            return ""

class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str, model_name: str = "gpt-4o", temperature: float = 0.7, top_p: float = 1.0):
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p

    def generate(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model_name,
                temperature=self.temperature,
                top_p=self.top_p
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI Error: {e}")
            return ""

class QwenProvider(LLMProvider):
    def __init__(self, api_key: str, model_name: str = "Qwen/Qwen2.5-72B-Instruct-Turbo", temperature: float = 0.7, top_p: float = 0.7):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.together.xyz/v1"
        )
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p

    def generate(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model_name,
                temperature=self.temperature,
                top_p=self.top_p
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Qwen/TogetherAI Error: {e}")
            return ""

import json
import importlib.util
import os

class Classifier:
    def __init__(
        self, 
        provider: str, 
        api_key: str, 
        prompt_path: str = "prompts/default.py",
        model_name: str = None,
        temperature: float = 0.7,
        top_p: float = 0.95,
        debug: bool = False
    ):
        self.provider_name = provider.lower()
        self.debug = debug
        
        # Default model names if not provided
        if not model_name:
            if self.provider_name == 'gemini':
                model_name = "gemini-1.5-flash"
            elif self.provider_name == 'openai':
                model_name = "gpt-4o"
            elif self.provider_name == 'qwen':
                model_name = "Qwen/Qwen2.5-72B-Instruct-Turbo"

        if self.provider_name == 'gemini':
            self.llm = GeminiProvider(api_key, model_name, temperature, top_p)
        elif self.provider_name == 'openai':
            self.llm = OpenAIProvider(api_key, model_name, temperature, top_p)
        elif self.provider_name == 'qwen':
            self.llm = QwenProvider(api_key, model_name, temperature, top_p)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        
        self.load_prompts(prompt_path)

    def load_prompts(self, prompt_path: str):
        try:
            # Load the module dynamically
            spec = importlib.util.spec_from_file_location("prompts_module", prompt_path)
            if spec and spec.loader:
                prompts_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(prompts_module)
                
                self.prompts = {
                    "classification_prompt": getattr(prompts_module, "CLASSIFICATION_PROMPT", ""),
                    "suggestion_prompt": getattr(prompts_module, "SUGGESTION_PROMPT", "")
                }
            else:
                raise FileNotFoundError(f"Could not load prompts from {prompt_path}")

        except Exception as e:
            print(f"Error loading prompts from {prompt_path}: {e}")
            raise e

    def format_keywords(self, keywords: List[Keyword]) -> str:
        # Organize by hierarchy for display
        # A simple indented list or path-based list
        # optimizing for token usage and clarity
        # Group by level 0
        
        # Let's map parent IDs to children
        tree = {} # parent_id -> list of keywords
        roots = []
        for kw in keywords:
            if kw.level == 0:
                roots.append(kw)
            
            pid = kw.parent_id
            if pid not in tree:
                tree[pid] = []
            tree[pid].append(kw)
            
        output = []
        for root in roots:
            output.append(f"- {root.name} (ID: {root.id})")
            children = tree.get(root.id, [])
            for child in children:
                output.append(f"  - {child.name} (ID: {child.id})")
        return "\n".join(output)

    def classify(self, text: str, keywords: List[Keyword], metadata: Dict[str, str] = {}) -> Tuple[List[str], List[str]]:
        """
        Returns: (matched_keyword_ids_or_names, new_suggested_keywords)
        """
        hierarchy_str = self.format_keywords(keywords)
        
        import re
        # Step 1: Classification
        prompt_1 = self.prompts.get("classification_prompt", "").format(
            hierarchy_str=hierarchy_str,
            text=text,
            source_name=metadata.get('source_name', 'Unknown'),
            group=metadata.get('group', 'Unknown'),
            name=metadata.get('name', 'Unknown'),
            Language=metadata.get('language', 'Hebrew')
        )
        
        if self.debug:
            print(f"\n[DEBUG] --- Classification Prompt ---\n{prompt_1}\n-----------------------------------")
        
        response_1 = self.llm.generate(prompt_1)
        
        if self.debug:
            print(f"\n[DEBUG] --- LLM Response 1 ---\n{response_1}\n------------------------------")
            
        # Regex to find the first number in a tuple-like structure (id, word)
        # Matches patterns like (123, "Word") or (123, 'Word')
        # We capture the digits inside the parenthesis before the comma
        matched_ids = re.findall(r'\((\d+)\s*,', response_1)
        
        # Fallback: if regex yields nothing, maybe it reverted to old CSV or just numbers?
        # But user explicitly asked for tuples. If the model fails to follow format, this might identify nothing.
        # Let's add a robust fallback to just finding IDs if the tuple parse fails, 
        # BUT strictly speaking we should follow the tuple structure to avoid false positives from the text part.
        if not matched_ids:
             # Try simple CSV backup just in case model ignores instruction
             matched_ids = [s.strip() for s in response_1.split(',') if s.strip().isdigit()]
        
        # Prepare subset of keywords for Step 2
        matched_kws_sub = [k for k in keywords if str(k.id) in matched_ids]
        # matched_str = self.format_keywords(matched_kws_sub) # Old bullet format
        # User wants list of tuples format: (id, 'name'), (id, 'name')
        matched_str = ", ".join([f"({k.id}, '{k.name}')" for k in matched_kws_sub])
        
        # Step 2: New Keyword Suggestion
        prompt_2 = self.prompts.get("suggestion_prompt", "").format(
            hierarchy_str=matched_str,
            text=text,
            source_name=metadata.get('source_name', 'Unknown'),
            group=metadata.get('group', 'Unknown'),
            name=metadata.get('name', 'Unknown')
        )

        if self.debug:
            print(f"\n[DEBUG] --- Suggestion Prompt ---\n{prompt_2}\n-------------------------------")

        response_2 = self.llm.generate(prompt_2)
        
        if self.debug:
            print(f"\n[DEBUG] --- LLM Response 2 ---\n{response_2}\n------------------------------")

        if "NONE" in response_2.upper():
            new_keywords = []
        else:
            new_keywords = [s.strip() for s in response_2.split(',')]
            
        return matched_ids, new_keywords


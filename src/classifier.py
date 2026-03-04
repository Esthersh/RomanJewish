import ast
import json
import importlib.util
from typing import List, Dict, Tuple
from openai import OpenAI
from data_loader import Keyword
from models import *
from google import genai
from time import sleep


class LLMProvider:
    def generate(self, prompt: str) -> str:
        raise NotImplementedError


class GeminiProvider(LLMProvider):
    def __init__(self, api_key: str, model_name: str = "gemini-3-flash", temperature: float = 0.7,
                 top_p: float = 1.,
                 thinking_level: str = "HIGH"):
        self.client = genai.Client(api_key=api_key)
        config_kwargs = {
            "temperature": temperature,
            "top_p": top_p,
            "thinking_config": {"thinking_level": thinking_level.upper()}
        }
        self.generation_config = genai.types.GenerateContentConfig(**config_kwargs)
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        try:
            response = self.client.models.generate_content(model=self.model_name,
                                                           contents=prompt,
                                                           config=self.generation_config,
                                                           )
            sleep(0.15)
            return response.text
        except Exception as e:
            print(f"Gemini Error: {e}")
            return ""


class OpenAIProvider(LLMProvider):
    def __init__(self,
                 api_key: str,
                 model_name: str,
                 temperature: float = None,
                 top_p: float = None,
                 reasoning_effort: str = "high"):
        self.thinking_level = None
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p

        # reasoning_effort can be "low", "medium", or "high"
        self.reasoning_effort = reasoning_effort

    def generate(self, prompt: str) -> str:
        try:
            # Build base arguments
            kwargs = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
            }
            if self.reasoning_effort is not None:
                kwargs["reasoning_effort"] = self.reasoning_effort.lower()
            # if temperature not null - add
            if self.temperature is not None:
                kwargs["temperature"] = self.temperature
            # if top_p not null - add
            if self.top_p is not None:
                kwargs["top_p"] = self.top_p

            response = self.client.chat.completions.create(**kwargs)
            sleep(0.1)
            return response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI Error: {e}")
            return ""


class QwenProvider(LLMProvider):
    def __init__(self, api_key: str, model_name: str = "Qwen/Qwen2.5-72B-Instruct-Turbo", temperature: float = 0.7,
                 top_p: float = 0.7):
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


def format_keywords(keywords: List[Keyword]) -> str:
    """
    Organize by hierarchy for display
    A simple indented list or path-based list
    optimizing for token usage and clarity
    Group by level 0
    :param keywords:
    :return:
    """
    # Let's map parent IDs to children
    tree = {}  # parent_id -> list of keywords
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


def format_keywords_by_category(keywords: List[Keyword]) -> str:
    """
    Create a string of all keywords and their IDs organized by categories.
    Categories are level 0 keywords, keywords are level 1.
    Format:
        Category {category_name}, id: {category_id}
          - {keyword_name} (id: {keyword_id})
    """
    # Separate categories (level 0) and keywords (level 1)
    categories = [kw for kw in keywords if kw.level == 0]
    # Build mapping: parent_id -> list of level-1 children
    children_map: Dict[int, List[Keyword]] = {}
    for kw in keywords:
        if kw.level == 1 and kw.parent_id is not None:
            children_map.setdefault(kw.parent_id, []).append(kw)

    output = []
    for cat in categories:
        output.append(f"Category {cat.name}, id: {cat.id}")
        children = children_map.get(cat.id, [])
        for child in children:
            output.append(f"  - {child.name} (id: {child.id})")
        output.append("")  # blank line between categories

    return "\n".join(output).rstrip()


class Classifier:
    def __init__(
            self,
            provider: str,
            api_key: str,
            prompt_path: str = "prompts/default.py",
            prompt_name: str = "CLASSIFICATION_PROMPT",
            model_name: str = None,
            temperature: float = 0.0,
            top_p: float = 1.,
            thinking_level: str = None,
            debug: bool = False
    ):
        self.provider_name = provider.lower()
        self.debug = debug
        self.prompt_name = prompt_name

        if self.provider_name == 'gemini':
            self.llm = GeminiProvider(api_key, model_name, temperature, top_p, thinking_level)
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
                    "classification_prompt": getattr(prompts_module, self.prompt_name, ""),
                    "suggestion_prompt": getattr(prompts_module, "SUGGESTION_PROMPT", "")
                }
            else:
                raise FileNotFoundError(f"Could not load prompts from {prompt_path}")

        except Exception as e:
            print(f"Error loading prompts from {prompt_path}: {e}")
            raise e

    def classify(self, text: str, keywords: List[Keyword], metadata: Dict[str, str] = {}) -> Tuple[
        List[str], List[str], str]:
        """
        Returns: (matched_keyword_ids_or_names, new_suggested_keywords, raw_response)
        """
        if self.prompt_name == "MATCH_KEYWORDS":
            return self._classify_match_keywords(text, keywords, metadata)
        else:
            return self._classify_default(text, keywords, metadata)

    def _classify_match_keywords(self, text: str, keywords: List[Keyword], metadata: Dict[str, str] = {}) -> Tuple[
        List[str], List[str], str]:
        """
        Classification using the MATCH_KEYWORDS prompt.
        Single-step: classification + suggestions in one LLM call.
        Output validated with Pydantic.
        """
        hierarchy = format_keywords_by_category(keywords)

        prompt = self.prompts.get("classification_prompt", "").format(
            hierarchy=hierarchy,
            text=text,
            translation=metadata.get('translation', ''),
            Language=metadata.get('language', 'Hebrew')
        )

        if self.debug:
            print(f"\n[DEBUG] --- MATCH_KEYWORDS Prompt ---\n{prompt}\n-----------------------------------")

        response = self.llm.generate(prompt)

        if self.debug:
            print(f"\n[DEBUG] --- LLM Response ---\n{response}\n------------------------------")

        # Parse JSON response and validate with Pydantic
        matched_ids = []
        suggested_kws = []
        try:
            # Strip potential markdown code fences
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3]
                cleaned = cleaned.strip()
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()

            raw_json = json.loads(cleaned)
            entries = validate_match_keywords_response(raw_json)

            for entry in entries:
                if entry.suggested:
                    suggested_kws.append(entry.keyword)
                else:
                    matched_ids.append(str(entry.keyword_id))

        except json.JSONDecodeError as e:
            print(f"Error parsing JSON from MATCH_KEYWORDS response: {e}")
        except Exception as e:
            print(f"Error validating MATCH_KEYWORDS response: {e}")

        return matched_ids, suggested_kws, response

    def _classify_default(self, text: str, keywords: List[Keyword], metadata: Dict[str, str] = {}) -> Tuple[
        List[str], List[str], str]:
        """
        Original classification flow: step 1 (classify) + step 2 (suggest).
        """
        hierarchy_str = format_keywords(keywords)

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

        matched_ids = re.findall(r'\((\d+)\s*,', response_1)

        if not matched_ids:
            matched_ids = [s.strip() for s in response_1.split(',') if s.strip().isdigit()]

        # Prepare subset of keywords for Step 2
        matched_kws_sub = [k for k in keywords if str(k.id) in matched_ids]
        matched_str = ", ".join([k.name for k in matched_kws_sub])

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
            try:
                new_keywords = ast.literal_eval(response_2)
            except Exception as e:
                print(f"Error parsing suggestion response: {e}")
                new_keywords = []

        return matched_ids, new_keywords, response_1

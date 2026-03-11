import hashlib
import json
import time

from diskcache import Cache

from mimic.config import TeacherConfig
from mimic.generator.client.interface import ClientInterface


class Client(ClientInterface):
    def __init__(self, teacher_config: TeacherConfig):
        match teacher_config.provider:
            case "openai":
                from mimic.generator.client.openai import OpenAI

                self.client = OpenAI(teacher_config)
            case _:
                raise ValueError(
                    f"Unsupported teacher provider: {teacher_config.provider}"
                )
        self.cache = Cache("mimic_generate_cache")
        self.cache_key_prefix = f"{teacher_config.provider}:{teacher_config.model}:"
        params_str = json.dumps(
            teacher_config.generation_params, sort_keys=True, ensure_ascii=True
        )
        params_hash = hashlib.md5(params_str.encode()).hexdigest()[:16]
        self.cache_key_prefix += f"{params_hash}:"

    def generate_chat_response(
        self, messages: list[dict[str, str]], teacher_config: TeacherConfig
    ) -> str:
        cache_key = f"{self.cache_key_prefix}chat:{hashlib.md5(str(messages).encode()).hexdigest()[:16]}"
        if cache_key in self.cache:
            return self.cache[cache_key]  # pyright: ignore[reportReturnType]

        max_retries = teacher_config.request_config.max_retries
        error = None
        for attempt in range(max_retries + 1):
            try:
                response = self.client.generate_chat_response(messages, teacher_config)
                self.cache[cache_key] = response  # pyright: ignore[reportAssignmentType]
                return response
            except Exception as e:
                error = e
                # Exponential backoff
                wait_time = 2**attempt
                time.sleep(wait_time)
        raise Exception(f"Failed after {max_retries} retries: {error}")

    def generate_text_response(self, prompt: str, teacher_config: TeacherConfig) -> str:
        cache_key = f"{self.cache_key_prefix}text:{hashlib.md5(prompt.encode()).hexdigest()[:16]}"
        if cache_key in self.cache:
            return self.cache[cache_key]  # pyright: ignore[reportReturnType]
        max_retries = teacher_config.request_config.max_retries
        error = None

        for attempt in range(max_retries + 1):
            try:
                response = self.client.generate_text_response(prompt, teacher_config)
                self.cache[cache_key] = response  # pyright: ignore[reportAssignmentType]
                return response
            except Exception as e:
                error = e
                # Exponential backoff
                wait_time = 2**attempt
                time.sleep(wait_time)
        raise Exception(f"Failed after {max_retries} retries: {error}")

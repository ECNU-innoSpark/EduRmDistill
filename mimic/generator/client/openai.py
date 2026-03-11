from openai import Client

from mimic.config import TeacherConfig
from mimic.generator.client.interface import ClientInterface


class OpenAI(ClientInterface):
    def __init__(self, teacher_config: TeacherConfig):
        self.client = Client(
            api_key=teacher_config.get_api_key(),
            base_url=teacher_config.base_url,
            timeout=teacher_config.request_config.timeout,
        )

    def generate_chat_response(
        self, messages: list[dict[str, str]], teacher_config: TeacherConfig
    ) -> str:
        response = self.client.chat.completions.create(
            model=teacher_config.model,
            messages=messages,  # type: ignore
            stream=False,  # 目前不支持流式输出
            **teacher_config.generation_params,
        )  # type: ignore
        return response.choices[0].message.content

    def generate_text_response(self, prompt: str, teacher_config: TeacherConfig) -> str:
        response = self.client.completions.create(
            model=teacher_config.model,
            prompt=prompt,
            stream=False,  # 目前不支持流式输出
            **teacher_config.generation_params,
        )
        return response.choices[0].text.strip()

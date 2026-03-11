from abc import abstractmethod
from mimic.config import TeacherConfig


class ClientInterface:
    @abstractmethod
    def generate_chat_response(
        self, messages: list[dict[str, str]], teacher_config: TeacherConfig
    ) -> str:
        """根据输入的消息列表生成回复"""
        pass

    @abstractmethod
    def generate_text_response(self, prompt: str, teacher_config: TeacherConfig) -> str:
        """根据输入的文本提示生成回复"""
        pass

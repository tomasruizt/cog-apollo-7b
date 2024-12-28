import json
from typing import Literal
from pydantic import BaseModel


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str

    @classmethod
    def parse_messages(cls, messages: str):
        return [cls(**msg) for msg in json.loads(messages)]


if __name__ == "__main__":
    conversation = '[{"role": "user", "content": "Hello, how are you?"}, {"role": "assistant", "content": "I\'m fine, thank you!"}, {"role": "user", "content": "What is your name?"}, {"role": "assistant", "content": "My name is Apollo."}]'
    print(Message.parse_messages(conversation))

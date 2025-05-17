from typing import TypedDict


# nice types
class Message(TypedDict):
    role: str
    content: str


MessageList = list[Message]
MessageListBatch = list[list[Message]]

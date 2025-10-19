class Message:
    def __init__(self, id: int, content: str, author: str, vector: list = None):
        self.id = id
        self.content = content
        self.author = author
        self.vector = vector

    def to_dict(self):
        return {
            "id": self.id,
            "content": self.content,
            "author": self.author,
            "vector": self.vector
        }
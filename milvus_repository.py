from pymilvus import MilvusClient

class MilvusRepository:
    def __init__(self, collection_name: str, dimensions_count: int):
        self.client = MilvusClient("milvus_demo.db")
        self.collection_name = collection_name
        self.dimensions_count = dimensions_count
        self.guarantee_collection_existence()

    def guarantee_collection_existence(self):
        if self.client.has_collection(collection_name=self.collection_name):
            self.client.drop_collection(collection_name=self.collection_name)
        self.client.create_collection(collection_name=self.collection_name, dimension=self.dimensions_count)

    def insert_data(self, data: list):
        self.client.insert(self.collection_name, data)

    def search(self, query_vectors: list, limit: int = 1):
        return self.client.search(
            collection_name=self.collection_name,
            data=query_vectors,
            limit=limit,
            output_fields=["author", "text"]
        )
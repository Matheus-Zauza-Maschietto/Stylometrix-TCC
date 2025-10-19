from milvus_repository import MilvusRepository
import stylo_metrix as sm

class PredictionService:
    def __init__(self, milvus_repo: MilvusRepository, stylo: sm.StyloMetrix):
        self.milvus_repo = milvus_repo
        self.stylo = stylo

    def predict_author(self, vector: list, limit: int = 1) -> str:
        search_results = self.milvus_repo.search(query_vectors=[vector], limit=limit)
        return search_results[0][0]["entity"]["author"] if search_results else None
    
    def is_correct_prediction(self, predicted_author: str, actual_author: str) -> bool:
        return predicted_author == actual_author
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
    
    def evaluate_predictions(self, testing_vectors: list, testing_messages: list) -> dict:
        correct_predictions = 0
        total_predictions = len(testing_vectors)
        human1_correct = 0
        human1_incorrect = 0
        human2_correct = 0
        human2_incorrect = 0

        for i, test_vector in enumerate(testing_vectors):
            predicted_author = self.predict_author(test_vector)

            if predicted_author is None:
                continue

            actual_author = testing_messages[i]['nomePessoa']
            
            if predicted_author == actual_author:
                correct_predictions += 1
                if actual_author == 'Human 1':
                    human1_correct += 1
                elif actual_author == 'Human 2':
                    human2_correct += 1
            else:
                if actual_author == 'Human 1':
                    human1_incorrect += 1
                elif actual_author == 'Human 2':
                    human2_incorrect += 1
        
        accuracy = (correct_predictions / total_predictions) * 100
        incorrect_predictions = total_predictions - correct_predictions

        human1_actual = sum(1 for msg in testing_messages if msg['nomePessoa'] == 'Human 1')
        human2_actual = sum(1 for msg in testing_messages if msg['nomePessoa'] == 'Human 2')
        
        return {
            'correct_predictions': correct_predictions,
            'total_predictions': total_predictions,
            'incorrect_predictions': incorrect_predictions,
            'accuracy': accuracy,
            'human1_correct': human1_correct,
            'human1_incorrect': human1_incorrect,
            'human2_correct': human2_correct,
            'human2_incorrect': human2_incorrect,
            'human1_actual': human1_actual,
            'human2_actual': human2_actual
        }
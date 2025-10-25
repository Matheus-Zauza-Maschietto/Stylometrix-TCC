import stylo_metrix as sm
from chat_reader import read_human_chat
from milvus_repository import MilvusRepository
from pandas_service import PandasService
from numpy_service import NumpyService
from message import Message
from prediction_service import PredictionService
from visualization_service import VisualizationService

chat_messages = read_human_chat('/home/matheus/github/Clustering/StyloMetrix/Datasets/human_chat.txt')

stylo = sm.StyloMetrix('en')

milvus_repo = MilvusRepository(collection_name="demo_collection", dimensions_count=197)
prediction_service = PredictionService(milvus_repo=milvus_repo, stylo=stylo)

texts = [msg['texto'] for msg in chat_messages]

training_size = int(0.7 * len(texts))
testing_size = len(texts) - training_size

training_texts = texts[:training_size]
training_messages = chat_messages[:training_size]

training_metrics = stylo.transform(training_texts)

PandasService.clean_non_numeric_metrics(training_metrics)

data = []
for i in range(len(training_metrics)):
    data.append(Message(
        id=i,
        content=training_texts[i],
        author=training_messages[i]['nomePessoa'],
        vector=NumpyService.to_float64_list(training_metrics.iloc[i])
    ))

milvus_repo.insert_data([msg.to_dict() for msg in data])

testing_texts = texts[training_size:]
testing_messages = chat_messages[training_size:]

testing_metrics = stylo.transform(testing_texts)

PandasService.clean_non_numeric_metrics(testing_metrics)

testing_vectors = []
for i in range(len(testing_metrics)):
    vector = NumpyService.to_float64_list(testing_metrics.iloc[i])
    testing_vectors.append(vector)

results = prediction_service.evaluate_predictions(testing_vectors, testing_messages)

print(f"\nGerando gráficos de visualização...")
VisualizationService.create_accuracy_bar_chart(
    correct_predictions=results['correct_predictions'],
    incorrect_predictions=results['incorrect_predictions'],
    output_path='./accuracy_chart.png'
)

VisualizationService.create_detailed_bar_chart(
    correct_predictions=results['correct_predictions'],
    incorrect_predictions=results['incorrect_predictions'],
    human1_correct=results['human1_correct'],
    human1_incorrect=results['human1_incorrect'],
    human2_correct=results['human2_correct'],
    human2_incorrect=results['human2_incorrect'],
    output_path='./detailed_accuracy_chart.png'
)

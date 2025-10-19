import stylo_metrix as sm
from chat_reader import read_human_chat
from milvus_repository import MilvusRepository
from pandas_service import PandasService
from numpy_service import NumpyService
from message import Message
from prediction_service import PredictionService

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

correct_predictions = 0
total_predictions = len(testing_vectors)

for i, test_vector in enumerate(testing_vectors):
    predicted_author = prediction_service.predict_author(test_vector)

    if predicted_author is None:
        continue

    actual_author = testing_messages[i]['nomePessoa']
    
    if predicted_author == actual_author:
        correct_predictions += 1
    
accuracy = (correct_predictions / total_predictions) * 100

human1_actual = sum(1 for msg in testing_messages if msg['nomePessoa'] == 'Human 1')
human2_actual = sum(1 for msg in testing_messages if msg['nomePessoa'] == 'Human 2')

print(f"\n" + "="*60)
print(f"AUTHOR IDENTIFICATION RESULTS")
print(f"="*60)
print(f"Dataset Distribution:")
print(f"  Training samples: {training_size} (70%)")
print(f"  Testing samples: {total_predictions} (30%)")
print(f"  - Human 1: {human1_actual} messages")
print(f"  - Human 2: {human2_actual} messages")
print(f"\nResults:")
print(f"  Correct predictions: {correct_predictions}")
print(f"  Incorrect predictions: {total_predictions - correct_predictions}")
print(f"  Overall Accuracy: {accuracy:.2f}%")
print(f"\nVector Details:")
print(f"  StyloMetrix features: {len(training_metrics.columns)} dimensions")
print(f"  Precision: 64-bit floating point (np.float64)")
print(f"="*60)

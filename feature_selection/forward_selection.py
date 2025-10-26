import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../base_implementation'))

import stylo_metrix as sm
from chat_reader import read_human_chat
from repositories.milvus_repository import MilvusRepository
from services.pandas_service import PandasService
from services.numpy_service import NumpyService
from models.message import Message
from services.prediction_service import PredictionService
import pandas as pd
import json
import traceback
from datetime import datetime

INITIAL_SELECTED_METRICS = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
]


class ForwardSelection:
    def __init__(self, chat_file_path: str, max_features: int = 60, checkpoint_file: str = 'forward_selection_checkpoint.json'):
        self.chat_file_path = chat_file_path
        self.max_features = max_features
        self.checkpoint_file = checkpoint_file
        self.stylo = sm.StyloMetrix('en')
        
        self.chat_messages = read_human_chat(chat_file_path)
        self.texts = [msg['texto'] for msg in self.chat_messages]
        
        training_size = int(0.7 * len(self.texts))
        self.training_texts = self.texts[:training_size]
        self.training_messages = self.chat_messages[:training_size]
        self.testing_texts = self.texts[training_size:]
        self.testing_messages = self.chat_messages[training_size:]
        
        print("Extracting training metrics...")
        self.training_metrics_full = self.stylo.transform(self.training_texts)
        PandasService.clean_non_numeric_metrics(self.training_metrics_full)
        
        print("Extracting testing metrics...")
        self.testing_metrics_full = self.stylo.transform(self.testing_texts)
        PandasService.clean_non_numeric_metrics(self.testing_metrics_full)
        
        self.total_features = self.training_metrics_full.shape[1]
        print(f"Total features available: {self.total_features}")
        
        self.selected_features = []
        self.available_features = list(range(self.total_features))
        self.results_history = []
        
        # Try to load checkpoint
        self._load_checkpoint()
        
    def _load_checkpoint(self):
        """Load checkpoint if it exists and continue from there."""
        if os.path.exists(self.checkpoint_file):
            try:
                print(f"\n{'='*80}")
                print(f"Checkpoint file found: {self.checkpoint_file}")
                print(f"{'='*80}")
                
                with open(self.checkpoint_file, 'r') as f:
                    checkpoint = json.load(f)
                
                # Validate checkpoint matches current configuration
                if (checkpoint['parameters']['total_features_available'] != self.total_features or
                    checkpoint['parameters']['training_size'] != len(self.training_texts) or
                    checkpoint['parameters']['testing_size'] != len(self.testing_texts)):
                    print("⚠ Warning: Checkpoint parameters don't match current data!")
                    print("  Starting fresh instead of loading checkpoint.")
                    return
                
                # Load state
                self.selected_features = checkpoint['selected_features']
                self.results_history = checkpoint['results_history']
                
                # Rebuild available features
                self.available_features = [f for f in range(self.total_features) 
                                          if f not in self.selected_features]
                
                print(f"  Checkpoint loaded successfully!")
                print(f"  Iterations completed: {len(self.results_history)}")
                print(f"  Features selected: {len(self.selected_features)}")
                print(f"  Current accuracy: {self.results_history[-1]['accuracy']:.2f}%" if self.results_history else "N/A")
                print(f"  Remaining features: {len(self.available_features)}")
                print(f"{'='*80}\n")
                
            except Exception as e:
                print(f"⚠ Error loading checkpoint: {e}")
                print("  Starting fresh instead.")
                self.selected_features = []
                self.available_features = list(range(self.total_features))
                self.results_history = []
        else:
            print(f"No checkpoint file found. Starting fresh.")
    
    def _save_checkpoint(self):
        """Save current state to checkpoint file."""
        checkpoint = {
            'timestamp': datetime.now().isoformat(),
            'parameters': {
                'max_features': self.max_features,
                'training_size': len(self.training_texts),
                'testing_size': len(self.testing_texts),
                'total_features_available': self.total_features
            },
            'selected_features': self.selected_features,
            'final_accuracy': self.results_history[-1]['accuracy'] if self.results_history else 0,
            'results_history': self.results_history
        }
        
        try:
            with open(self.checkpoint_file, 'w') as f:
                json.dump(checkpoint, f, indent=2)
            print(f"  💾 Checkpoint saved: {self.checkpoint_file}")
        except Exception as e:
            print(f"  ⚠ Error saving checkpoint: {e}")
    
    def evaluate_feature_set(self, feature_indices: list) -> float:
        """Evaluate a specific set of features and return accuracy."""
        print(f"  [Evaluate] Starting evaluation of {len(feature_indices)} features...")
        
        if not feature_indices:
            print(f"  [Evaluate] No features provided, returning 0.0")
            return 0.0
        
        if len(feature_indices) < 2:
            print(f"  [Evaluate] Only {len(feature_indices)} feature(s), need at least 2, returning 0.0")
            return 0.0
        
        print(f"  [Evaluate] Extracting training and testing metrics...")
        training_metrics = self.training_metrics_full.iloc[:, feature_indices]
        testing_metrics = self.testing_metrics_full.iloc[:, feature_indices]
        
        print(f"  [Evaluate] Creating Milvus repository...")
        milvus_repo = MilvusRepository(
            collection_name=f"forward_selection_{len(feature_indices)}", 
            dimensions_count=len(feature_indices)
        )
        
        print(f"  [Evaluate] Initializing prediction service...")
        prediction_service = PredictionService(milvus_repo=milvus_repo, stylo=self.stylo)
        
        print(f"  [Evaluate] Preparing training data ({len(training_metrics)} samples)...")
        data = []
        for i in range(len(training_metrics)):
            data.append(Message(
                id=i,
                content=self.training_texts[i],
                author=self.training_messages[i]['nomePessoa'],
                vector=NumpyService.to_float64_list(training_metrics.iloc[i])
            ))
        
        print(f"  [Evaluate] Inserting training data into Milvus...")
        milvus_repo.insert_data([msg.to_dict() for msg in data])
        
        print(f"  [Evaluate] Preparing testing vectors ({len(testing_metrics)} samples)...")
        testing_vectors = []
        for i in range(len(testing_metrics)):
            vector = NumpyService.to_float64_list(testing_metrics.iloc[i])
            testing_vectors.append(vector)
        
        print(f"  [Evaluate] Running predictions on test set...")
        results = prediction_service.evaluate_predictions(testing_vectors, self.testing_messages)
        
        accuracy = results['accuracy']
        print(f"  [Evaluate] Completed! Accuracy: {accuracy:.2f}%")
        
        return accuracy
    
    def run(self):
        print("\n" + "="*80)
        print("Starting Forward Selection Algorithm")
        print("="*80)
        print(f"Maximum features: {self.max_features}")
        print(f"Training samples: {len(self.training_texts)}")
        print(f"Testing samples: {len(self.testing_texts)}")
        print(f"Starting with {len(INITIAL_SELECTED_METRICS)} features from init.py")
        print("="*80 + "\n")
        
        if len(self.selected_features) == 0:
            print("Initializing with features from init.py...")
            
            initial_features = [f for f in INITIAL_SELECTED_METRICS if f < self.total_features]
            print(f"  Validating initial features: {initial_features}")
            
            try:
                print("  Evaluating initial feature set...")
                initial_accuracy = self.evaluate_feature_set(initial_features)
                
                self.selected_features = initial_features
                for feat in initial_features:
                    self.available_features.remove(feat)
                
                print(f"✓ Starting features: {initial_features}")
                print(f"  Initial accuracy: {initial_accuracy:.2f}%")
                print(f"  Remaining available features: {len(self.available_features)}\n")
                
                self.results_history.append({
                    'iteration': 0,
                    'feature_added': initial_features,
                    'selected_features': self.selected_features.copy(),
                    'accuracy': initial_accuracy,
                    'features_count': len(self.selected_features)
                })
                
                best_accuracy = initial_accuracy
                
                # Save initial checkpoint
                self._save_checkpoint()
                
            except Exception as e:
                import traceback
                print(f"\n✗ Error initializing with features {initial_features}:")
                print(f"  Error: {e}")
                print(f"  Traceback:\n{traceback.format_exc()}")
                return {
                    'selected_features': [],
                    'final_accuracy': 0.0,
                    'results_history': [],
                    'total_iterations': 0
                }
        else:
            best_accuracy = self.results_history[-1]['accuracy'] if self.results_history else 0.0
                
        while len(self.selected_features) < self.max_features:
            iteration = len(self.results_history)
            print(f"\n{'='*80}")
            print(f"Iteration {iteration + 1}")
            print(f"Current features: {len(self.selected_features)}")
            print(f"Available features: {len(self.available_features)}")
            print(f"Best accuracy so far: {best_accuracy:.2f}%")
            print(f"{'='*80}")
            
            best_feature = None
            best_feature_accuracy = best_accuracy
            
            print(f"\n[Main] Testing {len(self.available_features)} features sequentially...")
            print(f"[Main] Features to test: {self.available_features[:10]}{'...' if len(self.available_features) > 10 else ''}\n")
            
            for feature_idx in self.available_features:
                print(f"Evaluating feature {feature_idx}...")
                
                candidate_features = self.selected_features + [feature_idx]
                
                if len(candidate_features) < 2:
                    print(f"  Feature {feature_idx}: Skipping - not enough features ({len(candidate_features)} < 2)")
                    continue
                
                try:
                    training_metrics = self.training_metrics_full.iloc[:, candidate_features]
                    testing_metrics = self.testing_metrics_full.iloc[:, candidate_features]
                    
                    milvus_repo = MilvusRepository(
                        collection_name=f"forward_selection_{len(candidate_features)}_{feature_idx}", 
                        dimensions_count=len(candidate_features)
                    )
                    
                    prediction_service = PredictionService(milvus_repo=milvus_repo, stylo=self.stylo)
                    
                    data = []
                    for i in range(len(training_metrics)):
                        data.append(Message(
                            id=i,
                            content=self.training_texts[i],
                            author=self.training_messages[i]['nomePessoa'],
                            vector=NumpyService.to_float64_list(training_metrics.iloc[i])
                        ))
                    
                    milvus_repo.insert_data([msg.to_dict() for msg in data])
                    
                    testing_vectors = []
                    for i in range(len(testing_metrics)):
                        vector = NumpyService.to_float64_list(testing_metrics.iloc[i])
                        testing_vectors.append(vector)
                    
                    results = prediction_service.evaluate_predictions(testing_vectors, self.testing_messages)
                    
                    accuracy = results['accuracy']
                    
                    if accuracy > best_feature_accuracy:
                        best_feature_accuracy = accuracy
                        best_feature = feature_idx
                        print(f"  Feature {feature_idx}: ✓ {accuracy:.2f}% (NEW BEST!)")
                    else:
                        print(f"  Feature {feature_idx}: {accuracy:.2f}%")
                        
                except Exception as e:
                    print(f"  Feature {feature_idx}: ✗ Error - {str(e)}")
            
            print(f"\n[Main] Completed testing all features")
            
            if best_feature is not None and best_feature_accuracy > best_accuracy:
                self.selected_features.append(best_feature)
                self.available_features.remove(best_feature)
                best_accuracy = best_feature_accuracy
                no_improvement_count = 0
                
                result = {
                    'iteration': iteration + 1,
                    'feature_added': best_feature,
                    'selected_features': self.selected_features.copy(),
                    'accuracy': best_accuracy,
                    'features_count': len(self.selected_features)
                }
                self.results_history.append(result)
                
                improvement = best_accuracy - (self.results_history[-2]['accuracy'] if len(self.results_history) > 1 else 0)
                print(f"\n✓ Added feature {best_feature}")
                print(f"  New accuracy: {best_accuracy:.2f}% (+{improvement:.2f}%)")
                print(f"  Total features: {len(self.selected_features)}")
                print(f"  Remaining candidates: {len(self.available_features)}")
                
                # Save checkpoint after each successful iteration
                self._save_checkpoint()
                
            else:
                print(f"\n✗ No improvement found in this iteration")
                print(f"  Best feature tested: {best_feature if best_feature else 'None'}")
                print(f"  Best accuracy achieved: {best_feature_accuracy:.2f}%")
                print(f"  Current best: {best_accuracy:.2f}%")
                
                # Save checkpoint even when no improvement
                self._save_checkpoint()
                
        print(f"\n{'='*80}")
        print("Forward Selection Complete!")
        print(f"{'='*80}")
        print(f"Total iterations: {len(self.results_history)}")
        print(f"Total features selected: {len(self.selected_features)}")
        print(f"Final accuracy: {best_accuracy:.2f}%")
        print(f"Selected features: {self.selected_features}")
        print(f"{'='*80}\n")
        
        return {
            'selected_features': self.selected_features,
            'final_accuracy': best_accuracy,
            'results_history': self.results_history,
            'total_iterations': len(self.results_history)
        }
    
    def save_results(self, output_file: str = 'forward_selection_results.json'):
        results = {
            'timestamp': datetime.now().isoformat(),
            'parameters': {
                'max_features': self.max_features,
                'training_size': len(self.training_texts),
                'testing_size': len(self.testing_texts),
                'total_features_available': self.total_features
            },
            'selected_features': self.selected_features,
            'final_accuracy': self.results_history[-1]['accuracy'] if self.results_history else 0,
            'results_history': self.results_history
        }
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    selector = ForwardSelection(
        chat_file_path='/home/matheus/github/Clustering/StyloMetrix/Datasets/human_chat.txt',
        max_features=60,
        checkpoint_file='forward_selection_checkpoint.json'
    )
    
    results = selector.run()
    selector.save_results('forward_selection_results.json')

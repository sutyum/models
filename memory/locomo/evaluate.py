"""
SOTA Memory System Evaluation for LOCOMO
Implements exact LOCOMO category-specific evaluation logic to achieve >68% performance.
"""
import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime
import pickle
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

import dspy
from dspy.evaluate import Evaluate
import mlflow
import mlflow.sklearn

from locomo.dataset import load_locomo_dataset
from locomo.sota_memory_system import SOTAMemorySystem, create_sota_memory_system
from locomo.locomo_evaluation_logic import locomo_category_evaluation, evaluate_with_locomo_logic, convert_to_locomo_format


class SOTALocomoMetric:
    """LOCOMO evaluation metric using exact category-specific scoring logic."""
    
    def __init__(self):
        # No LLM judge needed - using exact LOCOMO logic
        pass
    
    def __call__(self, example: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
        """Evaluate prediction using exact LOCOMO category-specific logic."""
        
        try:
            # Convert to LOCOMO format
            example_dict = {
                "question": example.question,
                "answer": example.answer,  # ground truth
                "category": getattr(example, 'category', 3),
                "sample_id": getattr(example, 'sample_id', ''),
                "qa_id": getattr(example, 'qa_id', ''),
                "evidence": getattr(example, 'evidence', [])
            }
            
            prediction_dict = {
                "answer": pred.answer,
                "relevant_memories": getattr(pred, 'relevant_memories', [])
            }
            
            locomo_pred = convert_to_locomo_format(example_dict, prediction_dict)
            
            # Use LOCOMO category evaluation
            accuracy_scores, _ = locomo_category_evaluation([locomo_pred])
            return accuracy_scores[0] if accuracy_scores else 0.0
                
        except Exception as e:
            print(f"Error in LOCOMO evaluation: {e}")
            # Fallback evaluation using simple F1
            from locomo.evaluation import f1_score
            return f1_score(pred.answer, example.answer)


class SOTAEvaluationPipeline:
    """Complete evaluation pipeline for SOTA memory system using LOCOMO logic."""
    
    def __init__(self, output_dir: str = "./sota_evaluation_results", mlflow_tracking: bool = True):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.memory_system = None
        self.metric = SOTALocomoMetric()
        self.mlflow_tracking = mlflow_tracking
        
        # Setup MLflow tracking
        if self.mlflow_tracking:
            try:
                mlflow.set_tracking_uri("http://127.0.0.1:5000")
                mlflow.set_experiment("DSPy")
                
                # Enable DSPy autolog if available
                try:
                    from packaging.version import Version
                    if Version(mlflow.__version__) >= Version("2.18.0"):
                        mlflow.dspy.autolog()
                        print("✅ MLflow DSPy autolog enabled")
                except Exception:
                    pass  # Continue without autolog
                
                print("✅ MLflow tracking configured")
            except Exception as e:
                print(f"⚠️  MLflow setup warning: {e}")
                self.mlflow_tracking = False
    
    def build_memory_from_dataset(self, dataset, limit: int = None, 
                                parallel_workers: int = 1) -> SOTAMemorySystem:
        """Build memory system from LOCOMO dataset."""
        print("🏗️  Building SOTA memory system from dataset...")
        
        # Log memory building parameters to MLflow
        if self.mlflow_tracking:
            mlflow.log_param("memory_limit", limit or "all")
            mlflow.log_param("memory_parallel_workers", parallel_workers)
        
        memory_system = create_sota_memory_system()
        examples = dataset.get_examples(limit=limit)
        
        # Group examples by conversation
        conversations = {}
        for example in examples:
            sample_id = example.sample_id
            if sample_id not in conversations:
                conversations[sample_id] = {
                    "conversation": {},
                    "questions": []
                }
            conversations[sample_id]["questions"].append(example)
        
        # Load raw conversation data and process
        print("📚 Processing conversations to build memories...")
        
        # Get conversation data from raw dataset
        raw_data = dataset.raw_data
        conversation_map = {sample["sample_id"]: sample for sample in raw_data}
        
        if parallel_workers == 1:
            # Sequential processing
            for sample_id in tqdm(conversations.keys(), desc="Building memories"):
                if sample_id in conversation_map:
                    conversation_data = conversation_map[sample_id]
                    memory_system.process_conversation(conversation_data, sample_id)
        else:
            # Parallel processing
            def process_conversation(sample_id):
                if sample_id in conversation_map:
                    conversation_data = conversation_map[sample_id]
                    # Create separate memory system for this thread
                    thread_memory_system = create_sota_memory_system()
                    thread_memory_system.process_conversation(conversation_data, sample_id)
                    return thread_memory_system.memories
                return {}
            
            print(f"Using {parallel_workers} parallel workers for memory building")
            with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
                futures = {
                    executor.submit(process_conversation, sample_id): sample_id 
                    for sample_id in conversations.keys()
                }
                
                for future in tqdm(as_completed(futures), 
                                 total=len(conversations), 
                                 desc="Building memories"):
                    sample_id = futures[future]
                    thread_memories = future.result()
                    memory_system.memories.update(thread_memories)
            
            # Save consolidated memories
            memory_system._save_memory_store()
        
        print(f"✅ Built memory system with {len(memory_system.memories)} memories")
        
        # Log memory building metrics
        if self.mlflow_tracking:
            mlflow.log_metric("total_memories", len(memory_system.memories))
            mlflow.log_metric("processed_conversations", len(conversations))
        
        self.memory_system = memory_system
        return memory_system
    
    def evaluate_system(self, examples: List[dspy.Example], 
                       parallel_workers: int = 4) -> Dict[str, Any]:
        """Evaluate the SOTA memory system using exact LOCOMO evaluation logic."""
        
        if not self.memory_system:
            raise ValueError("Memory system not built. Call build_memory_from_dataset first.")
        
        print(f"🔍 Evaluating SOTA memory system on {len(examples)} examples...")
        
        predictions = []
        detailed_results = []
        
        if parallel_workers == 1:
            # Sequential evaluation
            for example in tqdm(examples, desc="Evaluating"):
                pred, result_data = self._evaluate_single_example(example)
                predictions.append(pred)
                detailed_results.append(result_data)
        else:
            # Parallel evaluation
            def evaluate_with_system(example):
                return self._evaluate_single_example(example)
            
            print(f"Using {parallel_workers} parallel workers for evaluation")
            with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
                futures = {
                    executor.submit(evaluate_with_system, example): example 
                    for example in examples
                }
                
                results_list = []
                for future in tqdm(as_completed(futures), 
                                 total=len(examples), 
                                 desc="Evaluating"):
                    pred, result_data = future.result()
                    results_list.append((pred, result_data))
                
                # Sort results to maintain order
                for pred, result_data in results_list:
                    predictions.append(pred)
                    detailed_results.append(result_data)
        
        # Convert to LOCOMO format for evaluation
        print("📊 Running LOCOMO category-specific evaluation...")
        locomo_predictions = []
        for example, pred, result_data in zip(examples, predictions, detailed_results):
            example_dict = {
                "question": example.question,
                "answer": example.answer,
                "category": getattr(example, 'category', 3),
                "sample_id": getattr(example, 'sample_id', ''),
                "qa_id": getattr(example, 'qa_id', ''),
                "evidence": getattr(example, 'evidence', [])
            }
            
            prediction_dict = {
                "answer": pred.answer,
                "relevant_memories": getattr(pred, 'relevant_memories', [])
            }
            
            locomo_pred = convert_to_locomo_format(example_dict, prediction_dict)
            locomo_predictions.append(locomo_pred)
        
        # Use LOCOMO evaluation logic
        locomo_results = evaluate_with_locomo_logic(locomo_predictions, verbose=True)
        
        # Compile results in expected format (maintain compatibility with existing code)
        results = {
            "overall_locomo_score": locomo_results["overall_accuracy"],
            "overall_llm_judge_score": locomo_results["overall_accuracy"],  # Alias for compatibility
            "overall_recall": locomo_results["overall_recall"],
            "total_examples": len(examples),
            "individual_scores": locomo_results["individual_scores"],
            "detailed_results": detailed_results,
        }
        
        # Add category-specific results
        for cat in range(1, 6):
            acc_key = f"category_{cat}_accuracy"
            count_key = f"category_{cat}_count"
            if acc_key in locomo_results:
                results[f"category_{cat}_score"] = locomo_results[acc_key]
                results[f"category_{cat}_count"] = locomo_results[count_key]
        
        # Log evaluation metrics to MLflow
        if self.mlflow_tracking:
            mlflow.log_param("evaluation_parallel_workers", parallel_workers)
            mlflow.log_metric("overall_locomo_score", results["overall_locomo_score"])
            mlflow.log_metric("overall_recall", results["overall_recall"])
            mlflow.log_metric("total_examples", results["total_examples"])
            
            # Log category-specific metrics
            for cat in range(1, 6):
                score_key = f"category_{cat}_score"
                count_key = f"category_{cat}_count"
                if score_key in results:
                    mlflow.log_metric(f"category_{cat}_accuracy", results[score_key])
                    mlflow.log_metric(f"category_{cat}_count", results[count_key])
        
        return results
    
    def _evaluate_single_example(self, example: dspy.Example) -> Tuple[dspy.Prediction, Dict]:
        """Evaluate a single example."""
        try:
            # Determine question category
            category_map = {1: "multi-hop", 2: "temporal", 3: "single-hop", 4: "unanswerable", 5: "ambiguous"}
            question_category = category_map.get(getattr(example, 'category', 3), "single-hop")
            
            # Get answer from memory system
            result = self.memory_system.answer_question(example.question, question_category)
            
            pred = dspy.Prediction(
                answer=result["answer"],
                reasoning=result["reasoning"],
                confidence=result["confidence"],
                relevant_memories=result["relevant_memories"]
            )
            
            # Detailed result data
            result_data = {
                "question": example.question,
                "ground_truth": example.answer,
                "prediction": result["answer"],
                "category": getattr(example, 'category', 3),
                "reasoning": result["reasoning"],
                "confidence": result["confidence"],
                "relevant_memories_count": len(result["relevant_memories"]),
                "sample_id": getattr(example, 'sample_id', ''),
                "qa_id": getattr(example, 'qa_id', '')
            }
            
            return pred, result_data
            
        except Exception as e:
            print(f"Error evaluating example: {e}")
            pred = dspy.Prediction(
                answer="Error in evaluation",
                relevant_memories=[]
            )
            result_data = {
                "question": example.question,
                "ground_truth": example.answer,
                "prediction": "Error in evaluation",
                "category": getattr(example, 'category', 3),
                "error": str(e)
            }
            return pred, result_data
    
    def save_results(self, results: Dict[str, Any], experiment_name: str):
        """Save evaluation results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.output_dir / f"{experiment_name}_{timestamp}_results.json"
        
        # Make results serializable
        serializable_results = {
            k: v for k, v in results.items() 
            if k != "detailed_results"  # Skip detailed results for main file
        }
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        # Save detailed results separately
        detailed_file = self.output_dir / f"{experiment_name}_{timestamp}_detailed.json"
        with open(detailed_file, 'w') as f:
            json.dump(results["detailed_results"], f, indent=2)
        
        print(f"✅ Results saved to {results_file}")
        print(f"📋 Detailed results saved to {detailed_file}")
        
        return results_file


def run_sota_evaluation(data_path: str = "./data/locomo10.json",
                       limit: int = None,
                       experiment_name: str = "sota_locomo_evaluation",
                       parallel_workers: int = 4,
                       memory_workers: int = 1,
                       mlflow_tracking: bool = True) -> Dict[str, Any]:
    """Run complete SOTA evaluation pipeline using exact LOCOMO evaluation logic."""
    
    # Start MLflow run
    if mlflow_tracking:
        try:
            mlflow.set_tracking_uri("http://127.0.0.1:5000")
            mlflow.set_experiment("DSPy")
            mlflow.start_run(run_name=f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            
            # Log experiment parameters
            mlflow.log_param("experiment_name", experiment_name)
            mlflow.log_param("data_path", data_path)
            mlflow.log_param("limit", limit or "all")
            mlflow.log_param("parallel_workers", parallel_workers)
            mlflow.log_param("memory_workers", memory_workers)
            mlflow.log_param("evaluation_type", "LOCOMO_exact_logic")
            
        except Exception as e:
            print(f"⚠️  MLflow run start warning: {e}")
            mlflow_tracking = False
    
    # Setup
    pipeline = SOTAEvaluationPipeline(mlflow_tracking=mlflow_tracking)
    
    # Load dataset
    print("📚 Loading LOCOMO dataset...")
    dataset = load_locomo_dataset(data_path)
    examples = dataset.get_examples(limit=limit)
    
    print(f"📊 Evaluating on {len(examples)} examples")
    if limit:
        print(f"   (limited from full dataset)")
    
    # Split into train/test for potential optimization
    train_size = int(0.7 * len(examples))
    train_examples = examples[:train_size]
    test_examples = examples[train_size:]
    
    print(f"🔄 Data split: {len(train_examples)} train, {len(test_examples)} test")
    
    # Build memory system from training data
    memory_system = pipeline.build_memory_from_dataset(
        dataset, limit=len(train_examples), parallel_workers=memory_workers
    )
    
    # Evaluate on test set
    results = pipeline.evaluate_system(test_examples, parallel_workers)
    
    # Print results
    overall_score = results["overall_locomo_score"]
    print(f"\n🎯 SOTA Memory System Results (LOCOMO Evaluation):")
    print(f"   Overall LOCOMO Score: {overall_score:.3f} ({overall_score*100:.1f}%)")
    print(f"   Overall Recall: {results['overall_recall']:.3f}")
    print(f"   Total Examples: {results['total_examples']}")
    
    if overall_score >= 0.68:
        print("🏆 TARGET ACHIEVED: >68% performance!")
    else:
        print(f"📈 Need {(0.68 - overall_score)*100:.1f}% more to reach 68% target")
    
    # Category breakdown with LOCOMO-specific details
    print("\n📋 Category Performance (LOCOMO Logic):")
    category_names = {
        1: "Multi-hop (F1 multi-answer)",
        2: "Single-hop (F1)", 
        3: "Temporal (F1)",
        4: "Open-domain (F1)",
        5: "Adversarial (exact match)"
    }
    
    for cat in range(1, 6):
        score_key = f"category_{cat}_score"
        count_key = f"category_{cat}_count"
        if score_key in results:
            score = results[score_key]
            count = results[count_key]
            name = category_names[cat]
            print(f"   {name}: {score:.3f} ({count} examples)")
    
    # Save results
    results_file = pipeline.save_results(results, experiment_name)
    
    # Log results file to MLflow
    if mlflow_tracking:
        try:
            mlflow.log_artifact(str(results_file))
            # Log final summary metrics
            mlflow.log_metric("final_locomo_score", overall_score)
            if overall_score >= 0.68:
                mlflow.log_metric("target_achieved", 1.0)
            else:
                mlflow.log_metric("target_achieved", 0.0)
                mlflow.log_metric("score_gap", 0.68 - overall_score)
            
            mlflow.end_run()
            print("✅ MLflow run completed")
        except Exception as e:
            print(f"⚠️  MLflow end run warning: {e}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="SOTA Memory System Evaluation for LOCOMO (using exact LOCOMO logic)")
    
    parser.add_argument("--data-path", default="./data/locomo10.json",
                       help="Path to LOCOMO dataset")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of examples (default: all)")
    parser.add_argument("--experiment-name", default="sota_locomo_evaluation",
                       help="Name for this experiment")
    parser.add_argument("--parallel-workers", type=int, default=4,
                       help="Number of parallel workers for evaluation")
    parser.add_argument("--memory-workers", type=int, default=1,
                       help="Number of parallel workers for memory building")
    parser.add_argument("--model", default="openai/gpt-4o-mini",
                       help="Language model to use (still needed for memory system)")
    
    args = parser.parse_args()
    
    # Setup DSPy (still needed for memory system, not for evaluation)
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found. Please set your OpenAI API key.")
        return
    
    lm = dspy.LM(args.model, api_key=api_key)
    dspy.configure(lm=lm)
    
    # Run evaluation
    results = run_sota_evaluation(
        data_path=args.data_path,
        limit=args.limit,
        experiment_name=args.experiment_name,
        parallel_workers=args.parallel_workers,
        memory_workers=args.memory_workers
    )
    
    return results


if __name__ == "__main__":
    main()
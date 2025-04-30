"""
Main entry point for agent system with modular teamwork components.
"""

import argparse
import logging
import os
import json
from typing import Dict, Any, List
from datetime import datetime

from simulator import AgentSystemSimulator
import config

def setup_logging():
    """Set up logging for the application."""
    # Create logs directory if it doesn't exist
    os.makedirs(config.LOG_DIR, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(config.LOG_DIR, "agent_system.log")),
            logging.StreamHandler()
        ]
    )

def run_simulation(
    use_team_leadership: bool = None, 
    use_closed_loop_comm: bool = None,
    use_mutual_monitoring: bool = None,
    use_shared_mental_model: bool = None,
    random_leader: bool = False,
    use_recruitment: bool = None,
    recruitment_method: str = None,
    recruitment_pool: str = None
) -> Dict[str, Any]:
    """
    Run a single agent system simulation with selected teamwork components.
    
    Args:
        use_team_leadership: Whether to use team leadership behaviors
        use_closed_loop_comm: Whether to use closed-loop communication
        use_mutual_monitoring: Whether to use mutual performance monitoring
        use_shared_mental_model: Whether to use shared mental models
        random_leader: Whether to randomly assign leadership
        use_recruitment: Whether to use dynamic agent recruitment
        recruitment_method: Method for recruitment (adaptive, basic, intermediate, advanced)
        recruitment_pool: Pool of agent roles to recruit from
        
    Returns:
        Simulation results
    """
    # Create simulator with specified configuration
    simulator = AgentSystemSimulator(
        use_team_leadership=use_team_leadership,
        use_closed_loop_comm=use_closed_loop_comm,
        use_mutual_monitoring=use_mutual_monitoring,
        use_shared_mental_model=use_shared_mental_model,
        random_leader=random_leader,
        use_recruitment=use_recruitment,
        recruitment_method=recruitment_method,
        recruitment_pool=recruitment_pool
    )
    
    # Run the simulation
    results = simulator.run_simulation()
    
    # Evaluate performance
    performance = simulator.evaluate_performance()
    results["performance"] = performance
    
    # Save updated results
    simulator.save_results()
    
    return results


def run_all_configurations(runs=1):
    """
    Run simulations with individual and all feature combinations.
    
    Args:
        runs: Number of runs per configuration
        
    Returns:
        Dictionary with results for each configuration
    """
    # Define configurations to test
    configurations = [
        # Baseline (no features)
        {
            "name": "Baseline", 
            "leadership": False, 
            "closed_loop": False,
            "mutual_monitoring": False,
            "shared_mental_model": False
        },
        # Single features
        {
            "name": "Leadership", 
            "leadership": True, 
            "closed_loop": False,
            "mutual_monitoring": False,
            "shared_mental_model": False
        },
        {
            "name": "Closed-loop", 
            "leadership": False, 
            "closed_loop": True,
            "mutual_monitoring": False,
            "shared_mental_model": False
        },
        {
            "name": "Mutual Monitoring", 
            "leadership": False, 
            "closed_loop": False,
            "mutual_monitoring": True,
            "shared_mental_model": False
        },
        {
            "name": "Shared Mental Model", 
            "leadership": False, 
            "closed_loop": False,
            "mutual_monitoring": False,
            "shared_mental_model": True
        },
        # All features
        {
            "name": "All Features", 
            "leadership": True, 
            "closed_loop": True,
            "mutual_monitoring": True,
            "shared_mental_model": True
        }
    ]
    
    # Store aggregated results by decision method and configuration
    aggregated_results = {
        "majority_voting": {},
        "weighted_voting": {},
        "borda_count": {}
    }
    
    # For each configuration
    for config in configurations:
        print(f"\nRunning {config['name']} configuration...")
        
        config_results = {
            "majority_voting": [],
            "weighted_voting": [],
            "borda_count": []
        }
        
        # Run specified number of times
        for run in range(1, runs + 1):
            print(f"  Run {run}/{runs}...")
            
            # Run simulation with this configuration
            result = run_simulation(
                use_team_leadership=config["leadership"],
                use_closed_loop_comm=config["closed_loop"],
                use_mutual_monitoring=config["mutual_monitoring"],
                use_shared_mental_model=config["shared_mental_model"]
            )
            
            # Extract decision results for each method
            for method in ["majority_voting", "weighted_voting", "borda_count"]:
                if method in result["decision_results"]:
                    config_results[method].append(result["decision_results"][method])
        
        # Store results for this configuration
        for method in ["majority_voting", "weighted_voting", "borda_count"]:
            aggregated_results[method][config["name"]] = config_results[method]
    
    # Calculate aggregated performance for ranking tasks
    if config.TASK["type"] == "ranking" and "ground_truth" in config.TASK:
        performance_metrics = calculate_ranking_performance(aggregated_results)
    elif config.TASK["type"] == "mcq" and "ground_truth" in config.TASK:
        performance_metrics = calculate_mcq_performance(aggregated_results)
    else:
        performance_metrics = {
            "note": "No quantitative metrics available for this task type or no ground truth provided"
        }
    
    # Save aggregate results
    output_path = os.path.join(config.OUTPUT_DIR, f"all_configurations_results.json")
    with open(output_path, 'w') as f:
        json.dump({
            "task": config.TASK["name"],
            "task_type": config.TASK["type"],
            "runs_per_configuration": runs,
            "configurations": configurations,
            "aggregated_results": aggregated_results,
            "performance_metrics": performance_metrics
        }, f, indent=2)
    
    print(f"\nDetailed results saved to: {output_path}")
    
    # Print summary for ranking tasks
    if config.TASK["type"] == "ranking" and "ground_truth" in config.TASK:
        print_ranking_summary(performance_metrics)
    elif config.TASK["type"] == "mcq" and "ground_truth" in config.TASK:
        print_mcq_summary(performance_metrics)
    
    return aggregated_results

def calculate_ranking_performance(aggregated_results):
    """Calculate performance metrics for ranking tasks across methods and configurations."""
    performance_metrics = {
        "average_correlation": {},
        "average_error": {}
    }
    
    for method, method_results in aggregated_results.items():
        performance_metrics["average_correlation"][method] = {}
        performance_metrics["average_error"][method] = {}
        
        for config_name, config_runs in method_results.items():
            # Extract correlations and errors from runs
            correlations = []
            errors = []
            
            for run_result in config_runs:
                if "correlation" in run_result:
                    correlations.append(run_result["correlation"])
                if "error" in run_result:
                    errors.append(run_result["error"])
            
            # Calculate averages
            if correlations:
                performance_metrics["average_correlation"][method][config_name] = sum(correlations) / len(correlations)
            if errors:
                performance_metrics["average_error"][method][config_name] = sum(errors) / len(errors)
    
    return performance_metrics

def calculate_mcq_performance(aggregated_results):
    """Calculate performance metrics for MCQ tasks across methods and configurations."""
    performance_metrics = {
        "accuracy": {},
        "average_confidence": {}
    }
    
    for method, method_results in aggregated_results.items():
        performance_metrics["accuracy"][method] = {}
        performance_metrics["average_confidence"][method] = {}
        
        for config_name, config_runs in method_results.items():
            # Extract correct answers and confidences from runs
            correct_count = 0
            confidences = []
            
            for run_result in config_runs:
                if "correct" in run_result:
                    correct_count += 1 if run_result["correct"] else 0
                if "confidence" in run_result:
                    confidences.append(run_result["confidence"])
            
            # Calculate metrics
            if config_runs:
                performance_metrics["accuracy"][method][config_name] = correct_count / len(config_runs)
            if confidences:
                performance_metrics["average_confidence"][method][config_name] = sum(confidences) / len(confidences)
    
    return performance_metrics

def print_ranking_summary(performance_metrics):
    """Print summary of ranking task performance metrics."""
    print("\n=== Performance Summary ===")
    
    # Print correlation summary
    print("\nAverage Correlation with Ground Truth (higher is better):")
    for method in ["majority_voting", "weighted_voting", "borda_count"]:
        if method in performance_metrics["average_correlation"]:
            print(f"\n  {method.replace('_', ' ').title()}:")
            correlations = performance_metrics["average_correlation"][method]
            sorted_configs = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
            
            for config, correlation in sorted_configs:
                print(f"    {config.ljust(20)}: {correlation:.4f}")
    
    # Print error summary
    print("\nAverage Error (lower is better):")
    for method in ["majority_voting", "weighted_voting", "borda_count"]:
        if method in performance_metrics["average_error"]:
            print(f"\n  {method.replace('_', ' ').title()}:")
            errors = performance_metrics["average_error"][method]
            sorted_configs = sorted(errors.items(), key=lambda x: x[1])
            
            for config, error in sorted_configs:
                print(f"    {config.ljust(20)}: {error:.2f}")

def print_mcq_summary(performance_metrics):
    """Print summary of MCQ task performance metrics."""
    print("\n=== Performance Summary ===")
    
    # Print accuracy summary
    print("\nAccuracy (higher is better):")
    for method in ["majority_voting", "weighted_voting", "borda_count"]:
        if method in performance_metrics["accuracy"]:
            print(f"\n  {method.replace('_', ' ').title()}:")
            accuracies = performance_metrics["accuracy"][method]
            sorted_configs = sorted(accuracies.items(), key=lambda x: x[1], reverse=True)
            
            for config, accuracy in sorted_configs:
                print(f"    {config.ljust(20)}: {accuracy:.2f}")
    
    # Print confidence summary
    print("\nAverage Confidence:")
    for method in ["majority_voting", "weighted_voting", "borda_count"]:
        if method in performance_metrics["average_confidence"]:
            print(f"\n  {method.replace('_', ' ').title()}:")
            confidences = performance_metrics["average_confidence"][method]
            sorted_configs = sorted(confidences.items(), key=lambda x: x[1], reverse=True)
            
            for config, confidence in sorted_configs:
                print(f"    {config.ljust(20)}: {confidence:.2f}")

# Add command-line arguments in main.py
def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run agent system simulations with teamwork components')
    parser.add_argument('--leadership', action='store_true', help='Use team leadership')
    parser.add_argument('--closedloop', action='store_true', help='Use closed-loop communication')
    parser.add_argument('--mutual', action='store_true', help='Use mutual performance monitoring')
    parser.add_argument('--mental', action='store_true', help='Use shared mental model')
    parser.add_argument('--all', action='store_true', help='Run all feature combinations')
    parser.add_argument('--random-leader', action='store_true', help='Randomly assign leadership')
    parser.add_argument('--runs', type=int, default=1, help='Number of runs for each configuration')
    
    # Add recruitment arguments
    parser.add_argument('--recruitment', action='store_true', help='Use dynamic agent recruitment')
    parser.add_argument('--recruitment-method', type=str, choices=['adaptive', 'basic', 'intermediate', 'advanced'], 
                      default='adaptive', help='Recruitment method to use')
    parser.add_argument('--recruitment-pool', type=str, choices=['general', 'medical'], 
                      default='general', help='Pool of roles to recruit from')
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging()
    
    # Display task information
    print(f"\nTask: {config.TASK['name']}")
    print(f"Type: {config.TASK['type']}")
    
    if args.all:
        logging.info(f"Running all feature combinations with {args.runs} runs per configuration")
        run_all_configurations(runs=args.runs)
    else:
        logging.info("Running single simulation")
        
        # Default to all components if none specified
        use_any = args.leadership or args.closedloop or args.mutual or args.mental or args.recruitment
        
        result = run_simulation(
            use_team_leadership=args.leadership if use_any else None,
            use_closed_loop_comm=args.closedloop if use_any else None,
            use_mutual_monitoring=args.mutual if use_any else None,
            use_shared_mental_model=args.mental if use_any else None,
            random_leader=args.random_leader,
            use_recruitment=args.recruitment if use_any else None,
            recruitment_method=args.recruitment_method,
            recruitment_pool=args.recruitment_pool
        )
        
        # Determine which features were used
        features = []
        if result["config"]["use_team_leadership"]:
            features.append("Team Leadership")
        if result["config"]["use_closed_loop_comm"]:
            features.append("Closed-loop Communication")
        if result["config"]["use_mutual_monitoring"]:
            features.append("Mutual Performance Monitoring")
        if result["config"]["use_shared_mental_model"]:
            features.append("Shared Mental Model")
        
        features_str = ", ".join(features) if features else "None"
        
        print(f"\nResults Summary:")
        print(f"Teamwork Features: {features_str}")
        
        # Print decision method results
        print("\nDecision Method Results:")
        
        for method, results in result["decision_results"].items():
            print(f"\n{method.replace('_', ' ').title()}:")
            
            if config.TASK["type"] == "ranking":
                if "final_ranking" in results and results["final_ranking"]:
                    print(f"  Top 3 items:")
                    for i, item in enumerate(results["final_ranking"][:3]):
                        print(f"    {i+1}. {item}")
                    print(f"  Confidence: {results.get('confidence', 'N/A'):.2f}")
            elif config.TASK["type"] == "mcq":
                if "winning_option" in results:
                    print(f"  Selected option: {results['winning_option']}")
                    print(f"  Confidence: {results.get('confidence', 'N/A'):.2f}")
            else:
                print(f"  Result: {results}")
        
        # Show where to find logs
        sim_id = result["simulation_id"]
        feature_dir = "_".join(features).lower().replace(" ", "_") if features else "baseline"
        
        print(f"\nDetailed logs available at: {os.path.join(config.LOG_DIR, feature_dir, sim_id)}")
        
        # Show a performance summary if ground truth is available
        if "performance" in result:
            print("\nPerformance Summary:")
            
            task_perf = result["performance"].get("task_performance", {})
            
            if task_perf:
                if config.TASK["type"] == "ranking":
                    for method, metrics in task_perf.items():
                        print(f"  {method.replace('_', ' ').title()}:")
                        print(f"    Correlation: {metrics.get('correlation', 'N/A'):.4f}")
                        print(f"    Error: {metrics.get('error', 'N/A')}")
                elif config.TASK["type"] == "mcq":
                    for method, metrics in task_perf.items():
                        print(f"  {method.replace('_', ' ').title()}:")
                        print(f"    Correct: {metrics.get('correct', 'N/A')}")
                        print(f"    Confidence: {metrics.get('confidence', 'N/A'):.2f}")
            else:
                print("  No quantitative metrics available")

if __name__ == "__main__":
    main()
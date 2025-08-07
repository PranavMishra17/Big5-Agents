"""
Main entry point for agent system with modular teamwork components.
Updated to work with the new question-level parallel processing approach.
"""

import argparse
import logging
import os
import json
from typing import Dict, Any, List
from datetime import datetime
import time

from simulator import AgentSystemSimulator
import config

import copy
from typing import List, Dict, Any, Optional, Tuple

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
    use_team_orientation=None,  # New parameter
    use_mutual_trust=None,      # New parameter
    mutual_trust_factor=0.8,    # New parameter
    random_leader: bool = False,
    use_recruitment: bool = None,
    recruitment_method: str = None,
    recruitment_pool: str = None,
    n_max: int = 4,
    runs= 1,
    deployment_config: Dict[str, str] = None
) -> Dict[str, Any]:
    """
    Run a single agent system simulation with selected teamwork components.
    
    Args:
        use_team_leadership: Whether to use team leadership
        use_closed_loop_comm: Whether to use closed-loop communication
        use_mutual_monitoring: Whether to use mutual performance monitoring
        use_shared_mental_model: Whether to use shared mental models
        use_team_orientation: Whether to use team orientation
        use_mutual_trust: Whether to use mutual trust
        mutual_trust_factor: Trust factor for mutual trust (0.0-1.0)
        random_leader: Whether to randomly assign leadership
        use_recruitment: Whether to use dynamic agent recruitment
        recruitment_method: Method for recruitment
        recruitment_pool: Pool of agent roles to recruit from
        runs: Number of simulation runs
        deployment_config: Optional specific deployment configuration
        
    Returns:
        Simulation results
    """
    results = []

    for i in range(runs):
        # Create simulator with the specified components
        simulator = AgentSystemSimulator(
            use_team_leadership=use_team_leadership,
            use_closed_loop_comm=use_closed_loop_comm,
            use_mutual_monitoring=use_mutual_monitoring,
            use_shared_mental_model=use_shared_mental_model,
            use_team_orientation=use_team_orientation,
            use_mutual_trust=use_mutual_trust,
            mutual_trust_factor=mutual_trust_factor,
            random_leader=random_leader,
            use_recruitment=use_recruitment,
            recruitment_method=recruitment_method,
            recruitment_pool=recruitment_pool,
            n_max=n_max,
            deployment_config=deployment_config
        )
        
        # Run the simulation
        result = simulator.run_simulation()
        results.append(result)
    
    # Return the last result (or aggregated if multiple runs)
    if runs == 1:
        return results[0]
    else:
        # Aggregate results from multiple runs
        return aggregate_results(results)
    

def aggregate_results(results_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate results from multiple simulation runs.
    
    Args:
        results_list: List of result dictionaries from individual runs
        
    Returns:
        Aggregated results dictionary
    """
    if not results_list:
        return {}
        
    # Start with the first result as a template
    aggregated = copy.deepcopy(results_list[0])
    
    # Update simulation ID to indicate aggregation
    aggregated["simulation_metadata"]["simulation_id"] = f"{aggregated['simulation_metadata']['simulation_id']}_aggregated_{len(results_list)}_runs"
    
    # Initialize aggregation containers
    aggregated["decision_results"]["aggregated"] = {
        "majority_voting": {"correct_count": 0, "total_runs": len(results_list), "avg_confidence": 0.0},
        "weighted_voting": {"correct_count": 0, "total_runs": len(results_list), "avg_confidence": 0.0},
        "borda_count": {"correct_count": 0, "total_runs": len(results_list), "avg_confidence": 0.0}
    }
    
    # Track all run results
    aggregated["all_runs"] = []
    
    # Aggregate metrics across runs
    total_trust_level = 0.0
    total_team_orientation_score = 0.0
    trust_count = 0
    orientation_count = 0
    
    for result in results_list:
        # Add run info
        run_summary = {
            "simulation_id": result["simulation_metadata"]["simulation_id"],
            "decision_results": result["decision_results"]
        }
        aggregated["all_runs"].append(run_summary)
        
        # Aggregate decision results
        for method in ["majority_voting", "weighted_voting", "borda_count"]:
            if result["decision_results"].get(method) and result["decision_results"][method].get("correct"):
                aggregated["decision_results"]["aggregated"][method]["correct_count"] += 1
                
            if result["decision_results"].get(method) and result["decision_results"][method].get("confidence"):
                aggregated["decision_results"]["aggregated"][method]["avg_confidence"] += \
                    result["decision_results"][method]["confidence"]
    
    # Calculate averages
    for method in ["majority_voting", "weighted_voting", "borda_count"]:
        if len(results_list) > 0:
            aggregated["decision_results"]["aggregated"][method]["avg_confidence"] /= len(results_list)
            aggregated["decision_results"]["aggregated"][method]["accuracy"] = \
                aggregated["decision_results"]["aggregated"][method]["correct_count"] / len(results_list)
    
    return aggregated


def run_all_configurations(runs=1, n_max=4):
    """
    Run simulations with individual and all feature combinations.
    
    Args:
        runs: Number of runs per configuration
        n_max: Maximum number of agents for intermediate teams
        
    Returns:
        Dictionary with results for each configuration
    """
    # Get available deployments for single-question runs
    deployments = config.get_all_deployments()
    default_deployment = deployments[0] if deployments else None
    
    # Define configurations to test
    configurations = [
        # No teamwork components, no recruitment
        {
            "name": "Baseline", 
            "leadership": False, 
            "closed_loop": False,
            "mutual_monitoring": False,
            "shared_mental_model": False,
            "team_orientation": False,
            "mutual_trust": False,
            "recruitment": False
        },
        # No teamwork components, only basic recruitment (1 agent)
        {
            "name": "Basic Recruitment Only", 
            "leadership": False, 
            "closed_loop": False,
            "mutual_monitoring": False,
            "shared_mental_model": False,
            "team_orientation": False,
            "mutual_trust": False,
            "recruitment": True,
            "recruitment_method": "basic" 
        },
        # Single features
        {
            "name": "Leadership", 
            "leadership": True, 
            "closed_loop": False,
            "mutual_monitoring": False,
            "shared_mental_model": False,
            "team_orientation": False,
            "mutual_trust": False,
            "recruitment": True,
            "recruitment_method": "intermediate",
            "n_max": n_max
        },
        {
            "name": "Closed-loop", 
            "leadership": False, 
            "closed_loop": True,
            "mutual_monitoring": False,
            "shared_mental_model": False,
            "team_orientation": False,
            "mutual_trust": False,
            "recruitment": True,
            "recruitment_method": "intermediate",
            "n_max": n_max
        },
        {
            "name": "Mutual Monitoring", 
            "leadership": False, 
            "closed_loop": False,
            "mutual_monitoring": True,
            "shared_mental_model": False,
            "team_orientation": False,
            "mutual_trust": False,
            "recruitment": True,
            "recruitment_method": "intermediate",
            "n_max": n_max
        },
        {
            "name": "Shared Mental Model", 
            "leadership": False, 
            "closed_loop": False,
            "mutual_monitoring": False,
            "shared_mental_model": True,
            "team_orientation": False,
            "mutual_trust": False,
            "recruitment": True,
            "recruitment_method": "intermediate",
            "n_max": n_max
        },
        # New single features
        {
            "name": "Team Orientation", 
            "leadership": False, 
            "closed_loop": False,
            "mutual_monitoring": False,
            "shared_mental_model": False,
            "team_orientation": True,
            "mutual_trust": False,
            "recruitment": True,
            "recruitment_method": "intermediate",
            "n_max": n_max
        },
        {
            "name": "Mutual Trust", 
            "leadership": False, 
            "closed_loop": False,
            "mutual_monitoring": False,
            "shared_mental_model": False,
            "team_orientation": False,
            "mutual_trust": True,
            "mutual_trust_factor": 0.8,
            "recruitment": True,
            "recruitment_method": "intermediate",
            "n_max": n_max
        },
        {
            "name": "High Trust", 
            "leadership": False, 
            "closed_loop": False,
            "mutual_monitoring": False,
            "shared_mental_model": False,
            "team_orientation": False,
            "mutual_trust": True,
            "mutual_trust_factor": 0.9,
            "recruitment": True,
            "recruitment_method": "intermediate",
            "n_max": n_max
        },
        {
            "name": "Low Trust", 
            "leadership": False, 
            "closed_loop": False,
            "mutual_monitoring": False,
            "shared_mental_model": False,
            "team_orientation": False,
            "mutual_trust": True,
            "mutual_trust_factor": 0.3,
            "recruitment": True,
            "recruitment_method": "intermediate",
            "n_max": n_max
        },
        # Combination features
        {
            "name": "Team Orientation + Trust", 
            "leadership": False, 
            "closed_loop": False,
            "mutual_monitoring": False,
            "shared_mental_model": False,
            "team_orientation": True,
            "mutual_trust": True,
            "recruitment": True,
            "recruitment_method": "intermediate",
            "n_max": n_max
        },
        # All features
        {
            "name": "Original Big 5", 
            "leadership": True, 
            "closed_loop": True,
            "mutual_monitoring": True,
            "shared_mental_model": True,
            "team_orientation": False,
            "mutual_trust": False,
            "recruitment": True,
            "recruitment_method": "intermediate",
            "n_max": n_max
        },
        {
            "name": "All Features", 
            "leadership": True, 
            "closed_loop": True,
            "mutual_monitoring": True,
            "shared_mental_model": True,
            "team_orientation": True,
            "mutual_trust": True,
            "recruitment": True,
            "recruitment_method": "intermediate",
            "n_max": n_max
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
                use_shared_mental_model=config["shared_mental_model"],
                use_team_orientation=config.get("team_orientation", False),
                use_mutual_trust=config.get("mutual_trust", False),
                mutual_trust_factor=config.get("mutual_trust_factor", 0.8),
                use_recruitment=config.get("recruitment", False),
                recruitment_method=config.get("recruitment_method", "adaptive"),
                recruitment_pool=config.get("recruitment_pool", "general"),
                n_max=config.get("n_max", n_max),
                deployment_config=default_deployment
            )
            
            # Extract decision results for each method
            for method in ["majority_voting", "weighted_voting", "borda_count"]:
                if method in result["decision_results"]:
                    config_results[method].append(result["decision_results"][method])
        
        # Store results for this configuration
        for method in ["majority_voting", "weighted_voting", "borda_count"]:
            aggregated_results[method][config["name"]] = config_results[method]
    
    # Separate GT from agent task for single-task runs
    if hasattr(config, 'TASK') and 'ground_truth' in config.TASK:
        eval_data = {
            "ground_truth": config.TASK.pop("ground_truth"),
            "rationale": config.TASK.pop("rationale", {})
        }
        config.TASK_EVALUATION = eval_data

    # Calculate aggregated performance
    if config.TASK["type"] == "ranking" and hasattr(config, "TASK_EVALUATION"):
        performance_metrics = calculate_ranking_performance(aggregated_results, config.TASK_EVALUATION)
    elif config.TASK["type"] == "mcq" and hasattr(config, "TASK_EVALUATION"):
        performance_metrics = calculate_mcq_performance(aggregated_results, config.TASK_EVALUATION)
    else:
        performance_metrics = {
            "note": "No quantitative metrics available for this task type or no ground truth provided"
        }

    # Save aggregate results
    output_path = os.path.join(config.OUTPUT_DIR, f"all_configurations_results.json")
    with open(output_path, 'w') as f:
        json.dump({
            "task": config.TASK.get("name", "Unnamed Task"),
            "task_type": config.TASK.get("type", "unknown"),
            "runs_per_configuration": runs,
            "n_max": n_max,
            "deployment_used": default_deployment['name'] if default_deployment else "default",
            "configurations": configurations,
            "aggregated_results": aggregated_results,
            "performance_metrics": performance_metrics
        }, f, indent=2)

    print(f"\nDetailed results saved to: {output_path}")

    # Print summary
    if config.TASK["type"] == "ranking" and hasattr(config, "TASK_EVALUATION"):
        print_ranking_summary(performance_metrics)
    elif config.TASK["type"] == "mcq" and hasattr(config, "TASK_EVALUATION"):
        print_mcq_summary(performance_metrics)

    return aggregated_results


def calculate_ranking_performance(aggregated_results, eval_data):
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

def calculate_mcq_performance(aggregated_results, eval_data):
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
    parser.add_argument('--orientation', action='store_true', help='Use team orientation')
    parser.add_argument('--trust', action='store_true', help='Use mutual trust')
    parser.add_argument('--trust-factor', type=float, default=0.8, help='Mutual trust factor (0.0-1.0)')
    parser.add_argument('--all', action='store_true', help='Run all feature combinations')
    parser.add_argument('--random-leader', action='store_true', help='Randomly assign leadership')
    parser.add_argument('--runs', type=int, default=1, help='Number of runs for each configuration')
    parser.add_argument('--recruitment', action='store_true', help='Use dynamic agent recruitment')
    parser.add_argument('--recruitment-method', type=str, choices=['adaptive', 'basic', 'intermediate', 'advanced'], 
                      default='adaptive', help='Recruitment method to use')
    parser.add_argument('--recruitment-pool', type=str, choices=['general', 'medical'], 
                      default='general', help='Pool of roles to recruit from')
    
    parser.add_argument('--n-max', type=int, default=5, 
                      help='Maximum number of agents for intermediate team (default: 5)')
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging()
    
    # Log deployment configuration
    deployments = config.get_all_deployments()
    logging.info(f"Available deployments: {[d['name'] for d in deployments]}")
    logging.info(f"Single-question simulation will use first deployment: {deployments[0]['name'] if deployments else 'default'}")
    
    # Display task information
    print(f"\nTask: {config.TASK['name']}")
    print(f"Type: {config.TASK['type']}")
    
    # If n_max is specified, automatically set recruitment method to intermediate
    if args.n_max is not None:
        args.recruitment = True
        args.recruitment_method = "intermediate"
    
    if args.all:
        logging.info(f"Running all feature combinations with {args.runs} runs per configuration, n_max={args.n_max}")
        run_all_configurations(runs=args.runs, n_max=args.n_max)
    else:
        logging.info("Running single simulation")
        
        # Default to all components if none specified
        use_any = args.leadership or args.closedloop or args.mutual or args.mental or args.orientation or args.trust
        
        # If recruitment method is specified, enable recruitment automatically
        if args.recruitment_method != "adaptive":
            args.recruitment = True
        
        # Get deployment for single simulation
        deployments = config.get_all_deployments()
        deployment_config = deployments[0] if deployments else None
        
        result = run_simulation(
            use_team_leadership=args.leadership if use_any else None,
            use_closed_loop_comm=args.closedloop if use_any else None,
            use_mutual_monitoring=args.mutual if use_any else None,
            use_shared_mental_model=args.mental if use_any else None,
            use_team_orientation=args.orientation if use_any else None,
            use_mutual_trust=args.trust if use_any else None,
            mutual_trust_factor=args.trust_factor,
            random_leader=args.random_leader,
            use_recruitment=args.recruitment,
            recruitment_method=args.recruitment_method,
            recruitment_pool=args.recruitment_pool,
            n_max=args.n_max if args.n_max is not None else 4,
            deployment_config=deployment_config
        )
        
        # Determine which features were used from simulation metadata
        simulation_config = result.get("simulation_metadata", {})
        features = []
        
        # Since the result format has changed, we need to check the config differently
        if hasattr(config, 'USE_TEAM_LEADERSHIP') and config.USE_TEAM_LEADERSHIP:
            features.append("Team Leadership")
        if hasattr(config, 'USE_CLOSED_LOOP_COMM') and config.USE_CLOSED_LOOP_COMM:
            features.append("Closed-loop Communication")
        if hasattr(config, 'USE_MUTUAL_MONITORING') and config.USE_MUTUAL_MONITORING:
            features.append("Mutual Performance Monitoring")
        if hasattr(config, 'USE_SHARED_MENTAL_MODEL') and config.USE_SHARED_MENTAL_MODEL:
            features.append("Shared Mental Model")
        if hasattr(config, 'USE_TEAM_ORIENTATION') and config.USE_TEAM_ORIENTATION:
            features.append("Team Orientation")
        if hasattr(config, 'USE_MUTUAL_TRUST') and config.USE_MUTUAL_TRUST:
            features.append(f"Mutual Trust (factor: {config.MUTUAL_TRUST_FACTOR:.1f})")
        if hasattr(config, 'USE_AGENT_RECRUITMENT') and config.USE_AGENT_RECRUITMENT:
            features.append(f"Agent Recruitment ({config.RECRUITMENT_METHOD})")
        
        features_str = ", ".join(features) if features else "None"
        
        print(f"\nResults Summary:")
        print(f"Teamwork Features: {features_str}")
        print(f"Deployment Used: {deployment_config['name'] if deployment_config else 'default'}")
        
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
        sim_id = result["simulation_metadata"]["simulation_id"]
        
        print(f"\nDetailed logs available at: {os.path.join(config.LOG_DIR, sim_id)}")
        
        # Show a performance summary if ground truth is available
        if hasattr(config, "TASK_EVALUATION") and config.TASK_EVALUATION:
            print("\nPerformance Summary:")
            
            # We need to evaluate performance manually since it's not returned in the new format
            try:
                # Create a temporary simulator to evaluate performance
                temp_simulator = AgentSystemSimulator(
                    deployment_config=deployment_config
                )
                temp_simulator.results = {"decision_results": result["decision_results"]}
                performance = temp_simulator.evaluate_performance()
                task_perf = performance.get("task_performance", {})
                
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
            except Exception as e:
                print(f"  Error evaluating performance: {str(e)}")

if __name__ == "__main__":
    main()
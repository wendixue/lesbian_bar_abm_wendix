import pandas as pd
import numpy as np
from model import LGBTQBarModel
import time

def run_batch_experiment():
    """
    Run batch experiment to test different gamma values
    """
    # Experiment parameters
    gamma_values = [0.3, 0.5, 0.7]  # Gamma values to test
    num_runs = 20  # Number of runs per gamma value
    num_steps = 100  # Run for 100 steps each time
    
    # Fixed parameters
    fixed_params = {
        'population_size': 200,
        'alpha': 0.5,
        'QW_ratio': 0.5,
        'QNW_ratio': 0.25,
        'adaptive_update_interval': 10
    }
    
    # Store results
    results = []
    
    print("Starting batch experiment...")
    print(f"Testing gamma values: {gamma_values}")
    print(f"Running {num_runs} times for each gamma, {num_steps} steps each")
    print("-" * 50)
    
    # Experiment for each gamma value
    for gamma in gamma_values:
        print(f"\nTesting gamma = {gamma}")
        
        for run_id in range(num_runs):
            print(f"  Run {run_id + 1}/{num_runs}...", end=" ")
            start_time = time.time()
            
            # Create model instance
            model = LGBTQBarModel(
                gamma=gamma,
                seed=run_id,  # Use run_id as seed for reproducibility
                **fixed_params
            )
            
            # Run model
            for step in range(num_steps):
                model.step()
            
            # Collect final results - only effective affinity and QW ratio
            final_data = {
                'gamma': gamma,
                'run_id': run_id,
                # Women Bar data
                'women_bar_qw_effective_affinity': model.women_bar.calculate_effective_affinity()["QW"],
                'women_bar_qw_ratio': model.get_bar_group_ratio(0, "QW"),
                # Queer Bar data
                'queer_bar_qw_effective_affinity': model.queer_bar.calculate_effective_affinity()["QW"],
                'queer_bar_qw_ratio': model.get_bar_group_ratio(1, "QW")
            }
            
            results.append(final_data)
            
            end_time = time.time()
            print(f"Done ({end_time - start_time:.2f}s)")
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    df.to_csv('/Users/xuewendi/Desktop/batch_run_results.csv', index=False)
    print(f"\nResults saved to 'batch_run_results.csv'")
    
    return df

def print_summary(df):
    """
    Print simple summary of results
    """
    print("\n" + "="*50)
    print("RESULTS SUMMARY")
    print("="*50)
    
    for gamma in [0.3, 0.5, 0.7]:
        gamma_data = df[df['gamma'] == gamma]
        
        print(f"\nGamma = {gamma}:")
        print(f"Women Bar - QW Effective Affinity: {gamma_data['women_bar_qw_effective_affinity'].mean():.3f}")
        print(f"Women Bar - QW Ratio: {gamma_data['women_bar_qw_ratio'].mean():.3f}")
        print(f"Queer Bar - QW Effective Affinity: {gamma_data['queer_bar_qw_effective_affinity'].mean():.3f}")
        print(f"Queer Bar - QW Ratio: {gamma_data['queer_bar_qw_ratio'].mean():.3f}")

if __name__ == "__main__":
    # Run batch experiment
    df = run_batch_experiment()
    
    # Print summary
    print_summary(df)
    
    print("\nExperiment completed!")
    print("Results saved to: batch_run_results.csv")
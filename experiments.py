import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

np.random.seed(42)
num_average_time = 10  # Number of times to run each experiment to calculate the average values

def create_fake_data(N, M, input_type='discrete', output_type='discrete'):
    """
    Create fake dataset with N samples and M features
    
    Parameters:
    N: Number of samples
    M: Number of features
    input_type: 'discrete' for binary features, 'real' for continuous features
    output_type: 'discrete' for classification, 'real' for regression
    
    Returns:
    X_df: DataFrame with features
    y_series: Series with target values
    """
    # Create features based on input type
    if input_type == 'discrete':
        # Create M binary/discrete features
        X = np.random.randint(0, 3, size=(N, M))  # 3 possible values: 0, 1, 2
    else:
        # Create M continuous features
        X = np.random.randn(N, M)
    
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(M)])
    
    # Create target based on output type
    if output_type == 'discrete':
        # Create discrete output (classification)
        y = np.random.randint(0, 3, size=N)  # 3 classes
        # Add some dependency on features to make it learnable
        for i in range(min(3, M)):
            if input_type == 'discrete':
                mask = X[:, i] == 1
            else:
                mask = X[:, i] > 0
            y[mask] = (y[mask] + i) % 3
    else:
        # Create real output (regression)
        y = np.random.normal(0, 1, N)
        # Add dependency on features
        for i in range(min(5, M)):
            if input_type == 'discrete':
                y += X[:, i] * np.random.normal(0.5, 0.1, N)
            else:
                y += X[:, i] * 0.3
    
    y_series = pd.Series(y)
    return X_df, y_series

def time_single_experiment(N, M, input_type='discrete', output_type='discrete', max_depth=5):
    """
    Time a single experiment for given parameters
    
    Returns:
    fit_time: Time taken to fit the tree
    predict_time: Time taken to predict on test data
    """
    # Create data
    X, y = create_fake_data(N, M, input_type, output_type)
    
    # Split data (80% train, 20% test)
    train_size = int(0.8 * N)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Ensure we have test data
    if len(X_test) == 0:
        X_test = X_train[:min(10, len(X_train))]
        y_test = y_train[:min(10, len(y_train))]
    
    # Create decision tree
    tree = DecisionTree(criterion="information_gain", max_depth=max_depth)
    
    # Time fitting
    start_time = time.time()
    tree.fit(X_train, y_train)
    fit_time = time.time() - start_time
    
    # Time prediction
    start_time = time.time()
    predictions = tree.predict(X_test)
    predict_time = time.time() - start_time
    
    return fit_time, predict_time

def calculate_average_time_varying_N(N_values, M_fixed, input_type, output_type, num_runs=10):
    """
    Calculate average time for varying N (keeping M fixed)
    """
    results = {
        'N_values': [],
        'fit_times_mean': [],
        'fit_times_std': [],
        'predict_times_mean': [],
        'predict_times_std': []
    }
    
    print(f"  Varying N (M={M_fixed}):")
    
    for N in N_values:
        fit_times = []
        predict_times = []
        
        for run in range(num_runs):
            try:
                fit_time, predict_time = time_single_experiment(N, M_fixed, input_type, output_type)
                fit_times.append(fit_time)
                predict_times.append(predict_time)
            except Exception as e:
                print(f"    Error in run {run} for N={N}: {e}")
                continue
        
        if len(fit_times) > 0:
            results['N_values'].append(N)
            results['fit_times_mean'].append(np.mean(fit_times))
            results['fit_times_std'].append(np.std(fit_times))
            results['predict_times_mean'].append(np.mean(predict_times))
            results['predict_times_std'].append(np.std(predict_times))
            
            print(f"    N={N:4d}: Fit={np.mean(fit_times):.4f}±{np.std(fit_times):.4f}s, "
                  f"Predict={np.mean(predict_times):.4f}±{np.std(predict_times):.4f}s")
        else:
            print(f"    N={N}: Failed all runs")
    
    return results

def calculate_average_time_varying_M(M_values, N_fixed, input_type, output_type, num_runs=10):
    """
    Calculate average time for varying M (keeping N fixed)
    """
    results = {
        'M_values': [],
        'fit_times_mean': [],
        'fit_times_std': [],
        'predict_times_mean': [],
        'predict_times_std': []
    }
    
    print(f"  Varying M (N={N_fixed}):")
    
    for M in M_values:
        fit_times = []
        predict_times = []
        
        for run in range(num_runs):
            try:
                fit_time, predict_time = time_single_experiment(N_fixed, M, input_type, output_type)
                fit_times.append(fit_time)
                predict_times.append(predict_time)
            except Exception as e:
                print(f"    Error in run {run} for M={M}: {e}")
                continue
        
        if len(fit_times) > 0:
            results['M_values'].append(M)
            results['fit_times_mean'].append(np.mean(fit_times))
            results['fit_times_std'].append(np.std(fit_times))
            results['predict_times_mean'].append(np.mean(predict_times))
            results['predict_times_std'].append(np.std(predict_times))
            
            print(f"    M={M:2d}: Fit={np.mean(fit_times):.4f}±{np.std(fit_times):.4f}s, "
                  f"Predict={np.mean(predict_times):.4f}±{np.std(predict_times):.4f}s")
        else:
            print(f"    M={M}: Failed all runs")
    
    return results

def plot_results(all_results):
    """
    Plot the timing results for all four cases
    """
    case_names = [
        ('discrete', 'discrete', 'Discrete Features\nDiscrete Output'),
        ('discrete', 'real', 'Discrete Features\nReal Output'),
        ('real', 'discrete', 'Real Features\nDiscrete Output'),
        ('real', 'real', 'Real Features\nReal Output')
    ]
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Decision Tree Runtime Complexity Analysis', fontsize=16)
    
    for i, (input_type, output_type, title) in enumerate(case_names):
        key = (input_type, output_type)
        if key not in all_results:
            continue
            
        n_results, m_results = all_results[key]
        
        # Plot 1: Time vs N (varying number of samples)
        ax1 = axes[0, i]
        if len(n_results['N_values']) > 0:
            ax1.errorbar(n_results['N_values'], n_results['fit_times_mean'], 
                        yerr=n_results['fit_times_std'], marker='o', label='Fit Time', 
                        capsize=3, capthick=1)
            ax1.errorbar(n_results['N_values'], n_results['predict_times_mean'], 
                        yerr=n_results['predict_times_std'], marker='s', label='Predict Time',
                        capsize=3, capthick=1)
        
        ax1.set_xlabel('Number of Samples (N)')
        ax1.set_ylabel('Time (seconds)')
        ax1.set_title(f'{title}\nTime vs N')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        
        # Plot 2: Time vs M (varying number of features)
        ax2 = axes[1, i]
        if len(m_results['M_values']) > 0:
            ax2.errorbar(m_results['M_values'], m_results['fit_times_mean'], 
                        yerr=m_results['fit_times_std'], marker='o', label='Fit Time',
                        capsize=3, capthick=1)
            ax2.errorbar(m_results['M_values'], m_results['predict_times_mean'], 
                        yerr=m_results['predict_times_std'], marker='s', label='Predict Time',
                        capsize=3, capthick=1)
        
        ax2.set_xlabel('Number of Features (M)')
        ax2.set_ylabel('Time (seconds)')
        ax2.set_title(f'{title}\nTime vs M')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log')
        ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.show()

def analyze_theoretical_complexity():
    """
    Print theoretical complexity analysis
    """
    print("\n" + "="*80)
    print("THEORETICAL TIME COMPLEXITY ANALYSIS")
    print("="*80)
    
    print("\n1. DECISION TREE LEARNING (FIT):")
    print("   Theoretical Complexity: O(N * M * log(N) * D)")
    print("   where:")
    print("   - N = number of samples")
    print("   - M = number of features") 
    print("   - D = maximum depth of tree")
    print("   - log(N) factor comes from sorting for finding best splits")
    
    print("\n2. DECISION TREE PREDICTION:")
    print("   Theoretical Complexity: O(N_test * D)")
    print("   where:")
    print("   - N_test = number of test samples")
    print("   - D = depth of tree (path from root to leaf)")
    print("   - Independent of M (number of features)")
    
    print("\n3. EXPECTED EXPERIMENTAL TRENDS:")
    print("   Fit Time:")
    print("   - Should increase with N (more samples to process)")
    print("   - Should increase with M (more features to evaluate at each split)")
    print("   - Growth rate: approximately N*log(N)*M")
    
    print("\n   Predict Time:")
    print("   - Should increase linearly with N (more test samples)")
    print("   - Should be roughly constant with M (only follows tree path)")
    print("   - Growth rate: approximately N_test")
    
    print("\n4. COMPARISON ACROSS FOUR CASES:")
    print("   All four cases should show similar complexity patterns:")
    print("   - Discrete vs Real features: Similar complexity")
    print("   - Discrete vs Real output: Similar complexity")
    print("   - Main difference may be in constant factors")

def run_experiments():
    """
    Run timing experiments for all four cases of decision trees
    """
    print("Decision Tree Runtime Complexity Experiments")
    print("="*60)
    print("Testing all four cases:")
    print("1. Discrete Features, Discrete Output")
    print("2. Discrete Features, Real Output")
    print("3. Real Features, Discrete Output")
    print("4. Real Features, Real Output")
    print("="*60)
    
    # Define parameter ranges
    N_values = [50, 100, 200, 500, 1000]    # Number of samples
    M_values = [5, 10, 20, 50]              # Number of features
    M_fixed = 10                            # Fixed M when varying N
    N_fixed = 500                           # Fixed N when varying M
    
    # Four cases of decision trees
    cases = [
        ('discrete', 'discrete'),  # Case 1: Discrete input, discrete output
        ('discrete', 'real'),      # Case 2: Discrete input, real output  
        ('real', 'discrete'),      # Case 3: Real input, discrete output
        ('real', 'real')           # Case 4: Real input, real output
    ]
    
    all_results = {}
    
    # Run experiments for each case
    for case_num, (input_type, output_type) in enumerate(cases, 1):
        print(f"\n{'='*60}")
        print(f"CASE {case_num}: {input_type.upper()} FEATURES, {output_type.upper()} OUTPUT")
        print(f"{'='*60}")
        
        try:
            # Test varying N
            n_results = calculate_average_time_varying_N(
                N_values, M_fixed, input_type, output_type, num_runs=num_average_time
            )
            
            # Test varying M  
            m_results = calculate_average_time_varying_M(
                M_values, N_fixed, input_type, output_type, num_runs=num_average_time
            )
            
            all_results[(input_type, output_type)] = (n_results, m_results)
            print(f"  ✓ Case {case_num} completed successfully")
            
        except Exception as e:
            print(f"  ✗ Error in Case {case_num}: {e}")
            continue
    
    # Plot results if we have data
    if all_results:
        print(f"\n{'='*60}")
        print("GENERATING PLOTS...")
        print(f"{'='*60}")
        plot_results(all_results)
    else:
        print("No results to plot!")
    
    # Print theoretical analysis
    analyze_theoretical_complexity()
    
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"Completed experiments for {len(all_results)}/4 cases")
    print("Check the generated plots to compare:")
    print("1. How fit/predict times scale with number of samples (N)")
    print("2. How fit/predict times scale with number of features (M)")
    print("3. Differences between the four decision tree cases")
    print("4. Comparison with theoretical complexity")
    
    return all_results

# Run the experiments
if __name__ == "__main__":
    results = run_experiments()
    print("\nExperiments completed!")
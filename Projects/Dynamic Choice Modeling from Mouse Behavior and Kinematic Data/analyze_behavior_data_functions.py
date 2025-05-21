import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import scipy.io as sio
from process_behavior_and_movement_data import SessionDataProcessor
import matplotlib as mpl
from scipy.stats import wilcoxon

class BehaviorDataAnalyzer:
    """
    A class for loading and analyzing behavioral session data.
    """
    
    def __init__(self, base_dir='/Volumes/Runyan5/Akhil/behavior/'):
        """
        Initialize the BehaviorDataAnalyzer.
        
        Args:
            base_dir (str): Base directory for data
        """
        self.base_dir = base_dir
        
    def load_session_data(self, mouse_name, date, verbose=False):
        """
        Load and process session data without plotting.
        
        Args:
            mouse_name (str): Mouse identifier
            date (str): Session date
            verbose (bool): Whether to print progress messages
            
        Returns:
            tuple: (task_df, trialized_data) - processed session data
        """
        # Create processor instance
        processor = SessionDataProcessor(mouse_name=mouse_name, date=date)
        
        if verbose:
            print("\nLooking for files:")
            print(f"Cell file: {os.path.join(self.base_dir, f'CB{mouse_name}', f'CB{mouse_name}_{date}_Cell.mat')}")
            print(f"Data file: {os.path.join(self.base_dir, f'CB{mouse_name}', f'CB{mouse_name}_{date}.mat')}")
            print(f"\nProcessing session for mouse {mouse_name} on date {date}")
        
        # Get task DataFrame and trialized movement data
        task_df, trialized_data = processor.get_session_data(mouse_name, date, self.base_dir, verbose=False)
        
        if verbose:
            print("\nSession Summary:")
            print(f"Task type: {task_df['task_type'].iloc[0]}")
            print(f"Total trials: {len(task_df)}")
            print(f"Performance: {task_df['outcome'].mean():.1%}")
        
        return task_df, trialized_data

class MovementAnalyzer:
    """
    A class for analyzing movement data from behavioral sessions.
    """
    
    def __init__(self):
        """
        Initialize the MovementAnalyzer.
        """
        pass
        
    def plot_velocity_mean_sem_by_context(self, all_trialized_data, all_task_dfs, figsize=(6, 2.5), dpi=800):
        """
        Compare mean X and Y velocities between opto and non-opto trials using mean and SEM,
        separated by context.
        
        Args:
            all_trialized_data (dict): Dictionary containing trialized data for each mouse
            all_task_dfs (dict): Dictionary containing task DataFrames for each mouse
            figsize (tuple): Figure size
            dpi (int): Figure resolution
        
        Returns:
            fig, axes: Figure and axes objects
        """
        from scipy.stats import sem, ttest_ind
        
        # Create figure with two subplots (X and Y velocity)
        fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
        
        # Define context colors and names
        context_colors = ['purple', '#EC008C', '#27AAE1']
        context_names = ['Congruent', 'Visual', 'Audio']
        
        # Collect all velocity data by context
        velocity_data = {
            'x': {(ctx, is_opto): [] for ctx in range(3) for is_opto in [0, 1]},
            'y': {(ctx, is_opto): [] for ctx in range(3) for is_opto in [0, 1]}
        }
        
        # For each mouse
        for mouse, sessions in all_trialized_data.items():
            # Get corresponding task data
            task_df = all_task_dfs[mouse]
            
            # Process each session
            for session_idx, session_data in enumerate(sessions):
                # Get session date
                if 'date' in task_df.columns:
                    session_date = task_df['date'].unique()[session_idx] if session_idx < len(task_df['date'].unique()) else None
                    session_task_df = task_df[task_df['date'] == session_date] if session_date else None
                else:
                    # If no date column, assume sessions are in order
                    session_task_df = task_df
                
                if session_task_df is None or len(session_task_df) == 0:
                    continue
                    
                # Extract velocity data from session
                if 'X_velocity' in session_data and 'Y_velocity' in session_data:
                    x_vel = session_data['X_velocity']
                    y_vel = session_data['Y_velocity']
                    
                    # Process each trial
                    for trial_idx, trial_row in session_task_df.iterrows():
                        if trial_idx >= len(x_vel) or len(x_vel[trial_idx]) == 0:
                            continue
                            
                        # Get context and opto status
                        context = trial_row['context']
                        is_opto = trial_row['opto']
                        
                        # Process X velocity
                        x_data = np.array(x_vel[trial_idx, 100:150])
                        x_data = x_data[~np.isnan(x_data)]
                        if len(x_data) > 0:
                            velocity_data['x'][(context, is_opto)].append(np.mean(np.abs(x_data)))
                        
                        # Process Y velocity
                        y_data = np.array(y_vel[trial_idx, :150])
                        y_data = y_data[~np.isnan(y_data)]
                        if len(y_data) > 0:
                            velocity_data['y'][(context, is_opto)].append(np.mean(np.abs(y_data)))
        
        # Plot data for each dimension
        for dim_idx, dim in enumerate(['x', 'y']):
            ax = axes[dim_idx]
            
            # Calculate mean and SEM for each context and condition
            means_no_opto = []
            means_opto = []
            sems_no_opto = []
            sems_opto = []
            
            for ctx in range(3):
                # Non-opto data
                no_opto_data = velocity_data[dim][(ctx, 0)]
                if no_opto_data:
                    means_no_opto.append(np.mean(no_opto_data))
                    sems_no_opto.append(sem(no_opto_data))
                else:
                    means_no_opto.append(np.nan)
                    sems_no_opto.append(np.nan)
                
                # Opto data
                opto_data = velocity_data[dim][(ctx, 1)]
                if opto_data:
                    means_opto.append(np.mean(opto_data))
                    sems_opto.append(sem(opto_data))
                else:
                    means_opto.append(np.nan)
                    sems_opto.append(np.nan)
            
            # Plot non-opto data (circles)
            for ctx in range(3):
                if not np.isnan(means_no_opto[ctx]):
                    ax.errorbar(ctx, means_no_opto[ctx], yerr=sems_no_opto[ctx],
                               fmt='o', color=context_colors[ctx], 
                               markersize=10, capsize=5, capthick=2,
                               label=f"{context_names[ctx]} NoStim" if ctx == 0 else "")
            
            # Plot opto data (stars)
            for ctx in range(3):
                if not np.isnan(means_opto[ctx]):
                    ax.errorbar(ctx + 0.25, means_opto[ctx], yerr=sems_opto[ctx],
                               fmt='*', color=context_colors[ctx], 
                               markersize=15, capsize=5, capthick=2,
                               label=f"{context_names[ctx]} Stim" if ctx == 0 else "")
            
            # Add statistical tests for each context
            for ctx in range(3):
                opto_data = velocity_data[dim][(ctx, 1)]
                non_opto_data = velocity_data[dim][(ctx, 0)]
                
                if len(opto_data) > 5 and len(non_opto_data) > 5:
                    t_stat, p_value = ttest_ind(opto_data, non_opto_data, equal_var=False)
                    
                    if p_value > 0.05:
                        text = "NS"
                    elif p_value > 0.01:
                        text = "*"
                    elif p_value > 0.001:
                        text = "**"
                    else:
                        text = "***"
                    
                    # Add significance marker
                    y_pos = max(means_opto[ctx] + sems_opto[ctx], 
                               means_no_opto[ctx] + sems_no_opto[ctx]) * 1.005
                    ax.text(ctx + 0.125, y_pos, text, ha='center', va='bottom', 
                           color=context_colors[ctx], fontsize=4)
            
            # Customize plot
            ax.set_ylabel(f'Mean {dim.upper()} Velocity')
            ax.set_title(f'{dim.upper()} Velocity')
            ax.set_xticks(np.arange(3) + 0.125)
            ax.set_xticklabels(context_names, fontsize=12)
            ax.set_xlim(-0.5, 2.5)
            
            # Add custom legend
            if dim_idx == 0:
                from matplotlib.lines import Line2D
                
                legend_elements = [
                    Line2D([0], [0], marker='o', color='gray', linestyle='', 
                          markersize=10, label='NoStim'),
                    Line2D([0], [0], marker='*', color='gray', linestyle='', 
                          markersize=15, label='Stim')
                ]
            
            # Connect points from the same context with lines
            for ctx in range(3):
                if not (np.isnan(means_no_opto[ctx]) or np.isnan(means_opto[ctx])):
                    ax.plot([ctx, ctx + 0.25], [means_no_opto[ctx], means_opto[ctx]], 
                           color=context_colors[ctx], linestyle='-', alpha=0.5)
            
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        return fig, axes

    def plot_velocity_delta_by_context(self, all_trialized_data, all_task_dfs, figsize=(6, 2.5), dpi=800):
        """
        Calculate and plot the delta (change) in X and Y velocities between Opto and Non-opto trials
        across each context.
        
        Parameters:
        -----------
        all_trialized_data : dict
            Dictionary containing trialized data for each mouse
        all_task_dfs : dict
            Dictionary containing DataFrames with task data for each mouse
        figsize : tuple, optional
            Figure size (width, height) in inches
        dpi : int, optional
            Figure resolution
            
        Returns:
        --------
        fig, axes : tuple
            Figure and axes objects for the plot
        """
        from scipy.stats import sem
        
        # Define context colors and labels
        context_colors = {0: 'purple', 1: '#EC008C', 2: '#27AAE1'}
        context_labels = {0: 'Congruent', 1: 'Visual', 2: 'Audio'}
        
        # Collect velocity data by context and opto condition
        velocity_data = {
            'x': {(ctx, is_opto): [] for ctx in range(3) for is_opto in [0, 1]},
            'y': {(ctx, is_opto): [] for ctx in range(3) for is_opto in [0, 1]}
        }
        
        # Process data for each mouse
        for mouse_id, sessions in all_trialized_data.items():
            # Get corresponding task data
            task_df = all_task_dfs[mouse_id]
            
            # Process each session
            for session_idx, session_data in enumerate(sessions):
                # Get session date
                if 'date' in task_df.columns:
                    session_date = task_df['date'].unique()[session_idx] if session_idx < len(task_df['date'].unique()) else None
                    session_task_df = task_df[task_df['date'] == session_date] if session_date else None
                else:
                    # If no date column, assume sessions are in order
                    session_task_df = task_df
                
                if session_task_df is None or len(session_task_df) == 0:
                    continue
                    
                if 'X_velocity' in session_data and 'Y_velocity' in session_data:
                    x_vel = session_data['X_velocity']
                    y_vel = session_data['Y_velocity']
                    
                    # Process each trial
                    for trial_idx, trial_row in session_task_df.iterrows():
                        if trial_idx >= len(x_vel) or len(x_vel[trial_idx]) == 0:
                            continue
                            
                        # Get context and opto status
                        context = trial_row['context']
                        is_opto = trial_row['opto']
                        
                        # Process X velocity
                        x_data = np.array(x_vel[trial_idx][100:150])
                        x_data = x_data[~np.isnan(x_data)]
                        if len(x_data) > 0:
                            velocity_data['x'][(context, is_opto)].append(np.mean(np.abs(x_data)))
                        
                        # Process Y velocity
                        y_data = np.array(y_vel[trial_idx][:150])
                        y_data = y_data[~np.isnan(y_data)]
                        if len(y_data) > 0:
                            velocity_data['y'][(context, is_opto)].append(np.mean(np.abs(y_data)))
        
        # Calculate deltas for each context
        x_vel_deltas = {ctx: [] for ctx in range(3)}
        y_vel_deltas = {ctx: [] for ctx in range(3)}
        
        # For each mouse and session, calculate the delta (opto - non-opto)
        for mouse_id, sessions in all_trialized_data.items():
            for session_idx, _ in enumerate(sessions):
                for ctx in range(3):
                    # Get mean velocities for this mouse, session, and context
                    x_opto_vals = [v for i, v in enumerate(velocity_data['x'][(ctx, 1)]) 
                                if i % len(sessions) == session_idx]
                    x_non_opto_vals = [v for i, v in enumerate(velocity_data['x'][(ctx, 0)]) 
                                    if i % len(sessions) == session_idx]
                    
                    y_opto_vals = [v for i, v in enumerate(velocity_data['y'][(ctx, 1)]) 
                                if i % len(sessions) == session_idx]
                    y_non_opto_vals = [v for i, v in enumerate(velocity_data['y'][(ctx, 0)]) 
                                    if i % len(sessions) == session_idx]
                    
                    # Calculate mean for this session
                    if x_opto_vals and x_non_opto_vals:
                        x_delta = np.mean(x_opto_vals) - np.mean(x_non_opto_vals)
                        x_vel_deltas[ctx].append(x_delta)
                    
                    if y_opto_vals and y_non_opto_vals:
                        y_delta = np.mean(y_opto_vals) - np.mean(y_non_opto_vals)
                        y_vel_deltas[ctx].append(y_delta)
        
        # Calculate means and SEMs for each context
        x_means = [np.mean(x_vel_deltas[ctx]) if x_vel_deltas[ctx] else np.nan for ctx in range(3)]
        x_sems = [sem(x_vel_deltas[ctx]) if len(x_vel_deltas[ctx]) > 1 else np.nan for ctx in range(3)]
        
        y_means = [np.mean(y_vel_deltas[ctx]) if y_vel_deltas[ctx] else np.nan for ctx in range(3)]
        y_sems = [sem(y_vel_deltas[ctx]) if len(y_vel_deltas[ctx]) > 1 else np.nan for ctx in range(3)]
        
        # Create the plot
        fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
        
        # X-axis positions for bars
        x_pos = np.arange(3)
        
        # Plot individual session data points with jitter
        jitter_width = 0.2  # Width of jitter around bar position
        
        # Plot X velocity deltas
        axes[0].bar(x_pos, x_means, yerr=x_sems, color=[context_colors[ctx] for ctx in range(3)], 
                    alpha=0.7, capsize=5)
        # Add individual session points
        for ctx in range(3):
            if x_vel_deltas[ctx]:
                jitter = np.random.uniform(-jitter_width, jitter_width, size=len(x_vel_deltas[ctx]))
                axes[0].scatter(x_pos[ctx] + jitter, x_vel_deltas[ctx],
                              color=context_colors[ctx], s=20, alpha=0.3)
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels([context_labels[ctx] for ctx in range(3)])
        axes[0].set_ylabel('Δ X-Velocity')
        axes[0].set_title('Change in X Velocity', fontsize=10)
        axes[0].axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        
        # Plot Y velocity deltas
        axes[1].bar(x_pos, y_means, yerr=y_sems, color=[context_colors[ctx] for ctx in range(3)], 
                    alpha=0.7, capsize=5)
        # Add individual session points
        for ctx in range(3):
            if y_vel_deltas[ctx]:
                jitter = np.random.uniform(-jitter_width, jitter_width, size=len(y_vel_deltas[ctx]))
                axes[1].scatter(x_pos[ctx] + jitter, y_vel_deltas[ctx],
                              color=context_colors[ctx], s=20, alpha=0.3)
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels([context_labels[ctx] for ctx in range(3)])
        axes[1].set_ylabel('Δ Y-Velocity')
        axes[1].set_title('Change in Y Velocity', fontsize=10)
        axes[1].axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)

        # Remove top and right spines
        for ax in axes:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_ylim(-30, 30)
        
        plt.tight_layout()
        return fig, axes

class BehaviorPlotter:
    """
    A class for creating plots of behavioral data analysis results.
    """
    
    def __init__(self):
        """
        Initialize the BehaviorPlotter.
        """
        pass
        
    def plot_context_accuracy(self, all_task_dfs, figsize=(3.5, 2.5), dpi=800):
        """
        Create a scatter plot of accuracy by context with each point representing 
        an animal's mean performance across sessions (with SEM).
        
        Args:
            all_task_dfs (dict): Dictionary of task DataFrames keyed by mouse ID
            figsize (tuple): Figure size
            dpi (int): Figure resolution
            
        Returns:
            fig, ax: Figure and axis objects
        """
        # Set up the plot
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        # Define colors and context names
        context_colors = ['purple', '#EC008C', '#27AAE1']
        context_names = ['Congruent', 'Visual', 'Audio']
        
        # Calculate accuracy for each mouse and context, including SEM across sessions
        accuracies = {ctx: [] for ctx in range(3)}
        accuracy_sems = {ctx: [] for ctx in range(3)}
        mouse_labels = []
        
        for mouse, task_df in all_task_dfs.items():
            mouse_labels.append(mouse)
            # Group by date to get session-wise performances
            dates = task_df['date'].unique()
            
            for ctx in range(3):
                session_accs = []  # Store accuracies for each session
                for date in dates:
                    date_trials = task_df[(task_df['date'] == date) & 
                                        (task_df['context'] == ctx)]
                    if len(date_trials) > 0:
                        session_accs.append(date_trials['outcome'].mean() * 100)
                
                # Calculate mean and SEM across sessions
                mean_acc = np.mean(session_accs) if session_accs else np.nan
                sem_acc = np.std(session_accs) / np.sqrt(len(session_accs)) if len(session_accs) > 1 else 0
                
                accuracies[ctx].append(mean_acc)
                accuracy_sems[ctx].append(sem_acc)
        
        # Plot points for each context
        for ctx in range(3):
            # Add individual points with error bars
            for i, (acc, sem) in enumerate(zip(accuracies[ctx], accuracy_sems[ctx])):
                ax.scatter(ctx, acc,
                          color=context_colors[ctx],
                          s=100,
                          alpha=0.7)
                
                # Add error bars for each point
                ax.errorbar(ctx, acc, yerr=sem,
                           color=context_colors[ctx],
                           capsize=5,
                           capthick=1,
                           linewidth=1,
                           alpha=0.7)
        
        # Customize plot
        ax.set_xticks(range(3))
        ax.set_xticklabels(context_names)
        ax.set_ylabel('Accuracy (%)')
        
        # Add mouse labels next to points
        for ctx in range(3):
            for i, (acc, mouse) in enumerate(zip(accuracies[ctx], mouse_labels)):
                ax.annotate(mouse, 
                           (ctx, acc),
                           xytext=(5, 0),
                           textcoords='offset points',
                           fontsize=2,
                           alpha=0.7)
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Set y-axis limits and add chance line
        ax.set_ylim(40, 105)
        ax.set_xlim(-0.5, 2.5)
        ax.axhline(y=50, color='grey', linestyle='--', linewidth=1)
        
        plt.tight_layout()
        return fig, ax
    
    def plot_context_accuracy_comparison(self, all_task_dfs, figsize=(3.5, 2.5), dpi=800):
        """
        Create a scatter plot of accuracy by context, comparing opto vs non-opto trials.
        Includes error bars for each animal's performance across days and connecting lines.
        
        Args:
            all_task_dfs (dict): Dictionary of task DataFrames keyed by mouse ID
            figsize (tuple): Figure size
            dpi (int): Figure resolution
            
        Returns:
            fig, ax: Figure and axis objects
        """
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        # Define colors and context names
        context_colors = ['purple', '#EC008C', '#27AAE1']
        context_names = ['Congruent', 'Visual', 'Audio']
        
        # Calculate accuracy for each mouse, context, and opto condition
        accuracies = {(ctx, is_opto): [] for ctx in range(3) for is_opto in [0, 1]}
        accuracy_sems = {(ctx, is_opto): [] for ctx in range(3) for is_opto in [0, 1]}
        mouse_labels = []
        
        for mouse, task_df in all_task_dfs.items():
            mouse_labels.append(mouse)
            # Group by date to get daily performances
            dates = task_df['date'].unique()
            
            for ctx in range(3):
                for is_opto in [0, 1]:
                    daily_accs = []
                    for date in dates:
                        date_trials = task_df[(task_df['date'] == date) & 
                                            (task_df['context'] == ctx) & 
                                            (task_df['opto'] == is_opto)]
                        if len(date_trials) > 0:
                            daily_accs.append(date_trials['outcome'].mean() * 100)
                    
                    if daily_accs:  # If we have any data for this condition
                        mean_acc = np.mean(daily_accs)
                        sem_acc = np.std(daily_accs) / np.sqrt(len(daily_accs))
                        accuracies[(ctx, is_opto)].append(mean_acc)
                        accuracy_sems[(ctx, is_opto)].append(sem_acc)
                    else:
                        accuracies[(ctx, is_opto)].append(np.nan)
                        accuracy_sems[(ctx, is_opto)].append(np.nan)
        
        # Plot points for each context and opto condition
        bar_width = 1
        for ctx in range(3):
            x_pos = ctx * 2
            
            # Plot connecting lines between control and opto points
            for i in range(len(mouse_labels)):
                if not (np.isnan(accuracies[(ctx, 0)][i]) or np.isnan(accuracies[(ctx, 1)][i])):
                    ax.plot([x_pos, x_pos + bar_width],
                           [accuracies[(ctx, 0)][i], accuracies[(ctx, 1)][i]],
                           color=context_colors[ctx],
                           alpha=0.3,
                           linestyle='-')
            
            # Non-opto trials (circles)
            ax.scatter([x_pos] * len(accuracies[(ctx, 0)]), 
                      accuracies[(ctx, 0)],
                      color=context_colors[ctx],
                      s=100,
                      alpha=0.7,
                      marker='o')
            
            # Add error bars for individual points
            for i, (acc, sem) in enumerate(zip(accuracies[(ctx, 0)], accuracy_sems[(ctx, 0)])):
                if not np.isnan(acc):
                    ax.errorbar(x_pos, acc, yerr=sem,
                              color=context_colors[ctx],
                              alpha=0.7,
                              capsize=3,
                              capthick=1,
                              linewidth=1)
            
            # Opto trials (triangles)
            ax.scatter([x_pos + bar_width] * len(accuracies[(ctx, 1)]), 
                      accuracies[(ctx, 1)],
                      color=context_colors[ctx],
                      s=100,
                      alpha=0.5,
                      marker='*')
            
            # Add error bars for individual opto points
            for i, (acc, sem) in enumerate(zip(accuracies[(ctx, 1)], accuracy_sems[(ctx, 1)])):
                if not np.isnan(acc):
                    ax.errorbar(x_pos + bar_width, acc, yerr=sem,
                              color=context_colors[ctx],
                              alpha=0.5,
                              capsize=3,
                              capthick=1,
                              linewidth=1)
        
        # Add mouse labels
        for ctx in range(3):
            for is_opto in [0, 1]:
                x_pos = ctx * 2 + (bar_width if is_opto else 0)
                for i, (acc, mouse) in enumerate(zip(accuracies[(ctx, is_opto)], mouse_labels)):
                    if not np.isnan(acc):
                        ax.annotate(mouse, 
                                   (x_pos, acc),
                                   xytext=(5, 0),
                                   textcoords='offset points',
                                   fontsize=2,
                                   alpha=0.7 if not is_opto else 0.5)
        
        # Customize plot
        ax.set_xticks([i * 2 + bar_width/2 for i in range(3)])
        ax.set_xticklabels(context_names)
        ax.set_ylabel('Accuracy (%)')
        ax.set_ylim(40, 105)
        ax.set_xlim(-1, 6)
        ax.axhline(y=50, color='grey', linestyle='--', linewidth=1)
        for x in [1.5,3.5]:
            ax.axvline(x=x, color='grey', linestyle='--', linewidth=1)
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                   markersize=8, alpha=0.7, label='No Opto'),
            Line2D([0], [0], marker='*', color='w', markerfacecolor='gray', 
                   markersize=8, alpha=0.5, label='Opto')
        ]
        ax.legend(handles=legend_elements, loc='lower left', fontsize=5, frameon=False)
        
        plt.tight_layout()
        return fig, ax

    def plot_context_accuracy_comparison_every_session(self, all_task_dfs, figsize=(3.5, 2.5), dpi=800):
        """
        Create a scatter plot of accuracy by context, comparing opto vs non-opto trials.
        Each session is plotted as a separate point, with statistical comparison.
        
        Args:
            all_task_dfs (dict): Dictionary of task DataFrames keyed by mouse ID
            figsize (tuple): Figure size
            dpi (int): Figure resolution
            
        Returns:
            fig, ax: Figure and axis objects
        """
        
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        # Define colors and context names
        context_colors = ['purple', '#EC008C', '#27AAE1']
        context_names = ['Congruent', 'Visual', 'Audio']
        
        # Collect paired data for statistical testing
        paired_data = {ctx: {'control': [], 'opto': []} for ctx in range(3)}
        
        # Plot points for each context and opto condition
        bar_width = 1
        for ctx in range(3):
            x_pos = ctx * 2
            
            # For each mouse
            for mouse, task_df in all_task_dfs.items():
                # Get unique dates
                dates = task_df['date'].unique()
                
                # For each session date
                for date in dates:
                    date_df = task_df[task_df['date'] == date]
                    
                    # Get non-opto accuracy for this context and session
                    non_opto_trials = date_df[(date_df['context'] == ctx) & (date_df['opto'] == 0)]
                    if len(non_opto_trials) > 0:
                        non_opto_acc = non_opto_trials['outcome'].mean() * 100
                        
                        # Plot non-opto point (circle)
                        ax.scatter(x_pos, non_opto_acc,
                                  color=context_colors[ctx],
                                  s=80,
                                  alpha=0.7,
                                  marker='o')
                    
                    # Get opto accuracy for this context and session
                    opto_trials = date_df[(date_df['context'] == ctx) & (date_df['opto'] == 1)]
                    if len(opto_trials) > 0:
                        opto_acc = opto_trials['outcome'].mean() * 100
                        
                        # Plot opto point (triangle)
                        ax.scatter(x_pos + bar_width, opto_acc,
                                  color=context_colors[ctx],
                                  s=80,
                                  alpha=0.7,
                                  marker='*')
                        
                        # Draw connecting line if we have both opto and non-opto data
                        if len(non_opto_trials) > 0:
                            ax.plot([x_pos, x_pos + bar_width],
                                   [non_opto_acc, opto_acc],
                                   color=context_colors[ctx],
                                   alpha=0.3,
                                   linestyle='-')
                            
                            # Store paired data for statistical testing
                            paired_data[ctx]['control'].append(non_opto_acc)
                            paired_data[ctx]['opto'].append(opto_acc)
        
        # Add statistical comparison for each context
        for ctx in range(3):
            x_pos = ctx * 2
            control_data = paired_data[ctx]['control']
            opto_data = paired_data[ctx]['opto']
            
            if len(control_data) >= 5 and len(opto_data) >= 5:  # Ensure enough data points for statistical test
                # Perform Wilcoxon signed-rank test (paired test)
                stat, p_value = wilcoxon(control_data, opto_data, alternative='two-sided')
                
                # Get y positions for significance marker
                y_max = max(max(control_data), max(opto_data))
                
                # Add significance markers
                if p_value > 0.05:
                    text = "NS"
                elif p_value > 0.01:
                    text = "*"
                elif p_value > 0.001:
                    text = "**"
                else:
                    text = "***"
                
                # Add the significance marker and p-value
                y_text = 50 + 5  # Position the text above the highest point
                ax.text(x_pos + bar_width/2, y_text, text,
                       horizontalalignment='center',
                       verticalalignment='bottom',
                       color=context_colors[ctx],
                       fontsize=6)
                ax.text(x_pos + bar_width/2, y_text-3, f"p={p_value:.3f}",
                       horizontalalignment='center',
                       verticalalignment='bottom',
                       color=context_colors[ctx],
                       fontsize=4)
        
        # Customize plot
        ax.set_xticks([i * 2 + bar_width/2 for i in range(3)])
        ax.set_xticklabels(context_names)
        ax.set_ylabel('Accuracy (%)')
        ax.set_ylim(40, 105)
        ax.set_xlim(-1, 6)
        ax.axhline(y=50, color='grey', linestyle='--', linewidth=1)
        for x in [1.5,3.5]:
            ax.axvline(x=x, color='grey', linestyle='--', linewidth=1)
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                   markersize=8, alpha=0.7, label='No Opto'),
            Line2D([0], [0], marker='*', color='w', markerfacecolor='gray', 
                   markersize=8, alpha=0.5, label='Opto')
        ]
        ax.legend(handles=legend_elements, loc='lower left', fontsize=5, frameon=False)
        
        plt.tight_layout()
        return fig, ax

    def plot_context_accuracy_with_sessions(self, all_task_dfs, figsize=(3.5, 2.5), dpi=800):
        """
        Create a scatter plot of accuracy by context, comparing opto vs non-opto trials.
        Shows both aggregated statistics and individual session data points.
        
        Args:
            all_task_dfs (dict): Dictionary of task DataFrames keyed by mouse ID
            figsize (tuple): Figure size
            dpi (int): Figure resolution
            
        Returns:
            fig, ax: Figure and axis objects
        """
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        # Define colors and context names
        context_colors = ['purple', '#EC008C', '#27AAE1']
        context_names = ['Congruent', 'Visual', 'Audio']
        
        # Calculate accuracy for each session, context, and opto condition
        session_accuracies = {(ctx, is_opto): [] for ctx in range(3) for is_opto in [0, 1]}
        paired_accuracies = {ctx: {'control': [], 'opto': []} for ctx in range(3)}
        
        # Plot points for each context and opto condition
        bar_width = 1
        
        # First, plot individual session data points with low alpha
        for mouse, task_df in all_task_dfs.items():
            dates = task_df['date'].unique()
            
            for date in dates:
                date_df = task_df[task_df['date'] == date]
                
                for ctx in range(3):
                    x_pos = ctx * 2
                    
                    # Get control and opto trials for this session and context
                    control_trials = date_df[
                        (date_df['context'] == ctx) & 
                        (date_df['opto'] == 0)
                    ]
                    opto_trials = date_df[
                        (date_df['context'] == ctx) & 
                        (date_df['opto'] == 1)
                    ]
                    
                    # Plot individual session points
                    if len(control_trials) > 0:
                        control_acc = control_trials['outcome'].mean() * 100
                        ax.scatter(x_pos, control_acc,
                                  color=context_colors[ctx],
                                  s=20,  # Smaller points
                                  alpha=0.3,  # Low alpha
                                  marker='o')
                        session_accuracies[(ctx, 0)].append(control_acc)
                    
                    if len(opto_trials) > 0:
                        opto_acc = opto_trials['outcome'].mean() * 100
                        ax.scatter(x_pos + bar_width, opto_acc,
                                  color=context_colors[ctx],
                                  s=20,  # Smaller points
                                  alpha=0.3,  # Low alpha
                                  marker='*')
                        session_accuracies[(ctx, 1)].append(opto_acc)
                    
                    # Draw connecting line for individual session if we have both control and opto
                    if len(control_trials) > 0 and len(opto_trials) > 0:
                        ax.plot([x_pos, x_pos + bar_width],
                               [control_acc, opto_acc],
                               color=context_colors[ctx],
                               alpha=0.15,  # Very light lines
                               linewidth=0.5,
                               zorder=1)
                        
                        # Store paired data for statistical testing
                        paired_accuracies[ctx]['control'].append(control_acc)
                        paired_accuracies[ctx]['opto'].append(opto_acc)
        
        # Now plot the aggregated statistics with higher alpha and larger markers
        for ctx in range(3):
            x_pos = ctx * 2
            
            # Calculate mean and SEM for control condition
            if session_accuracies[(ctx, 0)]:
                control_mean = np.mean(session_accuracies[(ctx, 0)])
                control_sem = np.std(session_accuracies[(ctx, 0)]) / np.sqrt(len(session_accuracies[(ctx, 0)]))
                
                # Plot aggregated control point
                ax.scatter(x_pos, control_mean,
                          color=context_colors[ctx],
                          s=100,  # Larger point
                          alpha=0.8,  # Higher alpha
                          marker='o',
                          zorder=3)
                
                # Add error bars
                ax.errorbar(x_pos, control_mean, yerr=control_sem,
                           color=context_colors[ctx],
                           alpha=0.8,
                           capsize=5,
                           capthick=1,
                           linewidth=1,
                           zorder=2)
            
            # Calculate mean and SEM for opto condition
            if session_accuracies[(ctx, 1)]:
                opto_mean = np.mean(session_accuracies[(ctx, 1)])
                opto_sem = np.std(session_accuracies[(ctx, 1)]) / np.sqrt(len(session_accuracies[(ctx, 1)]))
                
                # Plot aggregated opto point
                ax.scatter(x_pos + bar_width, opto_mean,
                          color=context_colors[ctx],
                          s=100,  # Larger point
                          alpha=0.8,  # Higher alpha
                          marker='*',
                          zorder=3)
                
                # Add error bars
                ax.errorbar(x_pos + bar_width, opto_mean, yerr=opto_sem,
                           color=context_colors[ctx],
                           alpha=0.8,
                           capsize=5,
                           capthick=1,
                           linewidth=1,
                           zorder=2)
            
            # Add connecting line between aggregated points
            if session_accuracies[(ctx, 0)] and session_accuracies[(ctx, 1)]:
                ax.plot([x_pos, x_pos + bar_width],
                       [control_mean, opto_mean],
                       color=context_colors[ctx],
                       alpha=0.7,
                       linewidth=2,
                       zorder=2)
            
            # Add statistical comparison
            control_data = paired_accuracies[ctx]['control']
            opto_data = paired_accuracies[ctx]['opto']
            
            if len(control_data) >= 5 and len(opto_data) >= 5:
                # Perform Wilcoxon signed-rank test (paired test)
                stat, p_value = wilcoxon(control_data, opto_data, alternative='two-sided')
                
                # Add significance markers
                if p_value > 0.05:
                    text = "NS"
                elif p_value > 0.01:
                    text = "*"
                elif p_value > 0.001:
                    text = "**"
                else:
                    text = "***"
                
                # Add the significance marker and p-value
                y_text = 105  # Position the text above the highest point
                ax.text(x_pos + bar_width/2, y_text, text,
                       horizontalalignment='center',
                       verticalalignment='bottom',
                       color=context_colors[ctx],
                       fontsize=4)
                ax.text(x_pos + bar_width/2, y_text-2, f"p={p_value:.3f}",
                       horizontalalignment='center',
                       verticalalignment='bottom',
                       color=context_colors[ctx],
                       fontsize=4)
        
        # Customize plot
        ax.set_xticks([i * 2 + bar_width/2 for i in range(3)])
        ax.set_xticklabels(context_names)
        ax.set_ylabel('Accuracy (%)')
        ax.set_ylim(40, 105)
        ax.set_xlim(-1, 6)
        ax.axhline(y=50, color='grey', linestyle='--', linewidth=1)
        for x in [1.5, 3.5]:
            ax.axvline(x=x, color='grey', linestyle='--', linewidth=1)
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                   markersize=8, alpha=0.7, label='No Opto'),
            Line2D([0], [0], marker='*', color='w', markerfacecolor='gray', 
                   markersize=8, alpha=0.7, label='Opto')
        ]
        ax.legend(handles=legend_elements, loc='lower left', fontsize=5, frameon=False)
        
        plt.tight_layout()
        return fig, ax

    def plot_context_accuracy_combine_mice(self, all_task_dfs, figsize=(3.5, 2.5), dpi=800):
        """
        Create a scatter plot of accuracy by context, comparing opto vs non-opto trials,
        with data combined across sessions and statistical comparisons.
        
        Args:
            all_task_dfs (dict): Dictionary of task DataFrames keyed by mouse ID
            figsize (tuple): Figure size
            dpi (int): Figure resolution
            
        Returns:
            fig, ax: Figure and axis objects
        """
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        # Define colors and context names
        context_colors = ['purple', '#EC008C', '#27AAE1']
        context_names = ['Congruent', 'Visual', 'Audio']
        
        # Calculate accuracy for each session, context, and opto condition
        session_accuracies = {(ctx, is_opto): [] for ctx in range(3) for is_opto in [0, 1]}
        paired_accuracies = {ctx: {'control': [], 'opto': []} for ctx in range(3)}
        
        # Collect accuracies for each session
        for mouse, task_df in all_task_dfs.items():
            dates = task_df['date'].unique()
            
            for date in dates:
                date_df = task_df[task_df['date'] == date]
                
                for ctx in range(3):
                    # Get control and opto trials for this session and context
                    control_trials = date_df[
                        (date_df['context'] == ctx) & 
                        (date_df['opto'] == 0)
                    ]
                    opto_trials = date_df[
                        (date_df['context'] == ctx) & 
                        (date_df['opto'] == 1)
                    ]
                    
                    # Only include if we have both control and opto trials
                    if len(control_trials) > 0 and len(opto_trials) > 0:
                        control_acc = control_trials['outcome'].mean() * 100
                        opto_acc = opto_trials['outcome'].mean() * 100
                        
                        session_accuracies[(ctx, 0)].append(control_acc)
                        session_accuracies[(ctx, 1)].append(opto_acc)
                        
                        paired_accuracies[ctx]['control'].append(control_acc)
                        paired_accuracies[ctx]['opto'].append(opto_acc)
        
        # Plot points for each context and opto condition
        bar_width = 1
        for ctx in range(3):
            x_pos = ctx * 2
            
            # Calculate mean and SEM across sessions
            for is_opto in [0, 1]:
                acc_data = session_accuracies[(ctx, is_opto)]
                if acc_data:
                    mean_acc = np.mean(acc_data)
                    sem_acc = np.std(acc_data) / np.sqrt(len(acc_data))
                    
                    # Plot point
                    ax.scatter(x_pos + (bar_width if is_opto else 0), 
                              mean_acc,
                              color=context_colors[ctx],
                              s=100,
                              alpha=0.7 if not is_opto else 0.5,
                              marker='o' if not is_opto else '*')
                    
                    # Add error bars
                    ax.errorbar(x_pos + (bar_width if is_opto else 0), 
                              mean_acc, 
                              yerr=sem_acc,
                              color=context_colors[ctx],
                              alpha=0.7 if not is_opto else 0.5,
                              capsize=5,
                              capthick=1,
                              linewidth=1)
            
            # Add connecting lines between control and opto points
            if acc_data:  # if we have data for both conditions
                ax.plot([x_pos, x_pos + bar_width],
                       [np.mean(session_accuracies[(ctx, 0)]), 
                        np.mean(session_accuracies[(ctx, 1)])],
                       color=context_colors[ctx],
                       alpha=0.3,
                       linestyle='-')
            
            # Add statistical comparison
            control_data = paired_accuracies[ctx]['control']
            opto_data = paired_accuracies[ctx]['opto']
            
            if len(control_data) > 10 and len(opto_data) > 10:
                stat, p_value = wilcoxon(control_data, opto_data, alternative='two-sided')
                
                # Get y positions for significance marker
                y_control = np.mean(control_data)
                y_opto = np.mean(opto_data)
                y_max = max(y_control, y_opto)
                
                # Add significance markers
                if p_value > 0.05:
                    text = "NS"
                elif p_value > 0.01:
                    text = "*"
                elif p_value > 0.001:
                    text = "**"
                else:
                    text = "***"
                
                # Add the significance marker and p-value
                y_text = y_max + 5  # Adjust this value to position the text higher/lower
                ax.text(x_pos + bar_width/2, y_text, text,
                       horizontalalignment='center',
                       verticalalignment='bottom',
                       color=context_colors[ctx],
                       fontsize=6)
                ax.text(x_pos + bar_width/2, y_text-2, f"p={p_value:.3f}",
                       horizontalalignment='center',
                       verticalalignment='bottom',
                       color=context_colors[ctx],
                       fontsize=4)
        
        # Customize plot
        ax.set_xticks([i * 2 + bar_width/2 for i in range(3)])
        ax.set_xticklabels(context_names)
        ax.set_ylabel('Accuracy (%)')
        ax.set_ylim(40, 105)
        ax.set_xlim(-1, 6)
        ax.axhline(y=50, color='grey', linestyle='--', linewidth=1)
        for x in [1.5,3.5]:
            ax.axvline(x=x, color='grey', linestyle='--', linewidth=1)
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                   markersize=8, alpha=0.7, label='No Opto'),
            Line2D([0], [0], marker='*', color='w', markerfacecolor='gray', 
                   markersize=8, alpha=0.5, label='Opto')
        ]
        ax.legend(handles=legend_elements, loc='lower left', fontsize=5, frameon=False)
        
        plt.tight_layout()
        return fig, ax

    def plot_opto_and_post_opto_effects(self, all_task_dfs, figsize=(6, 2.5), dpi=800):
        """
        Create side-by-side plots showing both:
        1. Direct effect: How opto affects performance on the current trial
        2. Post effect: How opto affects performance on the next non-opto trial
        
        Args:
            all_task_dfs (dict): Dictionary of task DataFrames keyed by mouse ID
            figsize (tuple): Figure size
            dpi (int): Figure resolution
        
        Returns:
            fig, axes: Figure and axes objects
        """
        # Create a figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
        
        # Plot direct opto effect
        from scipy.stats import sem, ttest_1samp
        
        # Define colors and context names
        context_colors = ['purple', '#EC008C', '#27AAE1']
        context_names = ['Congruent', 'Visual', 'Audio']
        
        # For the first subplot: Direct opto effect
        ax = axes[0]
        
        # Calculate delta for each mouse, session, and context
        direct_deltas = {ctx: [] for ctx in range(3)}
        
        # For each mouse
        for mouse, task_df in all_task_dfs.items():
            dates = task_df['date'].unique()
            
            for date in dates:
                date_df = task_df[task_df['date'] == date]
                
                for ctx in range(3):
                    non_opto_trials = date_df[(date_df['context'] == ctx) & (date_df['opto'] == 0)]
                    opto_trials = date_df[(date_df['context'] == ctx) & (date_df['opto'] == 1)]
                    
                    if len(non_opto_trials) > 0 and len(opto_trials) > 0:
                        non_opto_acc = non_opto_trials['outcome'].mean() * 100
                        opto_acc = opto_trials['outcome'].mean() * 100
                        
                        delta = opto_acc - non_opto_acc
                        direct_deltas[ctx].append(delta)
        
        # Calculate mean and SEM for each context
        mean_direct_deltas = [np.mean(direct_deltas[ctx]) if direct_deltas[ctx] else 0 for ctx in range(3)]
        sem_direct_deltas = [sem(direct_deltas[ctx]) if len(direct_deltas[ctx]) > 1 else 0 for ctx in range(3)]
        
        # Create bar plot for direct effect
        bars = ax.bar(range(3), mean_direct_deltas, color=context_colors, alpha=0.7)
        
        # Add error bars
        ax.errorbar(range(3), mean_direct_deltas, yerr=sem_direct_deltas, fmt='none', color='black', capsize=5)
        
        # Add individual session points with jitter
        jitter_width = 0.2
        for ctx in range(3):
            if direct_deltas[ctx]:
                jitter = np.random.uniform(-jitter_width, jitter_width, size=len(direct_deltas[ctx]))
                ax.scatter(np.full(len(direct_deltas[ctx]), ctx) + jitter, direct_deltas[ctx],
                          color=context_colors[ctx], s=20, alpha=0.3)
        
        # Add statistical significance markers
        for ctx in range(3):
            if direct_deltas[ctx]:
                t_stat, p_value = ttest_1samp(direct_deltas[ctx], 0)
                
                if p_value > 0.05:
                    text = "NS"
                elif p_value > 0.01:
                    text = "*"
                elif p_value > 0.001:
                    text = "**"
                else:
                    text = "***"
                
                height = mean_direct_deltas[ctx]
                y_pos = height + sem_direct_deltas[ctx] + 2 if height >= 0 else height - sem_direct_deltas[ctx] - 5
                
                #ax.text(ctx, y_pos, text,
                #       horizontalalignment='center',
                #       verticalalignment='center',
                #       color='black',
                #       fontsize=6)
        
        # Customize plot
        ax.set_xticks(range(3))
        ax.set_xticklabels(context_names)
        ax.set_ylabel('Δ Accuracy (%)')
        ax.set_title('LED on', fontsize=10)
        ax.axhline(y=0, color='grey', linestyle='--', linewidth=1)
        
        if any(direct_deltas.values()):
            max_abs_delta = max(abs(np.array(mean_direct_deltas) + np.array(sem_direct_deltas)) + 5)
            ax.set_ylim(-max_abs_delta, max_abs_delta)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # For the second subplot: Post-opto effect
        ax = axes[1]
        
        # Calculate delta for each mouse, session, and context
        post_deltas = {ctx: [] for ctx in range(3)}
        
        # For each mouse
        for mouse, task_df in all_task_dfs.items():
            dates = task_df['date'].unique()
            
            for date in dates:
                date_df = task_df[task_df['date'] == date].copy()
                
                # Reset index to make sure we have sequential indices
                date_df = date_df.reset_index(drop=True)
                
                # For each context
                for ctx in range(3):
                    # Create lists to store trials that follow opto and non-opto trials
                    post_opto_trials = []
                    post_non_opto_trials = []
                    
                    # Iterate through trials (except the last one)
                    for i in range(len(date_df) - 1):
                        current_trial = date_df.iloc[i]
                        next_trial = date_df.iloc[i + 1]
                        
                        # Check if both trials are in the same context
                        if current_trial['context'] == ctx and next_trial['context'] == ctx:
                            # Check if next trial is non-opto
                            if next_trial['opto'] == 0:
                                # Categorize based on current trial's opto status
                                if current_trial['opto'] == 1:
                                    post_opto_trials.append(next_trial)
                                elif current_trial['opto'] == 0:
                                    post_non_opto_trials.append(next_trial)
                    
                    # Convert lists to DataFrames
                    if post_opto_trials and post_non_opto_trials:
                        post_opto_df = pd.DataFrame(post_opto_trials)
                        post_non_opto_df = pd.DataFrame(post_non_opto_trials)
                        
                        # Calculate accuracy if we have enough trials
                        if len(post_opto_df) >= 5 and len(post_non_opto_df) >= 5:
                            post_opto_acc = post_opto_df['outcome'].mean() * 100
                            post_non_opto_acc = post_non_opto_df['outcome'].mean() * 100
                            
                            delta = post_opto_acc - post_non_opto_acc
                            post_deltas[ctx].append(delta)
        
        # Calculate mean and SEM for each context
        mean_post_deltas = [np.mean(post_deltas[ctx]) if post_deltas[ctx] else 0 for ctx in range(3)]
        sem_post_deltas = [sem(post_deltas[ctx]) if len(post_deltas[ctx]) > 1 else 0 for ctx in range(3)]
        
        # Create bar plot for post effect
        bars = ax.bar(range(3), mean_post_deltas, color=context_colors, alpha=0.7)
        
        # Add error bars
        ax.errorbar(range(3), mean_post_deltas, yerr=sem_post_deltas, fmt='none', color='black', capsize=5)
        
        # Add individual session points with jitter
        for ctx in range(3):
            if post_deltas[ctx]:
                jitter = np.random.uniform(-jitter_width, jitter_width, size=len(post_deltas[ctx]))
                ax.scatter(np.full(len(post_deltas[ctx]), ctx) + jitter, post_deltas[ctx],
                          color=context_colors[ctx], s=20, alpha=0.3)
        
        # Add statistical significance markers
        for ctx in range(3):
            if post_deltas[ctx]:
                t_stat, p_value = ttest_1samp(post_deltas[ctx], 0)
                
                if p_value > 0.05:
                    text = "NS"
                elif p_value > 0.01:
                    text = "*"
                elif p_value > 0.001:
                    text = "**"
                else:
                    text = "***"
                
                height = mean_post_deltas[ctx]
                y_pos = height + sem_post_deltas[ctx] + 2 if height >= 0 else height - sem_post_deltas[ctx] - 5
                
                #ax.text(ctx, y_pos, text,
                #       horizontalalignment='center',
                #       verticalalignment='center',
                #       color='black',
                #       fontsize=6)
        
        # Customize plot
        ax.set_xticks(range(3))
        ax.set_xticklabels(context_names)
        ax.set_ylabel('Δ Accuracy (%)')
        ax.set_title('Trial t+1 after LED on', fontsize=10)
        ax.axhline(y=0, color='grey', linestyle='--', linewidth=1)
        
        if any(post_deltas.values()):
            max_abs_delta = max(abs(np.array(mean_post_deltas) + np.array(sem_post_deltas)) + 5)
            ax.set_ylim(-max_abs_delta, max_abs_delta)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Ensure both plots have the same y-axis limits for fair comparison
        if any(direct_deltas.values()) and any(post_deltas.values()):
            max_direct = max(abs(np.array(mean_direct_deltas) + np.array(sem_direct_deltas)))
            max_post = max(abs(np.array(mean_post_deltas) + np.array(sem_post_deltas)))
            max_limit = max(max_direct, max_post) + 5
            
            axes[0].set_ylim(-40, 30)
            axes[1].set_ylim(-40, 30)
        
        plt.tight_layout()
        return fig, axes

    def plot_session_performance(self, all_task_dfs, mouse_id, session_date, window_size=30):
        """
        Plot rolling accuracy and context switches for a specific session.
        Highlights switches between context 1 and 2 specifically for 2loc task type.
        For 2loc tasks, congruent context transitions are not shown.
        
        Args:
            all_task_dfs (dict): Dictionary containing all task DataFrames
            mouse_id (str): Mouse identifier
            session_date (str): Session date
            window_size (int): Size of rolling window for accuracy calculation
            
        Returns:
            fig, ax: Figure and axis objects
        """
        # Get session data
        session_df = all_task_dfs[mouse_id][all_task_dfs[mouse_id]['date'] == session_date].copy()
        session_df = session_df.reset_index(drop=True)
        if len(session_df) == 0:
            print(f"No data found for mouse {mouse_id} on date {session_date}")
            return None, None
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 3), dpi=800)
        
        # Define context colors
        context_colors = {0: 'purple', 1: '#EC008C', 2: '#27AAE1'}
        context_names = {0: 'CG', 1: 'V', 2: 'A'}
        
        # Get task type
        task_type = session_df['task_type'].iloc[0] if 'task_type' in session_df.columns else None
        
        # Find context switches
        context_changes = session_df.index[session_df['context'].diff() != 0]
        context_changes = np.append(context_changes, len(session_df))  # Add end of session
        
        # Color regions between context switches
        for i in range(len(context_changes)):
            start = 0 if i == 0 else context_changes[i-1]
            end = context_changes[i]
            context = session_df.loc[start, 'context']
            ax.axvspan(start, end, 
                      color=context_colors[context], 
                      alpha=0.1)
        
        # Calculate and plot rolling accuracy
        rolling_acc = session_df['outcome'].rolling(window=window_size, center=True).mean()
        ax.plot(range(len(rolling_acc)), rolling_acc*100, 'k-', linewidth=1.5, zorder=2)
        
        # Add vertical lines and context labels at switches
        for i in range(len(context_changes)-1):  # Skip the last one (end of session)
            idx = context_changes[i]
            new_context = session_df.loc[idx, 'context']
            
            # Get previous context (for determining switch type)
            prev_idx = 0 if i == 0 else context_changes[i-1]
            prev_context = session_df.loc[prev_idx, 'context']
            
            # For 2loc task, skip lines and labels for transitions involving context 0 (CG)
            if task_type == '2loc' and (new_context == 0 or prev_context == 0):
                continue
            
            # Determine if this is a 1-to-2 or 2-to-1 switch for 2loc task
            is_special_switch = (task_type == '2loc' and 
                                ((prev_context == 1 and new_context == 2) or 
                                 (prev_context == 2 and new_context == 1)))
            
            # Use thicker, more prominent line for special switches
            if is_special_switch:
                ax.axvline(x=idx, color='red', linestyle='-', alpha=0.7, linewidth=2, zorder=3)
            else:
                ax.axvline(x=idx, color='gray', linestyle='--', alpha=0.5, zorder=1)
            
            # Add context label
            ax.text(idx, 1.05, context_names[new_context], 
                    transform=ax.get_xaxis_transform(), 
                    ha='center', va='bottom')
        
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, zorder=0)
        
        # Customize plot
        ax.set_xlabel('Trial Number')
        ax.set_ylabel('Accuracy (Rolling Window)')
        ax.set_ylim(0, 110)  # Leave room for context labels
        
        plt.tight_layout()
        return fig, ax

    def plot_context_accuracy_by_power(self, all_task_dfs, power_list, figsize=(3.5, 2.5), dpi=800):
        """
        Create a scatter plot of accuracy by context, comparing opto vs non-opto trials,
        with opto points shaded based on power levels.
        
        Args:
            all_task_dfs (dict): Dictionary of task DataFrames keyed by mouse ID
            power_list (dict): Dictionary mapping mouse IDs to lists of power levels
            figsize (tuple): Figure size
            dpi (int): Figure resolution
            
        Returns:
            fig, ax: Figure and axis objects
        """
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        # Define context colors and names
        context_colors = ['purple', '#EC008C', '#27AAE1']
        context_names = ['Congruent', 'Visual', 'Audio']
        
        # Bar width for plotting
        bar_width = 1
        
        # Dictionary to store paired data for statistical testing
        paired_data = {ctx: {'control': [], 'opto': []} for ctx in range(3)}
        
        # Dictionary to store power levels for each data point
        power_data = {ctx: {'x': [], 'y': [], 'power': []} for ctx in range(3)}
        
        # Find min and max power across all mice for normalization
        all_powers = []
        for mouse in power_list:
            all_powers.extend(power_list[mouse])
        min_power = min(all_powers) if all_powers else 0
        max_power = max(all_powers) if all_powers else 1
        power_range = max_power - min_power if max_power > min_power else 1
        
        # Process each mouse
        for mouse_idx, mouse in enumerate(all_task_dfs.keys()):
            # Get all sessions for this mouse
            mouse_df = all_task_dfs[mouse]
            
            # Group by date and context
            dates = mouse_df['date'].unique()
            
            # Process each date
            for date_idx, date in enumerate(dates):
                # Get data for this date
                date_df = mouse_df[mouse_df['date'] == date]
                
                # Get power level for this session
                if mouse in power_list and date_idx < len(power_list[mouse]):
                    power = power_list[mouse][date_idx]
                    # Normalize power to 0.3-1.0 range for alpha
                    normalized_power = 0.3 + 0.7 * ((power - min_power) / power_range)
                else:
                    power = None
                    normalized_power = 0.7  # Default alpha if power not specified
                
                # Process each context
                for ctx in range(3):
                    # Get trials for this context
                    ctx_trials = date_df[date_df['context'] == ctx]
                    
                    if len(ctx_trials) > 0:
                        # Calculate x position for this context
                        x_pos = ctx * 2
                        
                        # Split into opto and non-opto trials
                        non_opto_trials = ctx_trials[ctx_trials['opto'] == 0]
                        opto_trials = ctx_trials[ctx_trials['opto'] == 1]
                        
                        # Plot non-opto point (circle)
                        if len(non_opto_trials) > 0:
                            non_opto_acc = non_opto_trials['outcome'].mean() * 100
                            ax.scatter(x_pos, non_opto_acc,
                                    color=context_colors[ctx],
                                    s=80,
                                    alpha=0.7,
                                    marker='o')
                        
                        # Plot opto point with alpha based on power
                        if len(opto_trials) > 0:
                            opto_acc = opto_trials['outcome'].mean() * 100
                            
                            # Store power data for colorbar
                            power_data[ctx]['x'].append(x_pos + bar_width)
                            power_data[ctx]['y'].append(opto_acc)
                            power_data[ctx]['power'].append(power)
                            
                            # Plot opto point (star)
                            ax.scatter(x_pos + bar_width, opto_acc,
                                    color=context_colors[ctx],
                                    s=80,
                                    alpha=normalized_power,
                                    marker='*')
                            
                            # Add power label
                            if power is not None:
                                ax.annotate(f"{power:.1f}",
                                        (x_pos + bar_width, opto_acc),
                                        xytext=(5, 0),
                                        textcoords='offset points',
                                        fontsize=3,
                                        alpha=normalized_power)
                            
                            # Draw connecting line if we have both opto and non-opto data
                            if len(non_opto_trials) > 0:
                                ax.plot([x_pos, x_pos + bar_width],
                                    [non_opto_acc, opto_acc],
                                    color=context_colors[ctx],
                                    alpha=0.3,
                                    linestyle='-')
                                
                                # Store paired data for statistical testing
                                paired_data[ctx]['control'].append(non_opto_acc)
                                paired_data[ctx]['opto'].append(opto_acc)
        
        # Add statistical comparison for each context
        for ctx in range(3):
            x_pos = ctx * 2
            control_data = paired_data[ctx]['control']
            opto_data = paired_data[ctx]['opto']
            
            if len(control_data) >= 5 and len(opto_data) >= 5:  # Ensure enough data points for statistical test
                # Perform Wilcoxon signed-rank test (paired test)
                stat, p_value = wilcoxon(control_data, opto_data, alternative='two-sided')
                
                # Add significance markers
                if p_value > 0.05:
                    text = "NS"
                elif p_value > 0.01:
                    text = "*"
                elif p_value > 0.001:
                    text = "**"
                else:
                    text = "***"
                
                # Add the significance marker and p-value
                y_text = 50 + 5  # Position the text above the highest point
                ax.text(x_pos + bar_width/2, y_text, text,
                    horizontalalignment='center',
                    verticalalignment='bottom',
                    color=context_colors[ctx],
                    fontsize=6)
                ax.text(x_pos + bar_width/2, y_text-3, f"p={p_value:.3f}",
                    horizontalalignment='center',
                    verticalalignment='bottom',
                    color=context_colors[ctx],
                    fontsize=4)
        
        # Customize plot
        ax.set_xticks([i * 2 + bar_width/2 for i in range(3)])
        ax.set_xticklabels(context_names)
        ax.set_ylabel('Accuracy (%)')
        ax.set_ylim(40, 105)
        ax.set_xlim(-1, 6)
        ax.axhline(y=50, color='grey', linestyle='--', linewidth=1)
        for x in [1.5, 3.5]:
            ax.axvline(x=x, color='grey', linestyle='--', linewidth=1)
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                markersize=8, alpha=0.7, label='No Opto'),
            Line2D([0], [0], marker='*', color='w', markerfacecolor='gray', 
                markersize=8, alpha=0.7, label=f'Opto (Power: {min_power:.1f}-{max_power:.1f})')
        ]
        ax.legend(handles=legend_elements, loc='lower left', fontsize=5, frameon=False)
        
        # Add title with power range
        plt.title(f"Power Range: {min_power:.1f}-{max_power:.1f} mW", fontsize=8)
        
        plt.tight_layout()
        return fig, ax

class GLMAnalyzer:
    """
    A class for analyzing behavioral data using Generalized Linear Models (GLMs).
    """
    
    def __init__(self):
        """
        Initialize the GLMAnalyzer.
        """
        pass
        
    def fit_choice_glm_by_context(self, task_df, context):
        """
        Fit a GLM to predict right choices for a specific context.
        
        Args:
            task_df (pd.DataFrame): DataFrame containing trial data
            context (int): Context to analyze (0=congruent, 1=visual, 2=audio)
            
        Returns:
            model: Fitted statsmodels GLM object
            X: Feature matrix
            y: Target vector
        """
        from statsmodels.discrete.discrete_model import Logit
        from scipy.stats import zscore
        
        # Filter for specific context
        context_df = task_df[task_df['context'] == context].copy()
        
        print(f"\nDiagnostics for context {context}:")
        print(f"Number of trials: {len(context_df)}")
        
        # Create lagged variables for previous choice and outcome
        context_df['prev_choice'] = context_df['choice'].shift(1)
        context_df['prev_outcome'] = context_df['outcome'].shift(1)
        
        # Drop trials without previous trial info
        context_df = context_df.dropna(subset=['prev_choice', 'prev_outcome'])  # Only drop if history is NaN
        print(f"Number of trials after dropping NaN: {len(context_df)}")
        
        # Base features used in all contexts
        feature_dict = {
            'prev_choice_right': zscore((context_df['prev_choice'].values == 1).astype(float)),
            'prev_correct': zscore(context_df['prev_outcome'].values.astype(float)),
            'opto': zscore(context_df['opto'].values.astype(float)),
            'intercept': np.ones(len(context_df))
        }
        
        # Add context-specific stimulus features (binarized)
        if context == 0:  # Congruent
            feature_dict.update({
                'visual_right': (context_df['visual_stim'].values == 90).astype(float),
                'audio_right': (context_df['audio_stim'].values > 0).astype(float)
            })
        elif context == 1:  # Visual only
            feature_dict['visual_right'] = (context_df['visual_stim'].values == 90).astype(float)
            print("\nVisual stimulus distribution:")
            print(context_df['visual_stim'].value_counts())
        elif context == 2:  # Audio only
            feature_dict['audio_right'] = (context_df['audio_stim'].values > 0).astype(float)
            print("\nAudio stimulus distribution:")
            print(context_df['audio_stim'].value_counts())
        
        # Create feature matrix
        X = pd.DataFrame(feature_dict)
        
        # Target variable (1 = right choice)
        y = (context_df['choice'].values == 1).astype(float)
        
        print("\nFeature matrix info:")
        print(X.describe())
        print("\nChoice distribution:")
        print(pd.Series(y).value_counts())
        
        # Fit model
        model = Logit(y, X)
        results = model.fit(disp=0)
        
        return results, X, y
    
    def analyze_all_mice_by_context(self, all_task_dfs):
        """
        Fit GLM for each mouse and context combination.
        
        Args:
            all_task_dfs (dict): Dictionary of task DataFrames keyed by mouse ID
            
        Returns:
            results_dict: Dictionary of model results by context and mouse
        """
        context_names = ['Congruent', 'Visual', 'Audio']
        results_dict = {context: {} for context in range(3)}
        
        for mouse, task_df in all_task_dfs.items():
            print(f"\nResults for mouse {mouse}:")
            print("=" * 50)
            
            # Print trial counts first
            print("\nTrial counts by context:")
            print(task_df['context'].value_counts())
            
            for context in range(3):
                print(f"\nContext: {context_names[context]}")
                print("-" * 30)
                
                try:
                    model_results, X, y = self.fit_choice_glm_by_context(task_df, context)
                    results_dict[context][mouse] = model_results
                    
                    # Print summary statistics
                    print("\nCoefficient estimates and p-values:")
                    coef_df = pd.DataFrame({
                        'Coefficient': model_results.params,
                        'Std Error': model_results.bse,
                        'p-value': model_results.pvalues
                    })
                    print(coef_df)
                    
                    # Print model performance metrics
                    predictions = (model_results.predict(X) > 0.5).astype(float)
                    accuracy = np.mean(predictions == y)
                    print(f"\nModel accuracy: {accuracy:.3f}")
                    print(f"Pseudo R-squared: {model_results.prsquared:.3f}")
                    print(f"Number of trials: {len(y)}")
                    
                except Exception as e:
                    print(f"Could not fit model: {str(e)}")
                    continue
        
        return results_dict
    
    def plot_coefficients_by_context(self, results_dict, figsize=(10, 4)):
        """
        Plot coefficient values as scatter points with error bars for each context.
        
        Args:
            results_dict (dict): Dictionary of model results by context and mouse
            figsize (tuple): Figure size
            
        Returns:
            fig, axes: Figure and axes objects
        """
        context_names = ['Congruent', 'Visual', 'Audio']
        context_colors = ['purple', '#EC008C', '#27AAE1']  # Colors for each context
        fig, axes = plt.subplots(1, 3, figsize=figsize, dpi=800)
        
        for context_idx, (context_results, ax) in enumerate(zip(results_dict.values(), axes)):
            if not context_results:  # Skip if no results for this context
                continue
                
            # Get coefficient names from first mouse
            first_mouse = list(context_results.keys())[0]
            all_coef_names = context_results[first_mouse].params.index
            
            # Define desired order and filter out intercept
            if context_idx == 0:  # Congruent
                coef_order = ['visual_right', 'audio_right', 'prev_choice_right', 'prev_correct', 'opto']
            elif context_idx == 1:  # Visual
                coef_order = ['visual_right', 'prev_choice_right', 'prev_correct', 'opto']
            else:  # Audio
                coef_order = ['audio_right', 'prev_choice_right', 'prev_correct', 'opto']
                
            # Calculate mean and SEM across mice for each coefficient
            coef_means = []
            coef_sems = []
            
            for coef in coef_order:
                values = [results.params[coef] for results in context_results.values()]
                coef_means.append(np.mean(values))
                coef_sems.append(np.std(values) / np.sqrt(len(values)))
            
            # Plot points and error bars
            x = np.arange(len(coef_order))
            ax.scatter(x, coef_means, 
                      color=context_colors[context_idx],
                      s=100,
                      zorder=2)
            ax.errorbar(x, coef_means, yerr=coef_sems,
                       color=context_colors[context_idx],
                       fmt='none',
                       capsize=5,
                       capthick=2,
                       zorder=1)
            
            # Customize plot
            ax.set_xticks(x)
            ax.set_xticklabels([name.replace('_', ' ') for name in coef_order], 
                              rotation=45, 
                              ha='right')
            ax.set_ylabel('Coefficient Value')
            ax.set_title(f'{context_names[context_idx]} Context')
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, zorder=0)
            
            # Set y-axis limits consistently across plots
            ax.set_ylim(-4, 4.5)
        
        plt.tight_layout()
        return fig, axes

    def fit_choice_glm_by_context_no_opto(self, task_df, context):
        """
        Fit a GLM to predict right choices for a specific context.
        
        Args:
            task_df (pd.DataFrame): DataFrame containing trial data
            context (int): Context to analyze (0=congruent, 1=visual, 2=audio)
        
        Returns:
            model: Fitted statsmodels GLM object
            X: Feature matrix
            y: Target vector
        """
        from statsmodels.discrete.discrete_model import Logit
        from scipy.stats import zscore
        
        # Filter for specific context
        context_df = task_df[task_df['context'] == context].copy()
        
        print(f"\nDiagnostics for context {context}:")
        print(f"Number of trials: {len(context_df)}")
        
        # Create lagged variables for previous choice and outcome
        context_df['prev_choice'] = context_df['choice'].shift(1)
        context_df['prev_outcome'] = context_df['outcome'].shift(1)
        
        # Drop trials without previous trial info
        context_df = context_df.dropna(subset=['prev_choice', 'prev_outcome'])  # Only drop if history is NaN
        print(f"Number of trials after dropping NaN: {len(context_df)}")
        
        # Base features used in all contexts
        feature_dict = {
            'prev_choice_right': zscore((context_df['prev_choice'].values == 1).astype(float)),
            'prev_correct': zscore(context_df['prev_outcome'].values.astype(float)),
            'intercept': np.ones(len(context_df))
        }
        
        # Add context-specific stimulus features (binarized)
        if context == 0:  # Congruent
            # Use only one stimulus feature since they're perfectly correlated
            feature_dict['stimulus_right'] = (context_df['visual_stim'].values == 90).astype(float)
        elif context == 1:  # Visual only
            feature_dict['visual_right'] = (context_df['visual_stim'].values == 90).astype(float)
            print("\nVisual stimulus distribution:")
            print(context_df['visual_stim'].value_counts())
        elif context == 2:  # Audio only
            feature_dict['audio_right'] = (context_df['audio_stim'].values > 0).astype(float)
            print("\nAudio stimulus distribution:")
            print(context_df['audio_stim'].value_counts())
        
        # Create feature matrix
        X = pd.DataFrame(feature_dict)
        
        # Target variable (1 = right choice)
        y = (context_df['choice'].values == 1).astype(float)
        
        print("\nFeature matrix info:")
        print(X.describe())
        print("\nChoice distribution:")
        print(pd.Series(y).value_counts())
        
        # Fit model
        model = Logit(y, X)
        results = model.fit(disp=0)
        
        return results, X, y
    
    def analyze_all_mice_by_context_no_opto(self, all_task_dfs):
        """
        Fit GLM for each mouse and context combination.
        
        Args:
            all_task_dfs (dict): Dictionary of task DataFrames keyed by mouse ID
            
        Returns:
            results_dict: Dictionary of model results by context and mouse
        """
        context_names = ['Congruent', 'Visual', 'Audio']
        results_dict = {context: {} for context in range(3)}
        
        for mouse, task_df in all_task_dfs.items():
            print(f"\nResults for mouse {mouse}:")
            print("=" * 50)
            
            # Print trial counts first
            print("\nTrial counts by context:")
            print(task_df['context'].value_counts())
            
            for context in range(3):
                print(f"\nContext: {context_names[context]}")
                print("-" * 30)
                
                try:
                    model_results, X, y = self.fit_choice_glm_by_context_no_opto(task_df, context)
                    results_dict[context][mouse] = model_results
                    
                    # Print summary statistics
                    print("\nCoefficient estimates and p-values:")
                    coef_df = pd.DataFrame({
                        'Coefficient': model_results.params,
                        'Std Error': model_results.bse,
                        'p-value': model_results.pvalues
                    })
                    print(coef_df)
                    
                    # Print model performance metrics
                    predictions = (model_results.predict(X) > 0.5).astype(float)
                    accuracy = np.mean(predictions == y)
                    print(f"\nModel accuracy: {accuracy:.3f}")
                    print(f"Pseudo R-squared: {model_results.prsquared:.3f}")
                    print(f"Number of trials: {len(y)}")
                    
                except Exception as e:
                    print(f"Could not fit model: {str(e)}")
                    continue
        
        return results_dict

    def plot_coefficients_by_context_no_opto(self, results_dict, figsize=(9, 4)):
        """
        Plot coefficient values as scatter points with error bars for each context.
        
        Args:
            results_dict (dict): Dictionary of model results by context and mouse
            figsize (tuple): Figure size
            
        Returns:
            fig, axes: Figure and axes objects
        """
        context_names = ['Congruent', 'Visual', 'Audio']
        context_colors = ['purple', '#EC008C', '#27AAE1']  # Colors for each context
        fig, axes = plt.subplots(1, 3, figsize=figsize, dpi=800)
        
        for context_idx, (context_results, ax) in enumerate(zip(results_dict.values(), axes)):
            if not context_results:  # Skip if no results for this context
                continue
                
            # Get coefficient names from first mouse
            first_mouse = list(context_results.keys())[0]
            all_coef_names = context_results[first_mouse].params.index
            
            # Define desired order and filter out intercept
            if context_idx == 0:  # Congruent
                coef_order = ['stimulus_right', 'prev_choice_right', 'prev_correct']
            elif context_idx == 1:  # Visual
                coef_order = ['visual_right', 'prev_choice_right', 'prev_correct']
            else:  # Audio
                coef_order = ['audio_right', 'prev_choice_right', 'prev_correct']
                
            # Calculate mean and SEM across mice for each coefficient
            coef_means = []
            coef_sems = []
            
            for coef in coef_order:
                values = [results.params[coef] for results in context_results.values()]
                coef_means.append(np.mean(values))
                coef_sems.append(np.std(values) / np.sqrt(len(values)))
            
            # Plot points and error bars
            x = np.arange(len(coef_order))
            ax.scatter(x, coef_means, 
                      color=context_colors[context_idx],
                      s=100,
                      zorder=2)
            ax.errorbar(x, coef_means, yerr=coef_sems,
                       color=context_colors[context_idx],
                       fmt='none',
                       capsize=5,
                       capthick=2,
                       zorder=1)
            
            # Customize plot
            ax.set_xticks(x)
            # Format labels with nicer display names
            display_names = []
            for name in coef_order:
                if name == 'stimulus_right':
                    display_names.append('stimulus right')
                else:
                    display_names.append(name.replace('_', ' '))
            
            ax.set_xticklabels(display_names, 
                              rotation=45, 
                              ha='right')
            ax.set_ylabel('Coefficient Value')
            ax.set_title(f'{context_names[context_idx]} Context')
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, zorder=0)
            
            # Set y-axis limits consistently across plots
            ax.set_ylim(-4, 4.5)
        
        plt.tight_layout()
        return fig, axes

    def analyze_all_mice_by_context_split_opto(self, all_task_dfs):
        """
        Fit GLM for each mouse and context combination, separately for opto and non-opto trials.
        
        Args:
            all_task_dfs (dict): Dictionary of task DataFrames keyed by mouse ID
        
        Returns:
            Dictionary with structure: {context: {'opto': {mouse: results}, 'no_opto': {mouse: results}}}
        """
        context_names = ['Congruent', 'Visual', 'Audio']
        results_dict = {context: {'no_opto': {}, 'opto': {}} for context in range(3)}
        
        for mouse, task_df in all_task_dfs.items():
            print(f"\nResults for mouse {mouse}:")
            print("=" * 50)
            
            # Print trial counts first
            print("\nTrial counts by context:")
            print(task_df['context'].value_counts())
            
            for context in range(3):
                print(f"\nContext: {context_names[context]}")
                print("-" * 30)
                
                # Filter for this context
                context_df = task_df[task_df['context'] == context].copy()
                
                # Split by opto condition
                for condition_name, is_opto in [('no_opto', 0), ('opto', 1)]:
                    condition_df = context_df[context_df['opto'] == is_opto].copy()
                    
                    if len(condition_df) < 20:  # Skip if too few trials
                        print(f"Skipping {condition_name} for context {context} (only {len(condition_df)} trials)")
                        continue
                    
                    print(f"\n{condition_name.upper()} trials ({len(condition_df)} trials):")
                    
                    try:
                        # Use the existing GLM function but with the filtered data
                        from statsmodels.discrete.discrete_model import Logit
                        from scipy.stats import zscore
                        
                        # Create lagged variables for previous choice and outcome
                        condition_df['prev_choice'] = condition_df['choice'].shift(1)
                        condition_df['prev_outcome'] = condition_df['outcome'].shift(1)
                        
                        # Drop trials without previous trial info
                        condition_df = condition_df.dropna(subset=['prev_choice', 'prev_outcome'])
                        
                        # Base features used in all contexts
                        feature_dict = {
                            'prev_choice_right': zscore((condition_df['prev_choice'].values == 1).astype(float)),
                            'prev_correct': zscore(condition_df['prev_outcome'].values.astype(float)),
                            'intercept': np.ones(len(condition_df))
                        }
                        
                        # Add context-specific stimulus features (binarized)
                        if context == 0:  # Congruent
                            feature_dict.update({
                                'audio_right': (condition_df['audio_stim'].values > 0).astype(float)
                            })
                        elif context == 1:  # Visual only
                            feature_dict['visual_right'] = (condition_df['visual_stim'].values == 90).astype(float)
                        elif context == 2:  # Audio only
                            feature_dict['audio_right'] = (condition_df['audio_stim'].values > 0).astype(float)
                        
                        # Create feature matrix
                        X = pd.DataFrame(feature_dict)
                        
                        # Target variable (1 = right choice)
                        y = (condition_df['choice'].values == 1).astype(float)
                        
                        # Fit model
                        model = Logit(y, X)
                        results = model.fit(disp=0)
                        
                        results_dict[context][condition_name][mouse] = results
                        
                        # Print summary statistics
                        print("\nCoefficient estimates and p-values:")
                        coef_df = pd.DataFrame({
                            'Coefficient': results.params,
                            'Std Error': results.bse,
                            'p-value': results.pvalues
                        })
                        print(coef_df)
                        
                        # Print model performance metrics
                        predictions = (results.predict(X) > 0.5).astype(float)
                        accuracy = np.mean(predictions == y)
                        print(f"\nModel accuracy: {accuracy:.3f}")
                        print(f"Pseudo R-squared: {results.prsquared:.3f}")
                        print(f"Number of trials: {len(y)}")
                        
                    except Exception as e:
                        print(f"Could not fit model for {condition_name}: {str(e)}")
                        continue
        
        return results_dict

    def analyze_all_mice_by_context_split_opto_enhanced(self, all_task_dfs, noise_levels=[1e-5, 1e-4, 1e-3], alphas=[0.001, 0.01, 0.1]):
        """
        Enhanced GLM fitting with multiple fallback strategies and adaptive regularization.
        
        Args:
            all_task_dfs (dict): Dictionary of task DataFrames
            noise_levels (list): Noise levels to try (in increasing order)
            alphas (list): Regularization strengths to try (in increasing order)
        
        Returns:
            Dictionary with structure: {context: {'opto': {mouse: results}, 'no_opto': {mouse: results}}}
        """
        from statsmodels.discrete.discrete_model import Logit
        from scipy.stats import zscore
        import warnings
        import numpy as np
        
        # Temporarily suppress statsmodels convergence warnings
        import statsmodels.tools.sm_exceptions as sm_exceptions
        warnings.filterwarnings('ignore', category=sm_exceptions.ConvergenceWarning)
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        
        context_names = ['Congruent', 'Visual', 'Audio']
        results_dict = {context: {'no_opto': {}, 'opto': {}} for context in range(3)}
        
        for mouse, task_df in all_task_dfs.items():
            print(f"\nResults for mouse {mouse}:")
            print("=" * 50)
            
            # Print trial counts first
            print("\nTrial counts by context:")
            print(task_df['context'].value_counts())
            
            for context in range(3):
                print(f"\nContext: {context_names[context]}")
                print("-" * 30)
                
                # Filter for this context
                context_df = task_df[task_df['context'] == context].copy()
                
                # Split by opto condition
                for condition_name, is_opto in [('no_opto', 0), ('opto', 1)]:
                    condition_df = context_df[context_df['opto'] == is_opto].copy()
                    
                    if len(condition_df) < 10:  # Skip if too few trials
                        print(f"Skipping {condition_name} for context {context} (only {len(condition_df)} trials)")
                        continue
                    
                    print(f"\n{condition_name.upper()} trials ({len(condition_df)} trials):")
                    
                    try:
                        # Create lagged variables for previous choice and outcome
                        condition_df['prev_choice'] = condition_df['choice'].shift(1)
                        condition_df['prev_outcome'] = condition_df['outcome'].shift(1)
                        
                        # Drop trials without previous trial info
                        condition_df = condition_df.dropna(subset=['prev_choice', 'prev_outcome'])
                        
                        if len(condition_df) < 10:
                            print(f"  Too few trials after filtering: {len(condition_df)}")
                            continue
                        
                        # Base features used in all contexts
                        feature_dict = {
                            'prev_choice_right': zscore((condition_df['prev_choice'].values == 1).astype(float)),
                            'prev_correct': zscore(condition_df['prev_outcome'].values.astype(float)),
                            'intercept': np.ones(len(condition_df))
                        }
                        
                        # Add context-specific stimulus features (binarized)
                        if context == 0:  # Congruent
                            # Use only one stimulus feature for congruent to avoid multicollinearity
                            feature_dict['stimulus_right'] = (condition_df['visual_stim'].values == 90).astype(float)
                        elif context == 1:  # Visual only
                            feature_dict['visual_right'] = (condition_df['visual_stim'].values == 90).astype(float)
                        elif context == 2:  # Audio only
                            feature_dict['audio_right'] = (condition_df['audio_stim'].values > 0).astype(float)
                        
                        # Create feature matrix
                        X_orig = pd.DataFrame(feature_dict)
                        y = (condition_df['choice'].values == 1).astype(float)
                        
                        # Get stats on distribution of predictors and outcome
                        choice_balance = y.mean()
                        print(f"  Choice balance (fraction right): {choice_balance:.2f}")
                        
                        # Try multiple fitting strategies in sequence
                        result = None
                        success = False
                        
                        # Strategy 1: Try standard fit first
                        try:
                            model = Logit(y, X_orig)
                            result = model.fit(disp=0, method='newton', maxiter=200)
                            print("  Model fitted with standard Newton-Raphson")
                            success = True
                        except:
                            pass
                        
                        # Strategy 2: Try BFGS which is more robust to separation
                        if not success:
                            try:
                                result = model.fit(disp=0, method='bfgs', maxiter=1000)
                                print("  Model fitted with BFGS")
                                success = True
                            except:
                                pass
                        
                        # Strategy 3: Try progressively adding noise and regularization
                        if not success:
                            for noise_level in noise_levels:
                                if success:
                                    break
                                
                                # Add noise to predictors
                                np.random.seed(42)  # For reproducibility
                                X_noisy = X_orig.copy()
                                for col in X_noisy.columns:
                                    if col != 'intercept':
                                        X_noisy[col] = X_noisy[col] + np.random.normal(0, noise_level, len(X_noisy))
                                
                                model = Logit(y, X_noisy)
                                
                                # Try with different regularization strengths
                                for alpha in alphas:
                                    try:
                                        result = model.fit_regularized(
                                            method='l1', 
                                            alpha=alpha, 
                                            trim_mode='size',
                                            acc=1e-10,  # Higher accuracy
                                            maxiter=1000,  # More iterations
                                            disp=0
                                        )
                                        print(f"  Model fitted with noise={noise_level} and L1 alpha={alpha}")
                                        success = True
                                        break
                                    except:
                                        continue
                        
                        # Strategy 4: Last resort - Bayesian approach with strong priors
                        if not success:
                            try:
                                # Add maximum noise and try again with standard solver
                                X_max_noise = X_orig.copy()
                                for col in X_max_noise.columns:
                                    if col != 'intercept':
                                        X_max_noise[col] = X_max_noise[col] + np.random.normal(0, 0.01, len(X_max_noise))
                                
                                model = Logit(y, X_max_noise)
                                result = model.fit(disp=0, method='lbfgs', maxiter=2000)
                                print("  Model fitted with maximum noise and L-BFGS")
                                success = True
                            except Exception as e:
                                print(f"  All fitting strategies failed: {str(e)}")
                                continue
                        
                        if success:
                            results_dict[context][condition_name][mouse] = result
                            
                            # Print summarized results
                            print("\nCoefficient estimates:")
                            coef_df = pd.DataFrame({
                                'Coefficient': result.params,
                                'Std Error': result.bse if hasattr(result, 'bse') else "N/A"
                            })
                            print(coef_df)
                            
                            # Calculate model accuracy on original data
                            predictions = (result.predict(X_orig) > 0.5).astype(float)
                            accuracy = np.mean(predictions == y)
                            print(f"\nModel accuracy: {accuracy:.3f}")
                        
                    except Exception as e:
                        print(f"Error in overall model pipeline: {str(e)}")
                        continue
                    
        # Re-enable warnings
        warnings.filterwarnings('default', category=sm_exceptions.ConvergenceWarning)
        warnings.filterwarnings('default', category=RuntimeWarning)
        
        return results_dict

    def plot_coefficients_by_context_compare_opto(self, results_dict, figsize=(10, 4), dpi=800):
        """
        Plot coefficient values for opto and non-opto conditions side by side.
        
        Args:
            results_dict: Output from analyze_all_mice_by_context_split_opto
            figsize (tuple): Figure size
            dpi (int): Figure resolution
            
        Returns:
            fig, axes: Figure and axes objects
        """
        context_names = ['Congruent', 'Visual', 'Audio']
        context_colors = ['purple', '#EC008C', '#27AAE1']
        fig, axes = plt.subplots(1, 3, figsize=figsize, dpi=dpi)
        
        for context_idx, (context_results, ax) in enumerate(zip(results_dict.values(), axes)):
            # Define desired order of coefficients for each context
            if context_idx == 0:  # Congruent
                coef_order = ['stimulus_right', 'prev_choice_right', 'prev_correct']
            elif context_idx == 1:  # Visual
                coef_order = ['visual_right', 'prev_choice_right', 'prev_correct']
            else:  # Audio
                coef_order = ['audio_right', 'prev_choice_right', 'prev_correct']
            
            # Calculate means and SEMs for both conditions
            no_opto_means = []
            no_opto_sems = []
            opto_means = []
            opto_sems = []
            
            # Check if we have any mice with both conditions
            common_mice = set(context_results.get('no_opto', {}).keys()) & set(context_results.get('opto', {}).keys())
            if not common_mice:
                ax.text(0.5, 0.5, "Insufficient data", ha='center', va='center', transform=ax.transAxes)
                continue
            
            # Only use common mice for fair comparison
            for coef in coef_order:
                # Non-opto condition
                no_opto_values = [results.params[coef] for mouse, results in context_results['no_opto'].items() 
                                  if mouse in common_mice and coef in results.params.index]
                if no_opto_values:
                    no_opto_means.append(np.mean(no_opto_values))
                    no_opto_sems.append(np.std(no_opto_values) / np.sqrt(len(no_opto_values)))
                else:
                    no_opto_means.append(np.nan)
                    no_opto_sems.append(np.nan)
                
                # Opto condition
                opto_values = [results.params[coef] for mouse, results in context_results['opto'].items() 
                              if mouse in common_mice and coef in results.params.index]
                if opto_values:
                    opto_means.append(np.mean(opto_values))
                    opto_sems.append(np.std(opto_values) / np.sqrt(len(opto_values)))
                else:
                    opto_means.append(np.nan)
                    opto_sems.append(np.nan)
            
            # Plot points and error bars
            x = np.arange(len(coef_order))
            width = 0.25  # Spacing between conditions
            
            # Plot non-opto points (circles)
            ax.scatter(x - width/2, np.abs(no_opto_means), 
                      color=context_colors[context_idx],
                      s=100,
                      alpha=0.7,
                      marker='o',
                      label='No Opto')
            ax.errorbar(x - width/2, np.abs(no_opto_means), yerr=no_opto_sems,
                       color=context_colors[context_idx],
                       alpha=0.7,
                       fmt='none',
                       capsize=5,
                       capthick=2)
            
            # Plot opto points (stars)
            ax.scatter(x + width/2, np.abs(opto_means), 
                      color=context_colors[context_idx],
                      s=120,
                      alpha=0.7,
                      marker='*',
                      label='Opto')
            ax.errorbar(x + width/2, np.abs(opto_means), yerr=opto_sems,
                       color=context_colors[context_idx],
                       alpha=0.7,
                       fmt='none',
                       capsize=5,
                       capthick=2)
            
            # Draw connecting lines
            for i in range(len(x)):
                if not (np.isnan(no_opto_means[i]) or np.isnan(opto_means[i])):
                    ax.plot([x[i] - width/2, x[i] + width/2], 
                           [np.abs(no_opto_means[i]), np.abs(opto_means[i])],
                           color=context_colors[context_idx],
                           alpha=0.3,
                           linestyle='-')
            
            # Customize plot
            ax.set_xticks(x)
            ax.set_xticklabels([name.replace('_', ' ') for name in coef_order], 
                              rotation=45, 
                              ha='right')
            ax.set_ylabel('|Coefficient Value|')
            ax.set_title(f'{context_names[context_idx]} Context')
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            
            # Set y-axis limits
            ax.set_ylim(-1, 5)
            
            # Add legend only to the first subplot
            if context_idx == 0:
                ax.legend(fontsize=8, frameon=False, loc='upper right')
        
        plt.tight_layout()
        return fig, axes

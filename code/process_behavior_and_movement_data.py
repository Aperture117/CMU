import os 
import re
import pickle
import numpy as np
import pandas as pd
import numpy.random as npr
from pathlib import Path
from typing import Dict, List, Tuple, Any
import warnings
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import scipy.io as sio
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
import statsmodels.api as sm
from scipy.stats import zscore
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from datetime import datetime
import scipy.io as sio
from pathlib import Path
from typing import Dict, List, Tuple, Any
import warnings

class SessionDataProcessor:
    def __init__(self, mouse_name=None, date=None):
        """
        Initialize the SessionDataProcessor.
        
        Args:
            mouse_name (str, optional): Mouse name (with or without CB prefix)
            date (str, optional): Session date in YYMMDD format
        """
        self.mouse_name = mouse_name
        self.date = date
        self.data_cell = None
        self.raw_data = None
        self.trial_data = None
        self.task_df = None
        self.ranges = None  # Will be set during processing if needed
        
        # Initialize mapping dictionaries
        self.visual_stim_maps = {
            'psych': {
                1: 0, 2: 15, 3: 25, 4: 65, 5: 75, 6: 90,
                7: 0, 8: 15, 9: 25, 10: 65, 11: 75, 12: 90,
                13: 0, 14: 15, 15: 25, 16: 65, 17: 75, 18: 90
            },
            '2loc': {
                1: 0, 2: 90, 3: 0, 4: 90, 5: 90, 6: 0
            },
            'avc': {
                1: 0, 2: 90,      # Congruent trials
                3: 0, 4: 90,      # Visual only trials
                5: np.nan, 6: np.nan  # Audio only trials
            },
            'aud': {
                1: np.nan, 2: np.nan
            },
            'aud_psych': {
                1: np.nan, 2: np.nan, 3: np.nan, 4: np.nan, 5: np.nan, 6: np.nan,
                7: np.nan, 8: np.nan
            }
        }
        
        self.audio_stim_maps = {
            'psych': {
                1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1,
                7: 1, 8: 1, 9: 1, 10: 0, 11: 0, 12: 0,
                13: 1, 14: 1, 15: 1, 16: 0, 17: 0, 18: 0
            },
            '2loc': {
                1: 0, 2: 1, 3: 1, 4: 0, 5: 0, 6: 1
            },
            'avc': {
                1: 0, 2: 1,      # Congruent trials
                3: np.nan, 4: np.nan,  # Visual only trials
                5: 0, 6: 1       # Audio only trials
            },
            'aud': {
                1: 0, 2: 1
            },
            'aud_psych': {
                1: 0, 2: 0, 3: 0, 4: 0, 5: 1, 6: 1,
                7: 1, 8: 1
            }
        }
        
        self.context_maps = {
            'psych': {
                1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0,
                7: 1, 8: 1, 9: 1, 10: 1, 11: 1, 12: 1,
                13: 2, 14: 2, 15: 2, 16: 2, 17: 2, 18: 2
            },
            '2loc': {
                1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2
            },
            'avc': {
                1: 0, 2: 0,      # Congruent trials (0)
                3: 1, 4: 1,      # Visual only trials (1)
                5: 2, 6: 2       # Audio only trials (2)
            },
            'aud': {
                1: 2, 2: 2
            },
            'aud_psych': {
                1: 2, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2,
                7: 2, 8: 2
            }
        }

    def load_session_files(self, data_dir):
        """
        Load and combine multiple Cell and Data .mat files for a session.
        
        Args:
            data_dir (str): Directory containing the session files
        """
        if not hasattr(self, 'mouse_name') or not hasattr(self, 'date'):
            raise ValueError("Mouse name and date must be set before loading files")
        
        # Find all matching files using glob pattern
        data_dir = Path(data_dir)
        cell_files = list(data_dir.glob(f"{self.mouse_name}_{self.date}*Cell*.mat"))
        data_files = list(data_dir.glob(f"{self.mouse_name}_{self.date}*.mat"))
        data_files = [f for f in data_files if "Cell" not in f.name]  # Exclude Cell files
        
        print(f"Found {len(cell_files)} Cell files and {len(data_files)} Data files")
        
        if not cell_files or not data_files:
            raise FileNotFoundError(f"No matching files found for {self.mouse_name}_{self.date}")
        
        # Load and combine Cell files
        all_data_cells = []
        for cell_file in cell_files:
            print(f"Loading Cell file: {cell_file}")
            try:
                cell_data = sio.loadmat(str(cell_file))
                all_data_cells.append(cell_data['dataCell'])
            except Exception as e:
                print(f"Error loading {cell_file}: {str(e)}")
        
        # Combine data cells along axis 1 (trials)
        self.data_cell = np.hstack(all_data_cells) if all_data_cells else None
        
        # Load and combine Data files
        all_raw_data = []
        for data_file in data_files:
            print(f"Loading Data file: {data_file}")
            try:
                data = sio.loadmat(str(data_file))
                all_raw_data.append(data['data'])
            except Exception as e:
                print(f"Error loading {data_file}: {str(e)}")
        
        # Combine raw data along axis 1 (time points)
        self.raw_data = np.hstack(all_raw_data) if all_raw_data else None
        
        if self.data_cell is None or self.raw_data is None:
            raise Exception("Failed to load data")
        
        # Print structure of first trial for debugging
        if self.data_cell.size > 0:
            print("\nFirst trial structure:")
            first_trial = self.data_cell[0, 0]
            print("Available fields:", first_trial.dtype.names)

    def _get_cell_vals(self, data_cell, field_path):
        """Helper to get values from nested fields."""
        fields = field_path.split('.')
        values = []
        
        for i in range(data_cell.shape[1]):
            try:
                val = data_cell[0, i]
                for field in fields:
                    if field in val.dtype.names:
                        val = val[field]
                        if val.shape == (1, 1):
                            val = val[0, 0]
                    else:
                        val = None
                        break
                if val is not None:
                    values.append(val)
            except Exception as e:
                print(f"Error accessing field {field_path} in trial {i}: {str(e)}")
                continue
        
        return values

    def _get_trial_data(self, trial):
        """Helper method to get trial data."""
        if 'dat' in trial.dtype.names:
            return trial['dat']
        raise ValueError("Could not find data field in trial structure")

    def parse_data_array(self, data_cell, data):
        """
        Parse the data array and organize it into trials.
        
        Args:
            data_cell: MATLAB cell array containing trial information
            data: Raw data array of shape (9, N) where N is the number of time points
        """
        # Convert data to numpy array if it isn't already
        data = np.array(data)
        
        # Get ITI signal (9th row, index 8)
        iti_signal = data[8, :]
        
        # Find trial starts and stops
        t_starts = np.concatenate(([0], np.where(np.diff(iti_signal) == -1)[0] + 1))
        t_stops = np.where(np.diff(iti_signal) == 1)[0] + 1
        
        print(f"Found {len(t_starts)} trial starts and {len(t_stops)} trial stops")
        
        # Validate trial counts
        if len(t_starts) > len(t_stops):
            t_starts = t_starts[:-1]
            if len(t_starts) > len(t_stops):
                raise ValueError('Multiple incomplete trials')
        elif len(t_starts) < len(t_stops):
            raise ValueError('Trial starts missing')
        
        # Validate against data_cell length
        n_trials = len(t_starts)
        if n_trials > data_cell.shape[1]:
            print(f'Warning: {n_trials} trials in data array but only {data_cell.shape[1]} in dataCell')
            n_trials = data_cell.shape[1]
        elif n_trials < data_cell.shape[1]:
            print(f'Warning: Only {n_trials} trials in data array but {data_cell.shape[1]} in dataCell')
            data_cell = data_cell[:, :n_trials]
        
        # Create a new structured array with the same fields plus 'dat'
        dtype = [(name, data_cell[0, 0][name].dtype) for name in data_cell[0, 0].dtype.names]
        dtype.append(('dat', 'O'))
        new_data_cell = np.empty(data_cell.shape, dtype=dtype)
        
        # Fill in trial data
        for i in range(n_trials):
            try:
                # Copy existing fields
                for field in data_cell[0, 0].dtype.names:
                    new_data_cell[0, i][field] = data_cell[0, i][field]
                
                # Extract and store trial data
                trial_data = data[:, t_starts[i]:t_stops[i]]
                new_data_cell[0, i]['dat'] = trial_data
                
            except Exception as e:
                print(f"Error processing trial {i}: {str(e)}")
                continue
        
        return new_data_cell

    def get_run_param(self, data_cell, ranges):
        """Extract running parameters from trial data."""
        if ranges is None:
            ranges = self.ranges

        n_trials = data_cell.shape[1]
        
        # Initialize arrays
        theta = [None] * n_trials
        x_vel = [None] * n_trials
        ev_array = [None] * n_trials
        y_pos = [None] * n_trials
        turn_array = [None] * n_trials
        trial_id = [None] * n_trials
        
        # Check if segmented by safely accessing maze structure
        first_trial = data_cell[0, 0]
        maze_data = first_trial['maze'][0, 0]  # Correctly access MATLAB structure
        segmented = 'numLeft' in maze_data.dtype.names
        
        out = {}
        if segmented:
            # Safely get numSeg from first trial that has it
            for i in range(n_trials):
                maze = data_cell[0, i]['maze'][0, 0]
                if 'numLeft' in maze.dtype.names and maze['numLeft'].size > 0:
                    out['numSeg'] = int(maze['numLeft'][0, 0])
                    break
        
        for i in range(n_trials):
            try:
                trial = data_cell[0, i]
                dat = trial['dat']
                
                # Extract position and angle data
                y_pos[i] = dat[2, :]
                temp_theta = np.rad2deg(np.mod(dat[3, :], 2*np.pi))
                theta[i] = temp_theta
                
                # Store trial ID
                trial_id[i] = i * np.ones_like(y_pos[i])
                
                # Get x velocity
                x_vel[i] = dat[4, :]
                
                # Calculate evidence array
                if segmented:
                    ev_range = np.zeros(out['numSeg'])
                    maze = trial['maze'][0, 0]  # Correctly access MATLAB structure
                    
                    # Safely access leftTrial and leftDotLoc
                    is_left_trial = bool(maze['leftTrial'][0, 0]) if 'leftTrial' in maze.dtype.names else False
                    left_dot_loc = maze['leftDotLoc'][0, 0].flatten() - 1 if 'leftDotLoc' in maze.dtype.names else np.array([])
                    
                    if is_left_trial:
                        ev_range[left_dot_loc] = 1
                        ev_range = np.cumsum(ev_range)
                    else:
                        non_left_loc = np.setdiff1d(np.arange(out['numSeg']), left_dot_loc)
                        ev_range[non_left_loc] = 1
                        ev_range = np.cumsum(ev_range)
                    
                    # Expand based on position
                    ev_array[i] = np.zeros_like(x_vel[i])
                    for j in range(out['numSeg']):
                        if j < out['numSeg'] - 1:
                            ind = (y_pos[i] >= ranges[j]) & (y_pos[i] < ranges[j+1])
                        else:
                            ind = y_pos[i] >= ranges[j]
                        ev_array[i][ind] = ev_range[j]
                else:
                    ev_array[i] = np.nan * np.ones_like(y_pos[i])
                
                # Create turn array - safely access result structure
                result = trial['result'][0, 0]
                is_left_turn = bool(result['leftTurn'][0, 0]) if 'leftTurn' in result.dtype.names else False
                turn_array[i] = np.ones_like(x_vel[i]) if is_left_turn else np.zeros_like(x_vel[i])
                
                # Center theta at 0
                theta[i] = 90 - theta[i]
                
            except Exception as e:
                print(f"Error processing trial {i}: {str(e)}")
                continue
        
        # Package output
        out.update({
            'evArray': ev_array,
            'thetaArray': theta,
            'xVelArray': x_vel,
            'yPosArray': y_pos,
            'turnArray': turn_array,
            'trialIDArray': trial_id
        })
        
        # Remove any None values before concatenating
        valid_trials = [i for i in range(n_trials) if all(arr[i] is not None for arr in [ev_array, theta, x_vel, y_pos, turn_array, trial_id])]
        
        # Concatenate all arrays from valid trials
        out['evArrayAll'] = np.concatenate([ev_array[i] for i in valid_trials])
        out['thetaAll'] = np.concatenate([theta[i] for i in valid_trials])
        out['xVelAll'] = np.concatenate([x_vel[i] for i in valid_trials])
        out['yPosAll'] = np.concatenate([y_pos[i] for i in valid_trials])
        out['turnAll'] = np.concatenate([turn_array[i] for i in valid_trials])
        out['trialIDAll'] = np.concatenate([trial_id[i] for i in valid_trials])
        
        return out, segmented

    def bin_run_param(self, run_params: dict, n_bins: int, segmented: bool) -> dict:
        """Python equivalent of binRunParam.m"""
        out = {}
        
        # Create position bins
        y_pos_min = np.min(run_params['yPosAll'])
        y_pos_max = np.max(run_params['yPosAll'])
        out['yPosBinInd'] = np.linspace(y_pos_min, y_pos_max, n_bins + 1)
        
        if segmented:
            # Initialize arrays for segmented data
            shape = (run_params['numSeg'] + 1, n_bins)
            out.update({
                'meanTheta': np.zeros(shape),
                'meanXVel': np.zeros(shape),
                'stdTheta': np.zeros(shape),
                'stdXVel': np.zeros(shape),
                'binDataThetaEv': np.empty(shape, dtype=object),
                'binDataXVelEv': np.empty(shape, dtype=object),
                'binDataTurnEv': np.empty(shape, dtype=object)
            })
        
        # Initialize arrays for all data
        out.update({
            'binDataThetaAll': [None] * n_bins,
            'binDataXVelAll': [None] * n_bins,
            'binDataTurnAll': [None] * n_bins,
            'binDataTrialIDAll': [None] * n_bins,
            'meanYPosAll': np.zeros(n_bins)
        })
        
        unique_trials = np.unique(run_params['trialIDAll'])
        
        for i in range(n_bins):
            # Generate position mask
            pos_mask = (run_params['yPosAll'] >= out['yPosBinInd'][i]) & \
                      (run_params['yPosAll'] < out['yPosBinInd'][i+1])
            
            # Calculate statistics for all data
            trial_ids = run_params['trialIDAll'][pos_mask]
            out['binDataThetaAll'][i] = self._group_stats(run_params['thetaAll'][pos_mask], trial_ids)
            out['binDataXVelAll'][i] = self._group_stats(run_params['xVelAll'][pos_mask], trial_ids)
            out['binDataTurnAll'][i] = self._group_stats(run_params['turnAll'][pos_mask], trial_ids)
            out['meanYPosAll'][i] = np.mean(run_params['yPosAll'][pos_mask])
            out['binDataTrialIDAll'][i] = self._group_stats(trial_ids, trial_ids)
            
            if segmented:
                for k in range(run_params['numSeg'] + 1):
                    # Generate evidence mask
                    ev_mask = run_params['evArrayAll'] == k
                    combined_mask = pos_mask & ev_mask
                    
                    # Calculate statistics for segmented data
                    out['meanTheta'][k, i] = np.mean(run_params['thetaAll'][combined_mask])
                    out['meanXVel'][k, i] = np.mean(run_params['xVelAll'][combined_mask])
                    out['stdTheta'][k, i] = np.std(run_params['thetaAll'][combined_mask])
                    out['stdXVel'][k, i] = np.std(run_params['xVelAll'][combined_mask])
                    
                    # Store raw data
                    out['binDataThetaEv'][k, i] = run_params['thetaAll'][combined_mask]
                    out['binDataXVelEv'][k, i] = run_params['xVelAll'][combined_mask]
                    out['binDataTurnEv'][k, i] = run_params['turnAll'][combined_mask]
        
        return out

    def _group_stats(self, values: np.ndarray, groups: np.ndarray) -> np.ndarray:
        """Calculate mean for each group (equivalent to MATLAB's grpstats)"""
        unique_groups = np.unique(groups)
        return np.array([np.mean(values[groups == g]) for g in unique_groups])

    def create_task_dataframe(self):
        """Create a DataFrame containing trial-level behavioral data."""
        if self.data_cell is None:
            raise ValueError("Cell data not loaded. Call load_session_files first.")
        
        trial_info = []
        for trial in self.data_cell[0]:
            info = {
                'correct': trial['result'][0, 0]['correct'][0, 0][0, 0],
                'left_choice': trial['result'][0, 0]['leftTurn'][0, 0][0, 0],
                'condition': trial['maze'][0, 0]['condition'][0, 0][0, 0]
            }
            
            # Add additional maze information if available
            maze = trial['maze'][0, 0]
            if 'numLeft' in maze.dtype.names:
                info['num_left'] = maze['numLeft'][0, 0][0, 0]
            if 'leftDotLoc' in maze.dtype.names:
                info['left_dot_loc'] = maze['leftDotLoc'][0, 0].flatten()
            
            trial_info.append(info)
        
        self.task_df = pd.DataFrame(trial_info)
        self.task_df['date'] = self.date
        self.task_df['mouse'] = self.mouse_name

    def create_trial_data(self, data_cell):
        """
        Create a structured dictionary containing trial-by-trial data.
        
        Args:
            data_cell: Processed data cell array containing trial information
            
        Returns:
            dict: Trial data organized by fields
        """
        n_trials = data_cell.shape[1]
        
        # Initialize dictionary with empty lists for each field
        trial_data = {
            'Time': [],
            'X_pos': [],
            'Y_pos': [],
            'View': [],
            'X_velocity': [],
            'Y_velocity': [],
            'World': [],
            'Reward': [],
            'IsITI': [],
            'Correct': [],
            'Left': [],
            'Condition': []
        }
        
        # Loop through trials and fill in the data
        for i in range(n_trials):
            try:
                trial = data_cell[0, i]
                dat = trial['dat']
                result = trial['result'][0, 0]
                maze = trial['maze'][0, 0]
                
                # Store time series data
                trial_data['Time'].append(dat[0, :])
                trial_data['X_pos'].append(dat[1, :])
                trial_data['Y_pos'].append(dat[2, :])
                trial_data['View'].append(dat[3, :])
                trial_data['X_velocity'].append(dat[4, :])
                trial_data['Y_velocity'].append(dat[5, :])
                trial_data['World'].append(dat[6, :])
                trial_data['Reward'].append(dat[7, :])
                trial_data['IsITI'].append(dat[8, :])
                
                # Store trial info
                trial_data['Correct'].append(bool(result['correct'][0, 0]))
                trial_data['Left'].append(bool(result['leftTurn'][0, 0]))
                
                # Store condition (with error handling)
                if 'condition' in maze.dtype.names:
                    trial_data['Condition'].append(maze['condition'][0, 0])
                else:
                    trial_data['Condition'].append(None)
                    
            except Exception as e:
                print(f"Error processing trial {i}: {str(e)}")
                # Append None for this trial
                for key in trial_data.keys():
                    if len(trial_data[key]) < i + 1:
                        trial_data[key].append(None)
        
        return trial_data

    def create_movement_matrix(self, trial_data):
        """
        Convert trial data into continuous movement matrices.
        
        Args:
            trial_data (dict): Trial data structure containing time series data
            
        Returns:
            dict: Movement data matrices and trial event indices
        """
        # First, get total length of data
        total_length = sum(len(x) for x in trial_data['Time'] if x is not None)
        
        # Initialize movement matrices
        movement_data = {
            'xpos': np.zeros(total_length),
            'ypos': np.zeros(total_length),
            'view': np.zeros(total_length),
            'xvel': np.zeros(total_length),
            'yvel': np.zeros(total_length),
            'trial_start': [],
            'trial_end': []
        }
        
        # Fill matrices
        current_idx = 0
        for i, time_series in enumerate(trial_data['Time']):
            if time_series is None:
                continue
            
            length = len(time_series)
            
            # Record trial start/end indices
            movement_data['trial_start'].append(current_idx)
            movement_data['trial_end'].append(current_idx + length)
            
            # Fill in movement data
            slice_idx = slice(current_idx, current_idx + length)
            movement_data['xpos'][slice_idx] = trial_data['X_pos'][i]
            movement_data['ypos'][slice_idx] = trial_data['Y_pos'][i]
            movement_data['view'][slice_idx] = trial_data['View'][i]
            movement_data['xvel'][slice_idx] = trial_data['X_velocity'][i]
            movement_data['yvel'][slice_idx] = trial_data['Y_velocity'][i]
            
            current_idx += length
        
        # Convert trial events to arrays
        movement_data['trial_start'] = np.array(movement_data['trial_start'])
        movement_data['trial_end'] = np.array(movement_data['trial_end'])
        
        # Add trialized version
        movement_data['trialized'] = self._trialize_movement_data(
            movement_data,
            movement_data['trial_start'],
            movement_data['trial_end']
        )
        
        return movement_data

    def _trialize_movement_data(self, movement_data, trial_starts, trial_ends):
        """
        Convert continuous movement data into a trial-based format.
        
        Args:
            movement_data (dict): Movement data matrices
            trial_starts (array): Trial start indices
            trial_ends (array): Trial end indices
            
        Returns:
            dict: Trial-based movement data
        """
        n_trials = len(trial_starts)
        max_length = max(end - start for start, end in zip(trial_starts, trial_ends))
        
        # Initialize trial-based arrays
        trialized = {
            'xpos': np.full((n_trials, max_length), np.nan),
            'ypos': np.full((n_trials, max_length), np.nan),
            'view': np.full((n_trials, max_length), np.nan),
            'xvel': np.full((n_trials, max_length), np.nan),
            'yvel': np.full((n_trials, max_length), np.nan),
            'time_points': np.full((n_trials,), 0, dtype=int)
        }
        
        # Fill in trial-based data
        for i in range(n_trials):
            start = trial_starts[i]
            end = trial_ends[i]
            length = end - start
            
            trialized['time_points'][i] = length
            
            for key in ['xpos', 'ypos', 'view', 'xvel', 'yvel']:
                trialized[key][i, :length] = movement_data[key][start:end]
        
        return trialized

    def process_session(self, data_dir='.'):
        """Process a complete session."""
        # Load the data files
        self.load_session_files(data_dir)
        
        # Parse data array
        self.data_cell = self.parse_data_array(self.data_cell, self.raw_data)
        
        # Get run parameters
        run_params, segmented = self.get_run_param(self.data_cell, self.ranges)
        
        # Bin parameters
        binned_params = self.bin_run_param(run_params, n_bins=100, segmented=segmented)
        
        # Create trial data structure
        trial_data = self.create_trial_data(self.data_cell)
        
        # Create movement matrices
        movement_data = self.create_movement_matrix(trial_data)
        
        return {
            'run_params': run_params,
            'binned_params': binned_params,
            'segmented': segmented,
            'trial_data': trial_data,
            'movement': movement_data
        }

    def detect_task_type(self, conditions, trial_data):
        """
        Detect task type based on conditions and trial structure.
        
        Args:
            conditions (set): Set of condition numbers
            trial_data (dict): Trial data containing maze information
        
        Returns:
            str: Task type ('2loc', 'avc', 'psych', 'aud', 'aud_psych', or 'unknown')
        """
        max_condition = max(conditions)
        
        # First check for audio task (conditions 1-2 only)
        if max_condition <= 2 and conditions.issubset({1, 2}):
            # Additional check for audio task - look at visual/audio stim patterns
            # Sample a few trials to check for audio-only stimuli
            sample_trials = min(10, len(trial_data['Condition']))
            has_visual = False
            
            for i in range(sample_trials):
                maze_data = trial_data.get('maze', [])
                if maze_data and i < len(maze_data):
                    if 'visualStim' in maze_data[i].dtype.names:
                        has_visual = True
                        break
            
            return 'aud' if not has_visual else '2loc'
        
        elif max_condition <= 6 and len(conditions) <= 6:
            # Look at the temporal pattern of conditions
            condition_sequence = trial_data['Condition']
            
            # Convert condition sequence to integers
            condition_sequence = [int(cond.item()) if isinstance(cond, np.ndarray) else int(cond) 
                                for cond in condition_sequence]
            
            # Check for blocked structure characteristic of AVC
            runs = []
            current_run = [condition_sequence[0]]
            
            for cond in condition_sequence[1:]:
                current_base = (current_run[0] - 1) // 2 * 2 + 1
                if cond in (current_base, current_base + 1):
                    current_run.append(cond)
                else:
                    runs.append(current_run)
                    current_run = [cond]
            runs.append(current_run)
            
            avg_run_length = np.mean([len(run) for run in runs])
            return 'avc' if avg_run_length > 10 else '2loc'
        
        elif max_condition <= 18:
            return 'psych'
        elif max_condition <= 8:
            return 'aud_psych'
        
        return 'unknown'

    def get_session_data(self, mouse_name, date, base_dir='/Volumes/Runyan5/Akhil/behavior/', verbose=False):
        """
        Get processed task DataFrame and trialized movement data for a specific session.
        
        Args:
            mouse_name (str): Mouse identifier
            date (str): Session date
            base_dir (str): Base directory for data
            verbose (bool): Whether to print debug information
        """
        # Add CB or AB prefix if not present and update instance variables
        if not (mouse_name.startswith('CB') or mouse_name.startswith('AB')):
            # Try to determine if this is a CB or AB mouse based on directory existence
            cb_path = os.path.join(base_dir, f"CB{mouse_name}")
            ab_path = os.path.join(base_dir, f"AB{mouse_name}")
            
            if os.path.exists(cb_path):
                self.mouse_name = f"CB{mouse_name}"
            elif os.path.exists(ab_path):
                self.mouse_name = f"AB{mouse_name}"
            else:
                # Default to CB if can't determine
                self.mouse_name = f"CB{mouse_name}"
                if verbose:
                    print(f"Warning: Could not determine prefix for mouse {mouse_name}, defaulting to {self.mouse_name}")
        else:
            self.mouse_name = mouse_name
            
        self.date = date
        
        # Process session
        data_dir = os.path.join(base_dir, self.mouse_name)
        session_data = self.process_session(data_dir)
        
        # Extract conditions safely
        conditions = set()
        for trial_condition in session_data['trial_data']['Condition']:
            if isinstance(trial_condition, np.ndarray):
                if trial_condition.size > 0:
                    condition = trial_condition.item()
                    if isinstance(condition, np.ndarray):
                        condition = condition.item()
                    conditions.add(condition)
        
        # Determine task type
        task_type = self.detect_task_type(conditions, session_data['trial_data'])
        if verbose:
            print(f"Detected task type: {task_type}")
            print(f"Unique conditions: {sorted(conditions)}")
        
        # Create task DataFrame
        trial_data = []
        for i in range(len(session_data['trial_data']['Correct'])):
            # Safely extract condition number
            condition = session_data['trial_data']['Condition'][i]
            if isinstance(condition, np.ndarray):
                condition = condition.item()
                if isinstance(condition, np.ndarray):
                    condition = condition.item()
            
            # Get opto status
            opto = 0  # default to no opto
            try:
                # Direct access to trial data
                trial_maze = self.data_cell[0, i]['maze'][0, 0]
                if 'isOptoTrial' in trial_maze.dtype.names:
                    opto = int(trial_maze['isOptoTrial'][0, 0])
                    if verbose and i < 5:  # Debug first 5 trials only if verbose
                        print(f"Trial {i}: isOptoTrial value = {opto}")
            except Exception as e:
                if verbose:
                    print(f"Error accessing opto status for trial {i}: {str(e)}")
            
            # Create trial dictionary
            trial = {
                'trial_num': i,
                'choice': 0 if session_data['trial_data']['Left'][i] else 1,  # 0 = left, 1 = right
                'condition': condition,
                'task_type': task_type,
                'visual_stim': self.visual_stim_maps[task_type].get(condition, np.nan),
                'audio_stim': self.audio_stim_maps[task_type].get(condition, np.nan),
                'context': self.context_maps[task_type].get(condition, -1),
                'outcome': int(session_data['trial_data']['Correct'][i]),
                'opto': opto
            }
            trial_data.append(trial)
        
        # Create DataFrame
        task_df = pd.DataFrame(trial_data)
        
        # Add metadata
        task_df['date'] = date
        task_df['mouse'] = self.mouse_name
        
        # Reorder columns
        column_order = ['trial_num', 'choice', 'condition', 'task_type', 
                       'visual_stim', 'audio_stim', 'context',
                       'outcome', 'opto', 'date', 'mouse']
        task_df = task_df[column_order]
        
        # Get trialized movement data
        raw_trialized_data = {
            'X_pos': session_data['movement']['trialized']['xpos'],
            'Y_pos': session_data['movement']['trialized']['ypos'],
            'View': session_data['movement']['trialized']['view'],
            'X_velocity': session_data['movement']['trialized']['xvel'],
            'Y_velocity': session_data['movement']['trialized']['yvel'],
            'time_points': session_data['movement']['trialized']['time_points']
        }
        
        # Process movement data into binned format (features x 100)
        XPos, YPos, View, XVel, YVel = self.process_movement_data(raw_trialized_data, num_samples=200)
        
        # Package the processed data into a dictionary
        trialized_data = {
            'X_pos': XPos,
            'Y_pos': YPos,
            'View': View,
            'X_velocity': XVel,
            'Y_velocity': YVel
        }
        
        return task_df, trialized_data

    def process_movement_data(self, raw_trialized_data, num_samples=200):
        """
        Process raw movement data into binned format.
        
        Args:
            raw_trialized_data (dict): Dictionary containing raw movement data
            num_samples (int): Number of samples to extract per trial
        """
        # Store the raw trial data for use in round_act
        self.trial_data = raw_trialized_data
        
        # Process each movement variable
        XPos = self.round_act(raw_trialized_data['X_pos'], num_samples)
        YPos = self.round_act(raw_trialized_data['Y_pos'], num_samples)
        View = self.round_act(raw_trialized_data['View'], num_samples)
        XVel = self.round_act(raw_trialized_data['X_velocity'], num_samples)
        YVel = self.round_act(raw_trialized_data['Y_velocity'], num_samples)
        
        return XPos, YPos, View, XVel, YVel

    def round_act(self, matrix, num_elems):
        """
        Subsample elements from each trial with weighted sampling around Y-position threshold.
        
        Args:
            matrix: Trial data matrix
            num_elems: Number of elements to sample per trial
            
        Returns:
            np.ndarray: Subsampled data
        """
        act_rnd_all = []
        for trial in range(matrix.shape[0]):
            x = matrix[trial]
            valid_mask = ~np.isnan(x)
            len_act = np.sum(valid_mask)
            
            if len_act > 0:
                # Get corresponding Y-positions for this trial
                y_pos = self.trial_data['Y_pos'][trial][valid_mask]
                x = x[valid_mask]  # Only use valid data points
                
                # Find index where Y-position crosses 500
                threshold_idx = np.where(y_pos >= 500)[0]
                threshold_idx = threshold_idx[0] if len(threshold_idx) > 0 else len_act
                
                # Calculate number of samples for each segment (80% before, 20% after)
                n_samples_before = int(0.8 * num_elems)
                n_samples_after = num_elems - n_samples_before
                
                # Generate indices with replacement if necessary
                if threshold_idx > 0:
                    before_indices = np.round(np.linspace(0, threshold_idx-1, min(n_samples_before, threshold_idx))).astype(int)
                    if len(before_indices) < n_samples_before:
                        # Add additional samples with replacement
                        additional = np.random.choice(before_indices, n_samples_before - len(before_indices), replace=True)
                        before_indices = np.concatenate([before_indices, additional])
                else:
                    before_indices = np.array([], dtype=int)
                    
                remaining_points = len_act - threshold_idx
                if remaining_points > 0:
                    after_indices = np.round(np.linspace(threshold_idx, len_act-1, min(n_samples_after, remaining_points))).astype(int)
                    if len(after_indices) < n_samples_after:
                        # Add additional samples with replacement
                        additional = np.random.choice(after_indices, n_samples_after - len(after_indices), replace=True)
                        after_indices = np.concatenate([after_indices, additional])
                else:
                    after_indices = np.array([], dtype=int)
                
                # Combine indices
                rnd = np.concatenate([before_indices, after_indices])
                
                # If we still don't have enough samples, pad with random samples from all available data
                if len(rnd) < num_elems:
                    missing_samples = num_elems - len(rnd)
                    padding_indices = np.random.choice(len(x), size=missing_samples, replace=True)
                    rnd = np.concatenate([rnd, padding_indices])
                
                act_rnd_all.append(x[rnd])
            else:
                act_rnd_all.append(np.full(num_elems, np.nan))
            
        return np.array(act_rnd_all, dtype=np.float32)
    
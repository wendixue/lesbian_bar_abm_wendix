import mesa
import random
import numpy as np

# Define three identity groups
IDENTITY_GROUPS = ["QW", "NQW", "QNW"]

# Base belonging matrix - represents belonging relationships between different groups
## Need to adjust later based on further literature exploration
BASE_BELONGING_MATRIX = {
    "QW":   {"QW": 1.0, "NQW": 0.3, "QNW": 0.3},
    "NQW":  {"QW": 0.8, "NQW": 1.0, "QNW": 0.1},
    "QNW":  {"QW": 0.8, "NQW": 0.3, "QNW": 1.0}
}
# Standard deviation for normal distribution of belonging values
BELONGING_STD_DEV = 0.3

# Create Bar class to represent bars
class Bar:
    def __init__(self, unique_id, fixed_affinity, name=None, gamma=0.5, adaptive_update_interval=10, affinity_factor=0.01):
        self.unique_id = unique_id
        self.name = name
        self.fixed_affinity = fixed_affinity
        self.adaptive_affinity = {group: 0 for group in IDENTITY_GROUPS} 
        self.gamma = gamma  # Weight for fixed affinity
        
        # Bar visitor history
        self.visitor_history = []
        self.current_visitors = []
        self.update_count = 0  
        
        # Parameters for adaptive affinity
        self.adaptive_update_interval = adaptive_update_interval  # Update every X rounds
        self.affinity_factor = affinity_factor  # Multiplier for visitor count
        
        # Bar cultural state
        self.is_lesbian_bar = True  # Initial state as a lesbian bar by default
        self.qw_ratio_history = []  # Record of QW ratio history
        
    def calculate_effective_affinity(self):
        effective = {}
        for group in IDENTITY_GROUPS:
            effective[group] = (self.gamma * self.fixed_affinity[group] + 
                              (1 - self.gamma) * self.adaptive_affinity[group])
        return effective
    
    def update_adaptive_affinity(self, force=False):
        self.update_count += 1
        
        # Update adaptive affinity every X rounds
        if self.update_count >= self.adaptive_update_interval or force:
            if len(self.visitor_history) > 0:
                # Calculate average visitor counts for each group over last X rounds
                # (or fewer if not enough history yet)
                history_range = min(len(self.visitor_history), self.adaptive_update_interval)
                recent_history = self.visitor_history[-history_range:]
                
                # Count visitors for each group
                visitor_counts = {group: 0 for group in IDENTITY_GROUPS}
                total_visitors = 0
                for visitors in recent_history:
                    total_visitors += len(visitors)
                    for visitor in visitors:
                        visitor_counts[visitor] += 1
                
                # Calculate average visitors per round for each group
                avg_counts = {group: visitor_counts[group] / history_range for group in IDENTITY_GROUPS}
                
                # Update adaptive affinity directly based on average count * factor
                for group in IDENTITY_GROUPS:
                    self.adaptive_affinity[group] = min(1.0, avg_counts[group] * self.affinity_factor)
            
            # Check if still a lesbian bar 
            # (Average QW ratio below 30% for 10 rounds is considered de-lesbianization)
            ## May adjust later
            if len(self.visitor_history) > 10:  
                recent_qw_ratios = []
                
                for visitors in self.visitor_history[-10:]:
                    if visitors and len(visitors) > 0: 
                        qw_count = sum(1 for v in visitors if v == "QW")
                        qw_ratio = qw_count / len(visitors)
                        recent_qw_ratios.append(qw_ratio)
                
                if recent_qw_ratios:
                    avg_qw_ratio = sum(recent_qw_ratios) / len(recent_qw_ratios)
                    self.qw_ratio_history.append(avg_qw_ratio)
                    
                    if avg_qw_ratio < 0.3:
                        self.is_lesbian_bar = False
            
            self.update_count = 0  # Reset counter
    
    def add_visitors(self, visitors):

        self.current_visitors.extend(visitors)
    
    def end_round(self):
        self.visitor_history.append(self.current_visitors.copy())
        self.update_adaptive_affinity()
    
    def get_current_population_ratios(self):

        if not self.current_visitors:
            return {group: 0.0 for group in IDENTITY_GROUPS}
        
        counts = {group: 0 for group in IDENTITY_GROUPS}
        for visitor in self.current_visitors:
            counts[visitor] += 1
        
        total = len(self.current_visitors)
        return {group: counts[group] / total for group in IDENTITY_GROUPS}

# Create PersonAgent class
class PersonAgent:
    def __init__(self, unique_id, model, identity_group, threshold=0.7, memory_length=5):

        self.unique_id = unique_id
        self.exit_attempts = 0  
        self.permanent_exit = False  
        self.model = model
        self.identity_group = identity_group
        self.threshold = threshold  
        self.memory = {}  
        self.memory_length = memory_length
        self.current_bar = None  
        self.status = "active"  
        self.exit_counter = 0  
        self.cooldown_duration = 10
        self.belonging_matrix = self.generate_belonging_matrix()
    
        for bar_id in range(model.num_bars):
            self.memory[bar_id] = []
    
    def generate_belonging_matrix(self):

        personal_matrix = {}
        
        # Seed the random generator with the agent ID for reproducibility
        rng = np.random.RandomState(self.unique_id)
        
        # Generate values for all group combinations
        for from_group in IDENTITY_GROUPS:
            personal_matrix[from_group] = {}
            for to_group in IDENTITY_GROUPS:
                # Use group pair mean and sample from normal distribution
                mean = BASE_BELONGING_MATRIX[from_group][to_group]
                value = rng.normal(mean, BELONGING_STD_DEV)
                # Clamp between 0 and 2 (allow values above 1)
                value = max(0.0, min(2.0, value))
                personal_matrix[from_group][to_group] = value
        
        return personal_matrix
    
    def calculate_belonging(self, bar):

        if self.status == "temp_exited" or self.status == "permanently_exited" or self.permanent_exit:
            return 0.0
        
        alpha = self.model.alpha  # Weight of structural inclusion
        
        # Get the bar’s current affinity for the agent's group
        effective_affinity = bar.calculate_effective_affinity()
        bar_affinity = effective_affinity[self.identity_group]
        
        # Compute influence of peer group composition
        population_ratios = bar.get_current_population_ratios()
        social_belonging = 0.0
        
        for other_group in IDENTITY_GROUPS:
            # Get personal coefficient toward each group
            group_belonging = self.belonging_matrix[self.identity_group][other_group]
            # Multiply by population share
            social_belonging += group_belonging * population_ratios[other_group]
        
        # Combine structural and social components
        total_belonging = (alpha * bar_affinity) + ((1 - alpha) * social_belonging)
        
        return total_belonging
    
    def update_memory(self, bar_id, belonging_score):

        self.memory[bar_id].append(belonging_score)
        
        # Keep only most recent records
        if len(self.memory[bar_id]) > self.memory_length:
            self.memory[bar_id].pop(0)
    
    def clear_memories(self):

        for bar_id in range(self.model.num_bars):
            self.memory[bar_id] = []
    
    def choose_bar(self):

        if self.status == "permanently_exited" or self.permanent_exit:
            return None
            
        if self.status == "temp_exited":
            self.exit_counter += 1

            if self.exit_counter >= self.cooldown_duration:
                if self.exit_attempts >= 3:  
                    self.status = "permanently_exited"
                    self.permanent_exit = True
                    return None
                else:
                    self.status = "active"
                    self.exit_counter = 0
                    self.clear_memories()
            else:
                return None

        current_step = getattr(self.model.schedule, 'steps', 0)
        
        # During initial steps or after memory reset: choose based on bar affinity
        if current_step < 5 or all(len(self.memory[bar_id]) == 0 for bar_id in range(self.model.num_bars)):
            # Calculate affinity score for each bar
            bar_affinities = []
            for bar_id, bar in enumerate(self.model.bars):
                effective_affinity = bar.calculate_effective_affinity()
                affinity = effective_affinity[self.identity_group]
                bar_affinities.append((bar_id, affinity))
            
            # Choose a bar weighted by Affinity
            weights = [tol for _, tol in bar_affinities]
            bar_ids = [bid for bid, _ in bar_affinities]
            
            if sum(weights) > 0:
                chosen_bar = random.choices(bar_ids, weights=weights, k=1)[0]
                return chosen_bar
            else:
                # If all weights are zero, pick randomly
                chosen_bar = random.choice(range(self.model.num_bars))
                return chosen_bar
        
        # After initial rounds: choose based on average remembered belonging
        avg_belongings = {}
        valid_bars = []
        
        for bar_id in range(self.model.num_bars):
            if self.memory[bar_id]:
                avg_belonging = sum(self.memory[bar_id]) / len(self.memory[bar_id])
                avg_belongings[bar_id] = avg_belonging
                
                # Bar is considered valid if average exceeds threshold
                if avg_belonging >= self.threshold:
                    valid_bars.append(bar_id)
        
        # No valid bar found: temporarily exit
        if not valid_bars:
            self.status = "temp_exited"
            self.exit_counter = 0
            self.exit_attempts += 1
            return None
        
        # Choose among valid bars based on remembered belonging
        if valid_bars:
            valid_belongings = {bar_id: avg_belongings[bar_id] for bar_id in valid_bars}
            total_belonging = sum(valid_belongings.values())
            
            if total_belonging > 0:
                probs = [valid_belongings[bar_id] / total_belonging for bar_id in valid_bars]
                chosen_bar = random.choices(valid_bars, weights=probs, k=1)[0]
                return chosen_bar
            else:
                chosen_bar = random.choice(valid_bars)
                return chosen_bar
        
        return None
    
    def step(self):
        pass

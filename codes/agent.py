import mesa
import random
import numpy as np


# Create Bar class to represent bars
class Bar:
    def __init__(self, fixed_affinity, name=None, gamma=0.5, adaptive_update_interval=10):
        self.name = name
        self.fixed_affinity = fixed_affinity
        self.gamma = gamma
        self.visitor_history = []
        self.current_visitors = []
        self.update_count = 0  

        # Initialize adaptive affinity to match fixed affinity proportions
        total_fixed = sum(fixed_affinity.values())
        self.adaptive_affinity = {group: affinity / total_fixed 
                for group, affinity in fixed_affinity.items()}
        
        # Update every X rounds (10 by default)
        self.adaptive_update_interval = adaptive_update_interval
        
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
                # Calculate average visitor ratios for each group over last X rounds
                history_range = min(len(self.visitor_history), self.adaptive_update_interval)
                recent_history = self.visitor_history[-history_range:]
                
                # Count visitors for each group and total visitors
                visitor_counts = {group: 0 for group in IDENTITY_GROUPS}
                total_visitors = 0
                for visitors in recent_history:
                    total_visitors += len(visitors)
                    for visitor in visitors:
                        visitor_counts[visitor] += 1
                
                # Calculate average visitor ratios
                if total_visitors > 0:
                    avg_ratios = {group: visitor_counts[group] / total_visitors for group in IDENTITY_GROUPS}
                else:
                    avg_ratios = {group: 0.0 for group in IDENTITY_GROUPS}
                
                # Update adaptive affinity based on historical average ratios
                # adaptive_affinity = 1.0 * historical_average_ratio for each group
                for group in IDENTITY_GROUPS:
                    self.adaptive_affinity[group] = avg_ratios[group]
            
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


# Define three identity groups
IDENTITY_GROUPS = ["QW", "NQW", "QNW"]

# Base belonging matrix - represents belonging relationships between different groups
BASE_BELONGING_MATRIX = {
    "QW":   {"QW": 1.0, "NQW": 0.3, "QNW": 0.3},
    "NQW":  {"QW": 0.8, "NQW": 1.0, "QNW": 0.1},
    "QNW":  {"QW": 0.8, "NQW": 0.3, "QNW": 1.0}
}
# Standard deviation for normal distribution of belonging values
BELONGING_STD_DEV = 0.1

# Create PersonAgent class
class PersonAgent(mesa.Agent):
    def __init__(self, model, identity_group, threshold=0.5):
        super().__init__(model)  
        self.identity_group = identity_group
        self.threshold = threshold  
        self.last_bar_scores = {} 
        self.current_bar = None  
        self.status = "active"  
        self.exit_counter = 0  
        self.cooldown_duration = model.random.randint(5, 15)
        self.exit_attempts = 0  
        self.permanent_exit = False  
        self.belonging_matrix = self.generate_belonging_matrix()
    
        # Initialize last scores for all bars as None
        for bar_id in range(2):
            self.last_bar_scores[bar_id] = None
    
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
                # Clamp between 0 and 1
                value = max(0.0, min(1.0, value))
                personal_matrix[from_group][to_group] = value
        
        return personal_matrix
    
    def calculate_belonging(self, bar):
        if self.status == "temp_exited" or self.status == "permanently_exited" or self.permanent_exit:
            return 0.0
        
        alpha = self.model.alpha  # Weight of structural inclusion
        
        # Get the bar's current affinity for the agent's group
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
    
    def update_last_score(self, bar_id, belonging_score):
        self.last_bar_scores[bar_id] = belonging_score
    
    def choose_bar(self):
        if self.status == "permanently_exited" or self.permanent_exit:
            return None
            
        if self.status == "temp_exited":
            self.exit_counter += 1

            if self.exit_counter >= self.cooldown_duration:
                if self.exit_attempts >= 2: 
                    self.status = "permanently_exited"
                    self.permanent_exit = True
                    return None
                else:
                    self.status = "active"
                    self.exit_counter = 0
                    # Clear last scores when returning from temp exit
                    for bar_id in range(2):
                        self.last_bar_scores[bar_id] = None
            else:
                return None

        current_step = self.model.steps
        
        # During initial steps or when no previous scores: choose based on bar affinity
        if current_step < 5 or all(score is None for score in self.last_bar_scores.values()):
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
                chosen_bar = random.choice(range(2))
                return chosen_bar
        
        # After initial rounds: choose based on last belonging scores
        valid_bars = []
        
        for bar_id in range(2):
            last_score = self.last_bar_scores[bar_id]
            if last_score is not None and last_score >= self.threshold:
                valid_bars.append(bar_id)
        
        # No valid bar found: temporarily exit
        if not valid_bars:
            self.status = "temp_exited"
            self.exit_counter = 0
            self.exit_attempts += 1
            return None
        
        # Choose among valid bars based on last belonging scores
        if valid_bars:
            valid_scores = {bar_id: self.last_bar_scores[bar_id] for bar_id in valid_bars}
            total_score = sum(valid_scores.values())
            
            if total_score > 0:
                probs = [valid_scores[bar_id] / total_score for bar_id in valid_bars]
                chosen_bar = random.choices(valid_bars, weights=probs, k=1)[0]
                return chosen_bar
            else:
                chosen_bar = random.choice(valid_bars)
                return chosen_bar
        
        return None
    
    def step(self):
        pass
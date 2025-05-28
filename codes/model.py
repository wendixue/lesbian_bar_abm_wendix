import mesa
from mesa.datacollection import DataCollector
from agent import IDENTITY_GROUPS, BASE_BELONGING_MATRIX, Bar, PersonAgent
from mesa.visualization.utils import force_update
import numpy as np

class LGBTQBarModel(mesa.Model):
    def __init__(self, 
                 population_size=200, 
                 alpha=0.5,
                 gamma=0.5, 
                 init_identity_ratios=None,
                 QW_ratio=0.4,
                 QNW_ratio=0.3,
                 adaptive_update_interval=10,
                 seed=None):

        super().__init__(seed=seed)
        self.num_agents = population_size
        self.alpha = alpha  # Weight of bar affinity in belonging calculation
        self.gamma = gamma  # Learning rate for adaptive affinity updates
        self.running = True
        self.agent_threshold = 0.55
        
        # Store for synchronization
        self.bar_choices = {}  # Store each agent's choice {agent_id: bar_id}
        
        # Set initial identity group ratios
        if init_identity_ratios is None:
            NQW_ratio = 1.0 - QW_ratio - QNW_ratio
            
            init_identity_ratios = {
                "QW": QW_ratio,
                "NQW": NQW_ratio,
                "QNW": QNW_ratio
            }
        
        # Create the two bars with fixed configurations
        # Women-only bar
        women_only_bar_affinity = {
            "QW": 1.0,    # Fully welcome Queer Women
            "NQW": 0.7,   # Mostly welcome Non-Queer Women
            "QNW": 0.2    # not welcome Queer Non-Women
        }
        
        # Queer-friendly bar
        queer_friendly_bar_affinity = {
            "QW": 1.0,    # Fully welcome Queer Women
            "NQW": 0.2,   # not welcome Non-Queer Women
            "QNW": 0.7    # msostly welcome Queer Non-Women
        }
        
        # Create the two bars
        self.women_bar = Bar(women_only_bar_affinity, name="women_only_bar", 
                            adaptive_update_interval=adaptive_update_interval, gamma=self.gamma)
        self.queer_bar = Bar(queer_friendly_bar_affinity, name="queer_friendly_bar", 
                            adaptive_update_interval=adaptive_update_interval, gamma=self.gamma)
        
        # Keep bars list for compatibility with existing code
        self.bars = [self.women_bar, self.queer_bar]
        
        # Create Agents
        for i in range(self.num_agents):
            # Assign identity group based on ratios
            r = self.random.random()
            cumulative = 0
            assigned_group = None
            
            for group, ratio in init_identity_ratios.items():
                cumulative += ratio
                if r <= cumulative:
                    assigned_group = group
                    break
            
            # Setting individual thresholds
            base_threshold = self.agent_threshold 
            threshold = self.random.uniform(base_threshold-0.15, base_threshold+0.15)
            
            # Create Agent
            agent = PersonAgent(self, assigned_group, threshold)
            
        # Set data collector with simplified bar references
        model_reporters = {
            "WomenBar_QW_Ratio": lambda m: self.get_bar_group_ratio(0, "QW"),
            "WomenBar_NQW_Ratio": lambda m: self.get_bar_group_ratio(0, "NQW"),
            "WomenBar_QNW_Ratio": lambda m: self.get_bar_group_ratio(0, "QNW"),
            "WomenBar_Population": lambda m: self.get_bar_population(0),
            "WomenBar_QW_AdaptiveAffinity": lambda m: self.women_bar.adaptive_affinity["QW"],
            "WomenBar_NQW_AdaptiveAffinity": lambda m: self.women_bar.adaptive_affinity["NQW"],
            "WomenBar_QNW_AdaptiveAffinity": lambda m: self.women_bar.adaptive_affinity["QNW"],
            "WomenBar_QW_EffectiveAffinity": lambda m: self.women_bar.calculate_effective_affinity()["QW"],
            "QueerBar_QW_Ratio": lambda m: self.get_bar_group_ratio(1, "QW"),
            "QueerBar_NQW_Ratio": lambda m: self.get_bar_group_ratio(1, "NQW"),
            "QueerBar_QNW_Ratio": lambda m: self.get_bar_group_ratio(1, "QNW"),
            "QueerBar_Population": lambda m: self.get_bar_population(1),
            "QueerBar_QW_AdaptiveAffinity": lambda m: self.queer_bar.adaptive_affinity["QW"],
            "QueerBar_NQW_AdaptiveAffinity": lambda m: self.queer_bar.adaptive_affinity["NQW"],
            "QueerBar_QNW_AdaptiveAffinity": lambda m: self.queer_bar.adaptive_affinity["QNW"],
            "QueerBar_QW_EffectiveAffinity": lambda m: self.queer_bar.calculate_effective_affinity()["QW"],
            "TempExited_Agents": lambda m: self.count_temp_exited_agents(),
            "PermExited_Agents": lambda m: self.count_permanently_exited_agents(),  
            "Active_QW": lambda m: self.count_active_by_group("QW"),
            "Active_NQW": lambda m: self.count_active_by_group("NQW"),
            "Active_QNW": lambda m: self.count_active_by_group("QNW")
        }
            
        self.datacollector = DataCollector(model_reporters=model_reporters)
        
    def get_bar_group_ratio(self, bar_id, group):
        bar = self.bars[bar_id]
        ratios = bar.get_current_population_ratios()
        return ratios[group]
    
    def get_bar_population(self, bar_id):
        return len(self.bars[bar_id].current_visitors)
    
    def count_temp_exited_agents(self):
        return sum(1 for agent in self.agents if agent.status == "temp_exited")
    
    def count_permanently_exited_agents(self):
        return sum(1 for agent in self.agents 
                  if agent.status == "permanently_exited" or agent.permanent_exit)
    
    def count_active_by_group(self, group):
        return sum(1 for agent in self.agents 
                  if agent.identity_group == group and agent.status == "active")
    
    def get_average_belonging_matrix(self):
        # Initialize average matrix with zeros
        sum_matrix = {}
        for from_group in IDENTITY_GROUPS:
            sum_matrix[from_group] = {}
            for to_group in IDENTITY_GROUPS:
                sum_matrix[from_group][to_group] = 0.0
        
        # Count agents by group
        group_counts = {group: 0 for group in IDENTITY_GROUPS}
        
        # Sum up all belonging values
        for agent in self.agents:
            group = agent.identity_group
            group_counts[group] += 1
            
            for to_group in IDENTITY_GROUPS:
                sum_matrix[group][to_group] += agent.belonging_matrix[group][to_group]
        
        # Calculate averages
        avg_matrix = {}
        for from_group in IDENTITY_GROUPS:
            avg_matrix[from_group] = {}
            for to_group in IDENTITY_GROUPS:
                if group_counts[from_group] > 0:
                    avg_matrix[from_group][to_group] = sum_matrix[from_group][to_group] / group_counts[from_group]
                else:
                    avg_matrix[from_group][to_group] = 0.0
        
        return avg_matrix
    
    def get_belonging_matrix_stats(self):
        # Initialize data structure to collect all values
        all_values = {}
        for from_group in IDENTITY_GROUPS:
            all_values[from_group] = {}
            for to_group in IDENTITY_GROUPS:
                all_values[from_group][to_group] = []
        
        # Collect all belonging values
        for agent in self.agents:
            group = agent.identity_group
            for to_group in IDENTITY_GROUPS:
                all_values[group][to_group].append(agent.belonging_matrix[group][to_group])
        
        # Calculate statistics
        stats = {}
        for from_group in IDENTITY_GROUPS:
            stats[from_group] = {}
            for to_group in IDENTITY_GROUPS:
                values = all_values[from_group][to_group]
                if values:
                    stats[from_group][to_group] = {
                        "min": min(values),
                        "max": max(values),
                        "mean": np.mean(values),
                        "std": np.std(values),
                        "base": BASE_BELONGING_MATRIX[from_group][to_group]
                    }
                else:
                    stats[from_group][to_group] = {
                        "min": 0.0,
                        "max": 0.0,
                        "mean": 0.0,
                        "std": 0.0,
                        "base": BASE_BELONGING_MATRIX[from_group][to_group]
                    }
        
        return stats
    
    def step(self):
        # Clear current visitors from both bars
        self.women_bar.current_visitors = []
        self.queer_bar.current_visitors = []

        def agent_step(agent):
            if agent.status == "permanently_exited" or agent.permanent_exit:
                return
                
            # Agent chooses bar
            chosen_bar_id = agent.choose_bar()
            if chosen_bar_id is not None:
                # Agent enters bar immediately
                agent.current_bar = chosen_bar_id
                bar = self.bars[chosen_bar_id]
                bar.add_visitors([agent.identity_group])
                
                # Calculate and update belonging score
                belonging_score = agent.calculate_belonging(bar)
                agent.update_last_score(chosen_bar_id, belonging_score)
        
        self.agents.do(agent_step)
        
        # End current round for both bars
        self.women_bar.end_round()
        self.queer_bar.end_round()
        
        # Collect data
        self.datacollector.collect(self)
        
        # Force update visualization
        force_update()
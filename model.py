import mesa
from mesa.datacollection import DataCollector
from agent import IDENTITY_GROUPS, BASE_BELONGING_MATRIX, Bar, PersonAgent
from mesa.visualization.utils import force_update
import numpy as np

class CustomActivation:
    """Custom activator for synchronized agent updates"""
    def __init__(self, model):
        self.model = model
        self.agents = []
        self.steps = 0
    
    def add(self, agent):
        self.agents.append(agent)
    
    def step(self):
   
        # Randomly shuffle agent order (for action order, but state updates are synchronized)
        shuffled = self.agents.copy()
        self.model.random.shuffle(shuffled)
        
        # Phase 1: All agents choose bars but don't enter immediately
        bar_choices = {}  # Store each agent's choice {agent_id: bar_id}
        
        print(f"Phase 1: {len(shuffled)} agents choose bars")
        for agent in shuffled:
            # Skip permanently exited agents
            if agent.status == "permanently_exited" or agent.permanent_exit:
                continue
                
            # Process temporary exit status internally in choose_bar method
            chosen_bar_id = agent.choose_bar()  # Choose only, don't enter
            if chosen_bar_id is not None:
                bar_choices[agent.unique_id] = chosen_bar_id
        
        # Phase 2: All agents enter their chosen bars simultaneously
        print(f"Phase 2: Agents enter bars simultaneously")
        for agent in self.agents:
            # Skip permanently exited or temporarily exited agents
            if agent.status == "permanently_exited" or agent.permanent_exit or agent.status == "temp_exited" or agent.unique_id not in bar_choices:
                continue
                
            chosen_bar_id = bar_choices[agent.unique_id]
            agent.current_bar = chosen_bar_id
            
            # Add self to bar visitor list
            bar = self.model.bars[chosen_bar_id]
            bar.add_visitors([agent.identity_group])
        
        # Phase 3: Calculate belonging and update memories
        print(f"Phase 3: Calculate belonging and update memories")
        for agent in self.agents:
            # Skip permanently exited or temporarily exited agents
            if agent.status == "permanently_exited" or agent.permanent_exit or agent.status == "temp_exited" or agent.unique_id not in bar_choices:
                continue
                
            chosen_bar_id = bar_choices[agent.unique_id]
            bar = self.model.bars[chosen_bar_id]
            
            # Calculate and update belonging memory for this bar
            belonging_score = agent.calculate_belonging(bar)
            agent.update_memory(chosen_bar_id, belonging_score)
            
            print(f"Agent {agent.unique_id} ({agent.identity_group}) entered {bar.name}, belonging: {belonging_score:.2f}")
        
        # Increment step counter
        self.steps += 1
        
        # Force update visualization
        force_update()

class LGBTQBarModel(mesa.Model):
    def __init__(self, 
                 population_size=200, 
                 num_bars=2, 
                 alpha=0.4,
                 init_identity_ratios=None,
                 QW_ratio=0.5,
                 NQW_ratio=0.25,
                 QNW_ratio=0.25,
                 adaptive_update_interval=10,
                 tolerance_factor=0.01,
                 agent_threshold=0.5,  
                 seed=None):

        super().__init__(seed=seed)
        self.num_agents = population_size
        self.num_bars = num_bars
        self.alpha = alpha  # Weight of bar tolerance in belonging calculation
        self.schedule = CustomActivation(self)  # Use custom activator
        self._agent_storage = self.schedule  
        self.running = True  
        self.agent_threshold = agent_threshold  
        
        # Set initial identity group ratios
        if init_identity_ratios is None:
            # Use individual ratio parameters
            total = QW_ratio + NQW_ratio + QNW_ratio
            if total == 0:
                # Avoid division by zero
                init_identity_ratios = {
                    "QW": 0.33,
                    "NQW": 0.33,
                    "QNW": 0.34
                }
            else:
                # Normalize ratios to ensure sum is 1
                init_identity_ratios = {
                    "QW": QW_ratio / total,
                    "NQW": NQW_ratio / total,
                    "QNW": QNW_ratio / total
                }
        
        # Set default bar fixed tolerances
        # First bar: Women-only bar
        women_only_bar_tolerance = {
            "QW": 1.0,    # Fully welcome Queer Women
            "NQW": 0.7,   # Mostly welcome Non-Queer Women
            "QNW": 0.2    # Moderately welcome Queer Non-Women
        }
        
        # Second bar: Queer-friendly bar
        queer_friendly_bar_tolerance = {
            "QW": 1.0,    # Mostly welcome Queer Women
            "NQW": 0.2,   # Mostly welcome Non-Queer Women
            "QNW": 0.7    # Mostly welcome Queer Non-Women
        }
        
        bar_fixed_tolerances = [women_only_bar_tolerance, queer_friendly_bar_tolerance]
        
        # Create bars
        self.bars = []
        bar_names = ["women_only_bar", "queer_friendly_bar"]
        
        for i in range(num_bars):
            name = bar_names[i] if i < len(bar_names) else f"Bar {i+1}"
            # Adaptive Tolerance Parameter Passed When Creating Bar
            bar = Bar(i, bar_fixed_tolerances[i], name=name, 
                      adaptive_update_interval=adaptive_update_interval,
                      tolerance_factor=tolerance_factor)
            self.bars.append(bar)
        
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
            ## Needs to be adjusted! (the range? the variation?)
            base_threshold = agent_threshold
            threshold_variation = 0.15 
            threshold = self.random.uniform(
                max(0.5, base_threshold - threshold_variation),
                min(0.9, base_threshold + threshold_variation)
            )
            
            # Create Agent
            agent = PersonAgent(i, self, assigned_group, threshold)
            self.schedule.add(agent)
            
            # Randomly initialize bar memories
            for bar_id in range(num_bars):
                # Add some random initial memories to promote initial movement
                initial_score = self.random.uniform(0.4, 0.9)
                agent.update_memory(bar_id, initial_score)
        
        # Set data collector
        model_reporters = {
            "WomenBar_QW_Ratio": lambda m: self.get_bar_group_ratio(0, "QW"),
            "WomenBar_NQW_Ratio": lambda m: self.get_bar_group_ratio(0, "NQW"),
            "WomenBar_QNW_Ratio": lambda m: self.get_bar_group_ratio(0, "QNW"),
            "WomenBar_Population": lambda m: self.get_bar_population(0),
            "WomenBar_IsLesbianBar": lambda m: int(self.bars[0].is_lesbian_bar),
            "WomenBar_QW_AdaptiveTolerance": lambda m: self.bars[0].adaptive_tolerance["QW"],
            "WomenBar_NQW_AdaptiveTolerance": lambda m: self.bars[0].adaptive_tolerance["NQW"],
            "WomenBar_QNW_AdaptiveTolerance": lambda m: self.bars[0].adaptive_tolerance["QNW"],
            "QueerBar_QW_Ratio": lambda m: self.get_bar_group_ratio(1, "QW"),
            "QueerBar_NQW_Ratio": lambda m: self.get_bar_group_ratio(1, "NQW"),
            "QueerBar_QNW_Ratio": lambda m: self.get_bar_group_ratio(1, "QNW"),
            "QueerBar_Population": lambda m: self.get_bar_population(1),
            "QueerBar_IsLesbianBar": lambda m: int(self.bars[1].is_lesbian_bar),
            "QueerBar_QW_AdaptiveTolerance": lambda m: self.bars[1].adaptive_tolerance["QW"],
            "QueerBar_NQW_AdaptiveTolerance": lambda m: self.bars[1].adaptive_tolerance["NQW"],
            "QueerBar_QNW_AdaptiveTolerance": lambda m: self.bars[1].adaptive_tolerance["QNW"],
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

        return sum(1 for agent in self._agent_storage.agents if agent.status == "temp_exited")
    
    def count_permanently_exited_agents(self):

        return sum(1 for agent in self._agent_storage.agents 
                  if agent.status == "permanently_exited" or agent.permanent_exit)
    
    def count_active_by_group(self, group):

        return sum(1 for agent in self._agent_storage.agents 
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
        for agent in self._agent_storage.agents:
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
        for agent in self._agent_storage.agents:
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

        # Clear current visitors from bars
        for bar in self.bars:
            bar.current_visitors = []
        
        # Print information (for debugging)
        print(f"\n=== Executing step {self.schedule.steps if hasattr(self.schedule, 'steps') else '?'} ===")
        print(f"Current active agents: {len([a for a in self._agent_storage.agents if a.status == 'active'])}")
        print(f"Temporarily exited agents: {self.count_temp_exited_agents()}")
        print(f"Permanently exited agents: {self.count_permanently_exited_agents()}")
        
        # Execute all agents' steps
        self.schedule.step()
        
        # End current round for bars
        for bar in self.bars:
            bar.end_round()
        
        # Collect data
        self.datacollector.collect(self)
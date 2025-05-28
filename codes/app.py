from model import LGBTQBarModel
from mesa.visualization import Slider, SolaraViz
import solara
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import random
from mesa.visualization.utils import update_counter, force_update

# Create agent map component
@solara.component
def AgentMapComponent(model):

    update_counter.get()
    
    fig = Figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    
    # Set fixed coordinates for bar and exit zones
    bar_positions = [(0.3, 0.5), (0.7, 0.5)] 
    temp_exit_position = (0.5, 0.2)  
    perm_exit_position = (0.5, 0.8)
    
    # Draw bar positions
    for i, pos in enumerate(bar_positions):
        if i < len(model.bars):
            ax.scatter(pos[0], pos[1], s=300, color='gray', alpha=0.5, marker='s')
            ax.text(pos[0], pos[1], model.bars[i].name, ha='center', va='center', fontsize=13)
    
    # Draw temporary exit zone
    ax.scatter(temp_exit_position[0], temp_exit_position[1], s=300, color='orange', alpha=0.2, marker='s')
    ax.text(temp_exit_position[0], temp_exit_position[1], 'Temp Exit (5-15 rounds)', ha='center', va='center', fontsize=13)
    
    # Draw permanent exit zone
    ax.scatter(perm_exit_position[0], perm_exit_position[1], s=300, color='red', alpha=0.2, marker='s')
    ax.text(perm_exit_position[0], perm_exit_position[1], 'Permanent Exit', ha='center', va='center', fontsize=13)
    
    # Define colors for each identity group
    colors = {
        "QW": "red",
        "NQW": "pink", 
        "QNW": "purple"
    }
    
    # Count agents in each area
    temp_count = 0
    perm_count = 0
    bar_counts = [0, 0]
    
    # Draw each agent
    for agent in model.agents:
        # Determine agent position
        if agent.status == "permanently_exited" or agent.permanent_exit:
            # If permanently exited, place randomly near permanent exit zone
            x = perm_exit_position[0] + (random.random() - 0.5) * 0.2
            y = perm_exit_position[1] + (random.random() - 0.5) * 0.2
            perm_count += 1
        elif agent.status == "temp_exited":
            # If temporarily exited, place randomly near temporary exit zone
            x = temp_exit_position[0] + (random.random() - 0.5) * 0.2
            y = temp_exit_position[1] + (random.random() - 0.5) * 0.2
            temp_count += 1
        elif agent.current_bar is not None:
            # If in a bar, place randomly near the bar location
            bar_pos = bar_positions[agent.current_bar]
            x = bar_pos[0] + (random.random() - 0.5) * 0.2
            y = bar_pos[1] + (random.random() - 0.5) * 0.2
            if agent.current_bar < len(bar_counts):
                bar_counts[agent.current_bar] += 1
        else:
            # Skip if no valid location
            continue
        
        # Draw point using identity group color
        ax.scatter(x, y, s=50, color=colors[agent.identity_group], alpha=0.7)
    
    # Add legend
    for group, color in colors.items():
        ax.scatter([], [], color=color, label=group)
    
    ax.legend(loc='upper right')
    
    # Set axis limits and plot title
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # Show agent count summary (for debugging)
    step_text = model.steps
    title = f'Agent Distribution (Step: {step_text}, Active: {sum(bar_counts)}, Temp: {temp_count}, Perm: {perm_count})'
    ax.set_title(title)
    
    # Hide axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    return solara.FigureMatplotlib(figure=fig)

# Create bar population proportion component
@solara.component
def BarProportionTrendsComponent(model):

    update_counter.get()
    
    # Get current simulation step
    current_step = model.steps
    
    # Initialize time series history on first step
    if not hasattr(model, 'time_series_history'):
        model.time_series_history = {
            'steps': [],
            'bars': {},
        }
        # Initialize data structures for each bar
        for i in range(len(model.bars)):
            model.time_series_history['bars'][i] = {
                'QW': [],
                'NQW': [],
                'QNW': [],
                'total': []
            }
    
    # Only record once per step
    if not model.time_series_history['steps'] or model.time_series_history['steps'][-1] != current_step:
        # Record current step
        model.time_series_history['steps'].append(current_step)
        
        # Collect data per bar using visitor history
        for bar_id, bar in enumerate(model.bars):
            # Use last round of visitor history
            if bar.visitor_history and len(bar.visitor_history) > 0:
                last_visitors = bar.visitor_history[-1]
                
                # Count visitors by group
                counts = {'QW': 0, 'NQW': 0, 'QNW': 0}
                for visitor in last_visitors:
                    counts[visitor] += 1
                
                total_visitors = sum(counts.values())
                
                # Log counts and total to time series history
                for group in counts:
                    model.time_series_history['bars'][bar_id][group].append(counts[group])
                model.time_series_history['bars'][bar_id]['total'].append(total_visitors)
            else:
                # Append zero if no history
                for group in ['QW', 'NQW', 'QNW']:
                    model.time_series_history['bars'][bar_id][group].append(0)
                model.time_series_history['bars'][bar_id]['total'].append(0)
    
    # Create time series plot for each bar
    with solara.Column():
        
        for bar_id in range(len(model.bars)):
            # Get current total population
            current_total = (model.time_series_history['bars'][bar_id]['total'][-1] 
                           if model.time_series_history['bars'][bar_id]['total'] else 0)
            
            solara.Markdown(f"### {model.bars[bar_id].name} Population Proportion Trends (Current Total: {current_total})")
            
            # Create proportion trend plot
            fig = Figure(figsize=(12, 5))  # Reduced height
            ax = fig.add_subplot(111)
            
            # Plot lines for each identity group proportion
            steps = model.time_series_history['steps']
            
            # Define colors for each identity group
            colors = {
                "QW": "red",
                "NQW": "pink", 
                "QNW": "purple"
            }
            
            # Calculate and plot proportions for each group
            for group, color in colors.items():
                counts = model.time_series_history['bars'][bar_id][group]
                totals = model.time_series_history['bars'][bar_id]['total']
                
                # Calculate proportions
                proportions = []
                for i in range(len(counts)):
                    if totals[i] > 0:
                        proportions.append(counts[i] / totals[i])
                    else:
                        proportions.append(0.0)
                
                ax.plot(steps, proportions, label=group, color=color, marker='o', markersize=4)
            
            # Set plot properties
            ax.set_xlabel('Step')
            ax.set_ylabel('Proportion')
            ax.set_title(f'{model.bars[bar_id].name} Visitor Proportion by Group')
            ax.set_ylim(0, 1)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()
            
            # Set integer x-axis ticks
            if len(steps) > 1:
                ax.set_xticks(range(0, max(steps) + 1, max(1, max(steps) // 10)))
            
            plt.tight_layout()  # Ensure tight layout
            solara.FigureMatplotlib(figure=fig)
            

# Create bar visitor count trends component
@solara.component  
def BarVisitorCountTrendsComponent(model):
    update_counter.get()
    
    with solara.Column():
        solara.Markdown("### Bar Visitor Count Trends")
        
        if hasattr(model, 'time_series_history') and model.time_series_history['steps']:
            steps = model.time_series_history['steps'] 
            
            fig = Figure(figsize=(12, 5))  # Reduced height
            ax = fig.add_subplot(111)
            
            # Colors for each bar
            bar_colors = ["blue", "green"]
            
            # Plot total visitor count for each bar
            for i, bar in enumerate(model.bars):
                if 'total' in model.time_series_history['bars'][i]:
                    totals = model.time_series_history['bars'][i]['total']
                    ax.plot(steps, totals, label=f"{bar.name}", color=bar_colors[i], 
                           marker='o', markersize=4, linewidth=2)
            
            # Set plot properties
            ax.set_xlabel('Step')
            ax.set_ylabel('Total Visitor Count')
            ax.set_title('Total Visitor Count Comparison Between Bars')
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()
            
            # Set integer x-axis ticks
            if len(steps) > 1:
                ax.set_xticks(range(0, max(steps) + 1, max(1, max(steps) // 10)))
            
            plt.tight_layout()  # Ensure tight layout
            solara.FigureMatplotlib(figure=fig)


# Create effective affinity trends component
@solara.component
def EffectiveAffinityTrendsComponent(model):
    update_counter.get()
    
    # Get current simulation step
    current_step = model.steps
    
    # Initialize effective affinity history
    if not hasattr(model, 'effective_affinity_history'):
        model.effective_affinity_history = {
            'steps': [],
            'bars': {}
        }
        # Initialize for each bar
        for i in range(len(model.bars)):
            model.effective_affinity_history['bars'][i] = []
    
    # Record effective affinity once per step
    if not model.effective_affinity_history['steps'] or model.effective_affinity_history['steps'][-1] != current_step:
        model.effective_affinity_history['steps'].append(current_step)
        
        # Record effective affinity for QW in each bar
        for bar_id, bar in enumerate(model.bars):
            effective_affinity = bar.calculate_effective_affinity()
            qw_affinity = effective_affinity["QW"]
            model.effective_affinity_history['bars'][bar_id].append(qw_affinity)
    
    with solara.Column():
        solara.Markdown("### Effective Affinity for QW Trends")
        
        if model.effective_affinity_history['steps']:
            steps = model.effective_affinity_history['steps']
            
            fig = Figure(figsize=(12, 5))  # Reduced height
            ax = fig.add_subplot(111)
            
            # Colors for each bar
            bar_colors = ["blue", "green"]
            
            # Plot effective affinity for QW for each bar
            for i, bar in enumerate(model.bars):
                qw_affinities = model.effective_affinity_history['bars'][i]
                ax.plot(steps, qw_affinities, label=f"{bar.name}", color=bar_colors[i], 
                       marker='o', markersize=4, linewidth=2)
            
            # Set plot properties
            ax.set_xlabel('Step')
            ax.set_ylabel('Effective Affinity for QW')
            ax.set_title('Effective Affinity for QW Comparison Between Bars')
            ax.set_ylim(0, 1)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()
            
            # Set integer x-axis ticks
            if len(steps) > 1:
                ax.set_xticks(range(0, max(steps) + 1, max(1, max(steps) // 10)))
            
            plt.tight_layout()  # Ensure tight layout
            solara.FigureMatplotlib(figure=fig)
        

# Create bar status component
@solara.component
def BarStatusComponent(model):
    # Ensure component updates with model state
    update_counter.get()
    
    # Get current simulation step
    current_step = model.steps
    
    with solara.Column():
        # Combined QW Ratio Plot for both bars
        solara.Markdown(f"### Combined QW Ratio Comparison (Step: {current_step})")
        
        if hasattr(model, 'time_series_history') and model.time_series_history['steps']:
            steps = model.time_series_history['steps']
            
            fig = Figure(figsize=(12, 5))  # Reduced height
            ax = fig.add_subplot(111)
            
            # Colors for each bar
            bar_colors = ["blue", "green"]
            
            # Plot QW ratio for each bar
            for i, bar in enumerate(model.bars):
                qw_ratios = []
                for j in range(len(steps)):
                    try:
                        total = model.time_series_history['bars'][i]['total'][j]
                        qw_count = model.time_series_history['bars'][i]["QW"][j]
                        qw_ratio = qw_count / total if total > 0 else 0
                        qw_ratios.append(qw_ratio)
                    except (IndexError, KeyError):
                        qw_ratios.append(0)
                
                # Plot line for this bar
                ax.plot(steps, qw_ratios, label=f"{bar.name}", color=bar_colors[i], marker='o', markersize=4, linewidth=2)
            
            # Add threshold reference lines
            ax.axhline(y=0.3, color='gray', linestyle='--', alpha=0.5, label='30% threshold')
            ax.axhline(y=0.6, color='gray', linestyle='--', alpha=0.5, label='60% threshold')
            
            # Set plot properties
            ax.set_xlabel('Step')
            ax.set_ylabel('QW Ratio')
            ax.set_title('QW Ratio Comparison Between Bars')
            ax.set_ylim(0, 1)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend(loc="upper right")
            
            # Set integer x-axis ticks
            if len(steps) > 1:
                ax.set_xticks(range(0, max(steps) + 1, max(1, max(steps) // 10)))
            
            plt.tight_layout()
            solara.FigureMatplotlib(figure=fig)

model_params = {
    "seed": {
        "type": "InputText",
        "value": 42,
        "label": "Random Seed",
    },
    "population_size": Slider(
        label="Total Agents",
        value=200,
        min=50,
        max=500,
        step=50,
    ),

    "alpha": Slider(
        label="Structural Weight (α)",
        value=0.5,
        min=0.0,
        max=1.0,
        step=0.1,
    ),
    
    "gamma": Slider(
        label="Fixed Affinity Weight (γ)",
        value=0.5,
        min=0.1,
        max=0.9,
        step=0.1,
    ),

    "QW_ratio": Slider(
        label="Queer Women Ratio",
        value=0.5,
        min=0.0,
        max=1.0,
        step=0.05,
    ),
    "QNW_ratio": Slider(
        label="Queer Non-Women Ratio",
        value=0.25,
        min=0.0,
        max=1.0,
        step=0.05,
    ),
    "adaptive_update_interval": Slider(
        label="Adaptive Update Interval",
        value=10,
        min=1,
        max=30,
        step=1,
    )
}

components = [
    AgentMapComponent,
    BarVisitorCountTrendsComponent,
    EffectiveAffinityTrendsComponent,
    BarStatusComponent,
    BarProportionTrendsComponent
]

model = LGBTQBarModel()

page = SolaraViz(
    model,
    components=components,
    model_params=model_params,
    name="Lesbian Bars Simulation"
)
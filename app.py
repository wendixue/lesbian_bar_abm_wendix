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
            ax.text(pos[0], pos[1], model.bars[i].name, ha='center', va='center', fontsize=9)
    
    # Draw temporary exit zone
    ax.scatter(temp_exit_position[0], temp_exit_position[1], s=300, color='orange', alpha=0.2, marker='s')
    ax.text(temp_exit_position[0], temp_exit_position[1], 'Temp Exit (10 rounds)', ha='center', va='center', fontsize=8)
    
    # Draw permanent exit zone
    ax.scatter(perm_exit_position[0], perm_exit_position[1], s=300, color='red', alpha=0.2, marker='s')
    ax.text(perm_exit_position[0], perm_exit_position[1], 'Permanent Exit', ha='center', va='center', fontsize=8)
    
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
    for agent in model.schedule.agents:
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
    step_text = model.schedule.steps if hasattr(model.schedule, "steps") else 0
    title = f'Agent Distribution (Step: {step_text}, Active: {sum(bar_counts)}, Temp: {temp_count}, Perm: {perm_count})'
    ax.set_title(title)
    
    # Hide axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    return solara.FigureMatplotlib(figure=fig)

# Create bar population ratio component
@solara.component
def BarRatiosComponent(model):

    update_counter.get()
    
    # Get current simulation step
    current_step = model.schedule.steps if hasattr(model.schedule, 'steps') else 0
    
    # Initialize time series history on first step
    if not hasattr(model, 'time_series_history'):
        model.time_series_history = {
            'steps': [],
            'bars': {},
            'lesbian_status': {},  
        }
        # Initialize data structures for each bar
        for i in range(len(model.bars)):
            model.time_series_history['bars'][i] = {
                'QW': [],
                'NQW': [],
                'QNW': []
            }
            model.time_series_history['lesbian_status'][i] = [] 
    
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
                    
                # Log to time series history
                for group in counts:
                    model.time_series_history['bars'][bar_id][group].append(counts[group])
            else:
                # Append zero if no history
                for group in ['QW', 'NQW', 'QNW']:
                    model.time_series_history['bars'][bar_id][group].append(0)
            
            # Record lesbian bar symbolic identity
            model.time_series_history['lesbian_status'][bar_id].append(int(bar.is_lesbian_bar))
    
    # Create time series plot for each bar
    with solara.Column():
        for bar_id in range(len(model.bars)):
            solara.Markdown(f"### {model.bars[bar_id].name} Population Trends")
            
            # Create visitor trend plot
            fig = Figure(figsize=(10, 6))
            ax = fig.add_subplot(111)
            
            # Plot lines for each identity group
            steps = model.time_series_history['steps']
            
            # Define colors for each identity group
            colors = {
                "QW": "red",
                "NQW": "pink", 
                "QNW": "purple"
            }
            
            # draw lines for each group
            for group, color in colors.items():
                values = model.time_series_history['bars'][bar_id][group]
                ax.plot(steps, values, label=group, color=color, marker='o', markersize=4)
            
            # Set plot properties
            ax.set_xlabel('Step')
            ax.set_ylabel('Visitor Count')
            ax.set_title(f'{model.bars[bar_id].name} Visitor Count by Group')
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()
            
            # Set integer x-axis ticks
            if len(steps) > 1:
                ax.set_xticks(range(0, max(steps) + 1, max(1, max(steps) // 10)))
            
            solara.FigureMatplotlib(figure=fig)

# Create bar status component
@solara.component
def BarStatusComponent(model):
    # Ensure component updates with model state
    update_counter.get()
    
    # Get current simulation step
    current_step = model.schedule.steps if hasattr(model.schedule, 'steps') else 0
    
    with solara.Column():
        for i, bar in enumerate(model.bars):
            # Get current population ratios for bar
            ratios = bar.get_current_population_ratios()
            
            solara.Markdown(f"### {bar.name} Status (Step: {current_step})")
            
            # Create QW ratio and lesbian identity status plot
            if hasattr(model, 'time_series_history') and model.time_series_history['steps']:
                steps = model.time_series_history['steps']
                
                fig = Figure(figsize=(10, 4))
                ax = fig.add_subplot(111)
                
                # Get QW ratio data
                qw_ratios = []
                for j in range(len(steps)):
                    try:
                        total = sum(model.time_series_history['bars'][i][group][j] for group in ["QW", "NQW", "QNW"])
                        qw_ratio = model.time_series_history['bars'][i]["QW"][j] / total if total > 0 else 0
                        qw_ratios.append(qw_ratio)
                    except IndexError:
                        qw_ratios.append(0)
                
                # Get lesbian bar status data
                lesbian_status = model.time_series_history['lesbian_status'][i]
                if len(lesbian_status) < len(steps):
                    # Pad with zeros if data lengths mismatch
                    lesbian_status = lesbian_status + [0] * (len(steps) - len(lesbian_status))
                
                # Plot QW ratio curve
                line1 = ax.plot(steps, qw_ratios, label="QW Ratio", color="red", marker='o', markersize=4)
                ax.set_ylabel('QW Ratio', color='red')
                ax.set_ylim(0, 1)
                
                # Use second Y-axis for lesbian identity status
                ax2 = ax.twinx()
                line2 = ax2.plot(steps, lesbian_status, label="Lesbian Bar Status", color="blue", linestyle='--', linewidth=2)
                ax2.set_ylabel('Lesbian Bar Status', color='blue')
                ax2.set_ylim(-0.1, 1.1)
                ax2.set_yticks([0, 1])
                ax2.set_yticklabels(['No', 'Yes'])
                
                # Add threshold reference lines
                ax.axhline(y=0.3, color='gray', linestyle='--', alpha=0.5)
                ax.axhline(y=0.6, color='gray', linestyle='--', alpha=0.5)
                
                # Combine legends
                lines = line1 + line2
                labels = [l.get_label() for l in lines]
                ax.legend(lines, labels, loc="upper right")
                
                # Set plot properties
                ax.set_xlabel('Step')
                ax.set_title(f'{bar.name} QW Ratio & Lesbian Bar Status')
                ax.grid(True, linestyle='--', alpha=0.7)
                
                # Set integer x-axis ticks
                if len(steps) > 1:
                    ax.set_xticks(range(0, max(steps) + 1, max(1, max(steps) // 10)))
                
                solara.FigureMatplotlib(figure=fig)
            
            # Display current lesbian identity status
            is_lesbian = bar.is_lesbian_bar
            status_text = "Yes" if is_lesbian else "No"
            
            # Show basic stats for bar
            solara.Markdown(f"**Current Lesbian Bar Status:** {status_text}")
            solara.Markdown(f"**Current QW Ratio:** {ratios['QW']:.2f}")
            solara.Markdown(f"**Current NQW Ratio:** {ratios['NQW']:.2f}")
            solara.Markdown(f"**Current QNW Ratio:** {ratios['QNW']:.2f}")
            solara.Markdown(f"**Current Visitor Count:** {len(bar.current_visitors)}")

            if i < len(model.bars) - 1:
                solara.Markdown("---")


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
        label="Affinity Weight (alpha)",
        value=0.4,
        min=0.0,
        max=1.0,
        step=0.1,
    ),

    ## One of the ratio slider need to be deleted!
    "QW_ratio": Slider(
        label="Queer Women Ratio",
        value=0.4,
        min=0.0,
        max=1.0,
        step=0.05,
    ),
    "NQW_ratio": Slider(
        label="Non-Queer Women Ratio",
        value=0.3,
        min=0.0,
        max=1.0,
        step=0.05,
    ),
    "QNW_ratio": Slider(
        label="Queer Non-Women Ratio",
        value=0.3,
        min=0.0,
        max=1.0,
        step=0.05,
    ),
    "adaptive_update_interval": Slider(
        label="Adaptive Affinity Update Interval",
        value=10,
        min=1,
        max=30,
        step=1,
    ),
    "affinity_factor": Slider(
        label="Affinity Factor",
        value=0.01,
        min=0.01,
        max=0.1,
        step=0.01,
    ),
    "agent_threshold": Slider(
        label="Agent Belonging Threshold",
        value=0.5,
        min=0.1,
        max=0.9,
        step=0.05,
    )
}

components = [
    AgentMapComponent,
    BarRatiosComponent,
    BarStatusComponent 
]

model = LGBTQBarModel()

page = SolaraViz(
    model,
    components=components,
    model_params=model_params,
    name="Lesbian Bars Simulation",
    render_interval=1, 
)
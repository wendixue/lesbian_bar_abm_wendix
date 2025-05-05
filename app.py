from model import LGBTQBarModel
from mesa.visualization import Slider, SolaraViz
import solara
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import random

# 创建自定义组件来显示Agent位置分布
@solara.component
def AgentMapComponent(model):
    fig = Figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    
    # 设置固定的坐标位置
    bar_positions = [(0.3, 0.5), (0.7, 0.5)]  # 酒吧0和酒吧1的位置
    exit_position = (0.5, 0.1)  # 退出区域的位置
    
    # 绘制酒吧位置
    for i, pos in enumerate(bar_positions):
        if i < len(model.bars):
            ax.scatter(pos[0], pos[1], s=300, color='gray', alpha=0.5, marker='s')
            ax.text(pos[0], pos[1], f'酒吧{i+1}', ha='center', va='center')
    
    # 绘制退出区域
    ax.scatter(exit_position[0], exit_position[1], s=300, color='black', alpha=0.2, marker='s')
    ax.text(exit_position[0], exit_position[1], '已退出', ha='center', va='center')
    
    # 为每种身份群体定义颜色
    colors = {
        "QW": "red",
        "NQW": "pink", 
        "QNW": "purple",
        "NQNW": "blue"
    }
    
    # 绘制每个agent
    for agent in model.schedule.agents:
        # 决定agent的位置
        if agent.status == "exited":
            # 如果已退出，在退出区域周围随机放置
            x = exit_position[0] + (random.random() - 0.5) * 0.2
            y = exit_position[1] + (random.random() - 0.5) * 0.2
        elif agent.current_bar is not None:
            # 如果在酒吧中，在对应酒吧周围随机放置
            bar_pos = bar_positions[agent.current_bar]
            x = bar_pos[0] + (random.random() - 0.5) * 0.2
            y = bar_pos[1] + (random.random() - 0.5) * 0.2
        else:
            # 未决定去向的agent，不显示
            continue
        
        # 根据身份群体绘制不同颜色的点
        ax.scatter(x, y, s=50, color=colors[agent.identity_group], alpha=0.7)
    
    # 添加图例说明
    for group, color in colors.items():
        ax.scatter([], [], color=color, label=group)
    
    ax.legend(loc='upper right')
    
    # 设置坐标轴范围和标题
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('Agent 位置分布')
    
    # 隐藏坐标轴
    ax.set_xticks([])
    ax.set_yticks([])
    
    return solara.FigureMatplotlib(figure=fig)

# 创建酒吧人群比例图
@solara.component
def BarChartComponent(model):
    fig = Figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    
    # 获取各个酒吧的人群比例
    bar_labels = []
    qw_ratios = []
    nqw_ratios = []
    qnw_ratios = []
    nqnw_ratios = []
    
    for i, bar in enumerate(model.bars):
        if i < len(model.bars):  # 确保酒吧索引有效
            ratios = bar.get_current_population_ratios()
            bar_labels.append(f"酒吧{i+1}")
            qw_ratios.append(ratios["QW"])
            nqw_ratios.append(ratios["NQW"])
            qnw_ratios.append(ratios["QNW"])
            nqnw_ratios.append(ratios["NQNW"])
    
    # 如果没有数据，添加一个空的占位符
    if not bar_labels:
        return solara.Markdown("没有酒吧数据可显示")
    
    # 创建堆叠柱状图
    x = range(len(bar_labels))
    width = 0.6
    
    ax.bar(x, qw_ratios, width, label='QW', color='red')
    ax.bar(x, nqw_ratios, width, bottom=qw_ratios, label='NQW', color='pink')
    
    # 计算第三层的起始位置
    third_bottom = [qw + nqw for qw, nqw in zip(qw_ratios, nqw_ratios)]
    ax.bar(x, qnw_ratios, width, bottom=third_bottom, label='QNW', color='purple')
    
    # 计算第四层的起始位置
    fourth_bottom = [qw + nqw + qnw for qw, nqw, qnw in zip(qw_ratios, nqw_ratios, qnw_ratios)]
    ax.bar(x, nqnw_ratios, width, bottom=fourth_bottom, label='NQNW', color='blue')
    
    ax.set_title('酒吧人群比例')
    ax.set_ylabel('比例')
    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels)
    ax.legend()
    
    return solara.FigureMatplotlib(figure=fig)

# 创建自定义组件来显示酒吧状态
@solara.component
def BarStatusComponent(model):
    with solara.Column():
        for i, bar in enumerate(model.bars):
            # 获取各个群体在该酒吧的比例
            ratios = bar.get_current_population_ratios()
            
            solara.Markdown(f"### 酒吧 {i+1}")
            solara.Markdown(f"女同酒吧状态: {'是' if bar.is_lesbian_bar else '否'}")
            solara.Markdown(f"访客数量: {len(bar.current_visitors)}")
            solara.Markdown(f"QW比例: {ratios['QW']:.2f}")
            solara.Markdown(f"NQW比例: {ratios['NQW']:.2f}")
            solara.Markdown(f"QNW比例: {ratios['QNW']:.2f}")
            solara.Markdown(f"NQNW比例: {ratios['NQNW']:.2f}")
            solara.Markdown(f"对QW的适应性容忍度: {bar.adaptive_tolerance['QW']:.2f}")
            
            # 使用水平线分隔不同的酒吧信息
            if i < len(model.bars) - 1:
                solara.Markdown("---")

# 创建自定义组件来显示人群活跃状态
@solara.component
def PopulationStatusComponent(model):
    with solara.Column():
        exited = model.count_exited_agents()
        active_qw = model.count_active_by_group("QW")
        active_nqw = model.count_active_by_group("NQW")
        active_qnw = model.count_active_by_group("QNW")
        active_nqnw = model.count_active_by_group("NQNW")
        
        solara.Markdown(f"### 人群活跃状态")
        solara.Markdown(f"退出系统: {exited}")
        solara.Markdown(f"活跃QW: {active_qw}")
        solara.Markdown(f"活跃NQW: {active_nqw}")
        solara.Markdown(f"活跃QNW: {active_qnw}")
        solara.Markdown(f"活跃NQNW: {active_nqnw}")

# 定义模型参数
model_params = {
    "seed": {
        "type": "InputText",
        "value": 42,
        "label": "随机种子",
    },
    "population_size": Slider(
        label="Agent总数",
        value=200,
        min=50,
        max=500,
        step=50,
    ),
    "num_bars": Slider(
        label="酒吧数量",
        value=2,
        min=1,
        max=3,
        step=1,
    ),
    "alpha": Slider(
        label="容忍度权重 (alpha)",
        value=0.4,
        min=0.0,
        max=1.0,
        step=0.1,
    ),
    "QW_ratio": Slider(
        label="Queer Women占比",
        value=0.4,
        min=0.0,
        max=1.0,
        step=0.05,
    ),
    "NQW_ratio": Slider(
        label="Non-Queer Women占比",
        value=0.3,
        min=0.0,
        max=1.0,
        step=0.05,
    ),
    "QNW_ratio": Slider(
        label="Queer Non-Women占比",
        value=0.2,
        min=0.0,
        max=1.0,
        step=0.05,
    ),
    "NQNW_ratio": Slider(
        label="Non-Queer Non-Women占比",
        value=0.1,
        min=0.0,
        max=1.0,
        step=0.05,
    ),
}

# 创建可视化组件列表
components = [
    AgentMapComponent,
    BarChartComponent,
    BarStatusComponent,
    PopulationStatusComponent
]

# 创建模型实例
model = LGBTQBarModel()

# 创建可视化页面
page = SolaraViz(
    model,
    components=components,
    model_params=model_params,
    name="归属感与文化张力驱动的女同酒吧演化模拟",
)
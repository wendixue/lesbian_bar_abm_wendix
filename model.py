import mesa
from mesa.datacollection import DataCollector
from agent import PersonAgent, Bar, IDENTITY_GROUPS

# 自定义激活器替代RandomActivation
class CustomActivation:
    """自定义激活器，替代mesa.time.RandomActivation"""
    def __init__(self, model):
        self.model = model
        self.agents = []
        self.steps = 0
    
    def add(self, agent):
        """添加一个Agent到调度器
        
        Args:
            agent: Agent实例
        """
        self.agents.append(agent)
    
    def step(self):
        """执行一个步骤：随机顺序激活所有Agent"""
        # 随机打乱Agent顺序
        shuffled = self.agents.copy()
        self.model.random.shuffle(shuffled)
        
        # 顺序执行每个Agent的step方法
        print(f"执行 {len(shuffled)} 个Agent的step方法")
        for agent in shuffled:
            agent.step()
        
        # 增加步数计数
        self.steps += 1

class LGBTQBarModel(mesa.Model):
    def __init__(self, 
                 population_size=200, 
                 num_bars=2, 
                 alpha=0.4,
                 init_identity_ratios=None,
                 QW_ratio=0.4,
                 NQW_ratio=0.3,
                 QNW_ratio=0.2,
                 NQNW_ratio=0.1,
                 bar_fixed_tolerances=None,
                 seed=None):
        """初始化模型
        
        Args:
            population_size: Agent总数
            num_bars: 酒吧数量
            alpha: 归属感计算中酒吧容忍度的权重
            init_identity_ratios: 初始身份群体比例字典（如果提供）
            QW_ratio: Queer Women的初始比例（如果init_identity_ratios未提供）
            NQW_ratio: Non-Queer Women的初始比例（如果init_identity_ratios未提供）
            QNW_ratio: Queer Non-Women的初始比例（如果init_identity_ratios未提供）
            NQNW_ratio: Non-Queer Non-Women的初始比例（如果init_identity_ratios未提供）
            bar_fixed_tolerances: 酒吧的固定容忍度
            seed: 随机种子
        """
        super().__init__(seed=seed)
        self.num_agents = population_size
        self.num_bars = num_bars
        self.alpha = alpha  # 归属感计算中酒吧容忍度的权重
        self.schedule = CustomActivation(self)  # 使用自定义激活器
        self._agent_storage = self.schedule  # 使用其他名称存储Agent
        
        # 设置初始身份群体比例
        if init_identity_ratios is None:
            # 使用单独的比例参数
            total = QW_ratio + NQW_ratio + QNW_ratio + NQNW_ratio
            if total == 0:
                # 避免除以0的情况
                init_identity_ratios = {
                    "QW": 0.25,
                    "NQW": 0.25,
                    "QNW": 0.25,
                    "NQNW": 0.25
                }
            else:
                # 归一化比例，确保总和为1
                init_identity_ratios = {
                    "QW": QW_ratio / total,
                    "NQW": NQW_ratio / total,
                    "QNW": QNW_ratio / total,
                    "NQNW": NQNW_ratio / total
                }
        
        # 设置默认的酒吧固定容忍度 (如未指定)
        if bar_fixed_tolerances is None:
            # 第一个酒吧: 女同友好型
            lesbian_bar_tolerance = {
                "QW": 1.0,    # 完全欢迎 Queer Women
                "NQW": 0.8,   # 较欢迎 Non-Queer Women
                "QNW": 0.6,   # 中等欢迎 Queer Non-Women
                "NQNW": 0.2   # 较少欢迎 Non-Queer Non-Women
            }
            
            # 第二个酒吧: 中性型
            neutral_bar_tolerance = {
                "QW": 0.7,    # 较欢迎 Queer Women
                "NQW": 0.7,   # 较欢迎 Non-Queer Women
                "QNW": 0.7,   # 较欢迎 Queer Non-Women
                "NQNW": 0.7   # 较欢迎 Non-Queer Non-Women
            }
            
            bar_fixed_tolerances = [lesbian_bar_tolerance]
            
            # 如果有多个酒吧，添加中性酒吧
            if num_bars > 1:
                bar_fixed_tolerances.append(neutral_bar_tolerance)
                
            # 如果有更多酒吧，添加随机容忍度的酒吧
            for _ in range(num_bars - 2):
                random_tolerance = {
                    group: self.random.uniform(0.3, 0.9) 
                    for group in IDENTITY_GROUPS
                }
                bar_fixed_tolerances.append(random_tolerance)
        
        # 创建酒吧
        self.bars = []
        for i in range(num_bars):
            bar = Bar(i, bar_fixed_tolerances[i])
            self.bars.append(bar)
        
        # 创建Agents
        for i in range(self.num_agents):
            # 根据比例分配身份群体
            r = self.random.random()
            cumulative = 0
            assigned_group = None
            
            for group, ratio in init_identity_ratios.items():
                cumulative += ratio
                if r <= cumulative:
                    assigned_group = group
                    break
            
            # 设置个体归属感阈值 (在0.5-0.8之间随机)
            threshold = self.random.uniform(0.5, 0.8)
            
            # 创建Agent
            agent = PersonAgent(i, self, assigned_group, threshold)
            self.schedule.add(agent)
            
            # 随机初始化对酒吧的记忆
            for bar_id in range(num_bars):
                # 添加一些随机初始记忆，促进初始流动
                initial_score = self.random.uniform(0.4, 0.9)
                agent.update_memory(bar_id, initial_score)
        
        # 设置数据收集器
        model_reporters = {
            "Bar1_QW_Ratio": lambda m: self.get_bar_group_ratio(0, "QW"),
            "Bar1_NQW_Ratio": lambda m: self.get_bar_group_ratio(0, "NQW"),
            "Bar1_QNW_Ratio": lambda m: self.get_bar_group_ratio(0, "QNW"),
            "Bar1_NQNW_Ratio": lambda m: self.get_bar_group_ratio(0, "NQNW"),
            "Bar1_Population": lambda m: self.get_bar_population(0),
            "Bar1_IsLesbianBar": lambda m: int(self.bars[0].is_lesbian_bar),
            "Bar1_QW_AdaptiveTolerance": lambda m: self.bars[0].adaptive_tolerance["QW"],
            "Exited_Agents": lambda m: self.count_exited_agents(),
            "Active_QW": lambda m: self.count_active_by_group("QW"),
            "Active_NQW": lambda m: self.count_active_by_group("NQW"),
            "Active_QNW": lambda m: self.count_active_by_group("QNW"),
            "Active_NQNW": lambda m: self.count_active_by_group("NQNW")
        }
        
        # 如果有第二个酒吧，添加相关数据收集
        if num_bars > 1:
            additional_reporters = {
                "Bar2_QW_Ratio": lambda m: self.get_bar_group_ratio(1, "QW"),
                "Bar2_NQW_Ratio": lambda m: self.get_bar_group_ratio(1, "NQW"),
                "Bar2_QNW_Ratio": lambda m: self.get_bar_group_ratio(1, "QNW"),
                "Bar2_NQNW_Ratio": lambda m: self.get_bar_group_ratio(1, "NQNW"),
                "Bar2_Population": lambda m: self.get_bar_population(1),
                "Bar2_IsLesbianBar": lambda m: int(self.bars[1].is_lesbian_bar),
                "Bar2_QW_AdaptiveTolerance": lambda m: self.bars[1].adaptive_tolerance["QW"]
            }
            model_reporters.update(additional_reporters)
            
        self.datacollector = DataCollector(model_reporters=model_reporters)
        
    def get_bar_group_ratio(self, bar_id, group):
        """获取特定酒吧中特定群体的比例
        
        Args:
            bar_id: 酒吧ID
            group: 群体类型
            
        Returns:
            float: 群体比例
        """
        bar = self.bars[bar_id]
        ratios = bar.get_current_population_ratios()
        return ratios[group]
    
    def get_bar_population(self, bar_id):
        """获取特定酒吧的总人数
        
        Args:
            bar_id: 酒吧ID
            
        Returns:
            int: 酒吧人数
        """
        return len(self.bars[bar_id].current_visitors)
    
    def count_exited_agents(self):
        """计算已退出系统的Agent数量"""
        return sum(1 for agent in self._agent_storage.agents if agent.status == "exited")
    
    def count_active_by_group(self, group):
        """计算特定群体中活跃的Agent数量
        
        Args:
            group: 群体类型
            
        Returns:
            int: 活跃Agent数量
        """
        return sum(1 for agent in self._agent_storage.agents 
                  if agent.identity_group == group and agent.status == "active")
    
    def step(self):
        """模型每步运行逻辑"""
        # 清空酒吧当前访客
        for bar in self.bars:
            bar.current_visitors = []
        
        # 打印信息 (调试用)
        print(f"\n=== 执行第 {self.schedule.steps if hasattr(self.schedule, 'steps') else '?'} 步 ===")
        print(f"当前活跃Agent数: {len([a for a in self._agent_storage.agents if a.status == 'active'])}")
        
        # 执行所有Agent的步骤
        self.schedule.step()
        
        # 结束酒吧的当前轮次
        for bar in self.bars:
            bar.end_round()
        
        # 收集数据
        self.datacollector.collect(self)
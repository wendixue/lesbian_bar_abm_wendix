import mesa
import random
import numpy as np

# 定义四类身份群体
IDENTITY_GROUPS = ["QW", "NQW", "QNW", "NQNW"]
# QW: Queer Women (包含lesbian与其他queer女性)
# NQW: Non-Queer Women (异性恋女性)
# QNW: Queer Non-Women (queer男性、非二元人群等)
# NQNW: Non-Queer Non-Women (异性恋男性等)

# 归属感矩阵 - 表示不同群体间的归属关系 (非对称结构)
BELONGING_MATRIX = {
    "QW":   {"QW": 1.0, "NQW": 0.6, "QNW": 0.3, "NQNW": 0.0},
    "NQW":  {"QW": 0.5, "NQW": 1.0, "QNW": 0.3, "NQNW": 0.1},
    "QNW":  {"QW": 0.4, "NQW": 0.4, "QNW": 1.0, "NQNW": 0.5},
    "NQNW": {"QW": 0.7, "NQW": 0.8, "QNW": 0.0, "NQNW": 1.0}
}

# 创建Bar类来表示酒吧
class Bar:
    def __init__(self, unique_id, fixed_tolerance, gamma=0.7):
        """初始化酒吧
        
        Args:
            unique_id: 酒吧的唯一ID
            fixed_tolerance: 对各群体的固定文化容忍度字典
            gamma: 固定容忍度与适应容忍度的权重参数
        """
        self.unique_id = unique_id
        self.fixed_tolerance = fixed_tolerance
        self.adaptive_tolerance = {group: 0.5 for group in IDENTITY_GROUPS}  # 初始为中性
        self.gamma = gamma  # 固定容忍度的权重
        
        # 酒吧的访客历史记录
        self.visitor_history = []
        self.current_visitors = []
        self.update_count = 0  # 用于追踪更新adaptive_tolerance的轮次
        
        # 酒吧的文化状态
        self.is_lesbian_bar = True  # 初始状态默认为女同酒吧
        self.qw_ratio_history = []  # 记录QW比例历史
        
    def calculate_effective_tolerance(self):
        """计算当前有效容忍度 (固定与适应容忍度的加权平均)"""
        effective = {}
        for group in IDENTITY_GROUPS:
            effective[group] = (self.gamma * self.fixed_tolerance[group] + 
                              (1 - self.gamma) * self.adaptive_tolerance[group])
        return effective
    
    def update_adaptive_tolerance(self, force=False):
        """每15轮更新一次适应性容忍度
        
        Args:
            force: 是否强制更新容忍度
        """
        self.update_count += 1
        
        # 每15轮或强制更新时更新适应性容忍度
        if self.update_count >= 15 or force:
            if len(self.visitor_history) > 0:
                # 计算最近历史中各群体的比例
                visitor_counts = {group: 0 for group in IDENTITY_GROUPS}
                total_visitors = 0
                
                # 统计最近5轮的访客
                recent_history = self.visitor_history[-5:] if len(self.visitor_history) > 5 else self.visitor_history
                for visitors in recent_history:
                    for visitor in visitors:
                        visitor_counts[visitor] += 1
                        total_visitors += 1
                
                # 更新适应性容忍度 (根据访客比例)
                if total_visitors > 0:
                    for group in IDENTITY_GROUPS:
                        group_ratio = visitor_counts[group] / total_visitors
                        # 适应性容忍度逐渐向访客比例靠拢
                        self.adaptive_tolerance[group] = (0.8 * self.adaptive_tolerance[group] + 
                                                        0.2 * group_ratio)
            
            # 检查是否仍为女同酒吧 (QW比例低于30%被视为去女同化)
            if len(self.visitor_history) > 5:
                recent_qw_ratios = []
                for visitors in self.visitor_history[-5:]:
                    if visitors:  # 确保有访客
                        qw_count = sum(1 for v in visitors if v == "QW")
                        qw_ratio = qw_count / len(visitors) if len(visitors) > 0 else 0
                        recent_qw_ratios.append(qw_ratio)
                
                avg_qw_ratio = sum(recent_qw_ratios) / len(recent_qw_ratios) if recent_qw_ratios else 0
                self.qw_ratio_history.append(avg_qw_ratio)
                
                # 更新女同酒吧状态 (连续3轮QW比例低于30%则视为去女同化)
                if len(self.qw_ratio_history) >= 3:
                    recent_status = self.qw_ratio_history[-3:]
                    if all(ratio < 0.3 for ratio in recent_status):
                        self.is_lesbian_bar = False
            
            self.update_count = 0  # 重置计数器
    
    def add_visitors(self, visitors):
        """添加当前轮次的访客
        
        Args:
            visitors: 访客身份列表
        """
        self.current_visitors.extend(visitors)
        # 打印信息 (调试用)
        print(f"酒吧 {self.unique_id} 添加访客: {visitors}, 当前访客数: {len(self.current_visitors)}")
    
    def end_round(self):
        """轮次结束时的处理"""
        self.visitor_history.append(self.current_visitors.copy())
        self.current_visitors = []
        self.update_adaptive_tolerance()
    
    def get_current_population_ratios(self):
        """获取当前访客的群体比例"""
        if not self.current_visitors:
            return {group: 0.0 for group in IDENTITY_GROUPS}
        
        counts = {group: 0 for group in IDENTITY_GROUPS}
        for visitor in self.current_visitors:
            counts[visitor] += 1
        
        total = len(self.current_visitors)
        return {group: counts[group] / total for group in IDENTITY_GROUPS}


# 创建个体Agent类
class PersonAgent:
    def __init__(self, unique_id, model, identity_group, threshold=0.6, memory_length=5):
        """初始化个体Agent
        
        Args:
            unique_id: Agent的唯一ID
            model: 所属的模型实例
            identity_group: 身份群体 (QW, NQW, QNW, NQNW)
            threshold: 归属感接受阈值
            memory_length: 记忆长度
        """
        self.unique_id = unique_id
        self.model = model
        self.identity_group = identity_group
        self.threshold = threshold  # 个体的归属感接受阈值
        self.memory = {}  # 记忆: {bar_id: [最近的归属感评分]}
        self.memory_length = memory_length
        self.current_bar = None  # 当前所在的酒吧
        self.status = "active"  # 状态: active(活跃) 或 exited(退出)
        
        # 初始化对所有酒吧的记忆
        for bar_id in range(self.model.num_bars):
            self.memory[bar_id] = []
    
        self.identity_group = identity_group
        self.threshold = threshold  # 个体的归属感接受阈值
        self.memory = {}  # 记忆: {bar_id: [最近的归属感评分]}
        self.memory_length = memory_length
        self.current_bar = None  # 当前所在的酒吧
        self.status = "active"  # 状态: active(活跃) 或 exited(退出)
        
        # 初始化对所有酒吧的记忆
        for bar_id in range(self.model.num_bars):
            self.memory[bar_id] = []
    
    def calculate_belonging(self, bar):
        """计算在特定酒吧中的归属感
        
        Args:
            bar: 酒吧实例
            
        Returns:
            float: 归属感分数 (0-1)
        """
        if self.status == "exited":
            return 0.0
        
        alpha = self.model.alpha  # 容忍度权重
        
        # 获取酒吧对自身群体的容忍度
        effective_tolerance = bar.calculate_effective_tolerance()
        bar_tolerance = effective_tolerance[self.identity_group]
        
        # 计算由其他访客构成的社交氛围影响
        population_ratios = bar.get_current_population_ratios()
        social_belonging = 0.0
        
        for other_group in IDENTITY_GROUPS:
            # 获取当前群体对其他群体的归属感系数
            group_belonging = BELONGING_MATRIX[self.identity_group][other_group]
            # 加权计算归属感 (其他群体比例 * 归属感系数)
            social_belonging += group_belonging * population_ratios[other_group]
        
        # 综合计算最终归属感 = 酒吧容忍度部分 + 社交氛围部分
        total_belonging = (alpha * bar_tolerance) + ((1 - alpha) * social_belonging)
        
        return total_belonging
    
    def update_memory(self, bar_id, belonging_score):
        """更新对特定酒吧的归属感记忆
        
        Args:
            bar_id: 酒吧ID
            belonging_score: 归属感分数
        """
        self.memory[bar_id].append(belonging_score)
        
        # 仅保留最近的记忆
        if len(self.memory[bar_id]) > self.memory_length:
            self.memory[bar_id].pop(0)
    
    def choose_bar(self):
        """基于记忆中的归属感选择酒吧"""
        if self.status == "exited":
            return None
        
        # 获取当前步数
        current_step = getattr(self.model.schedule, 'steps', 0)
        
        # 前5轮基于酒吧的基础欢迎度来选择，不能退出
        if current_step < 5:
            # 计算每个酒吧对此Agent群体的基础欢迎度
            bar_tolerances = []
            for bar_id, bar in enumerate(self.model.bars):
                tolerance = bar.fixed_tolerance[self.identity_group]
                bar_tolerances.append((bar_id, tolerance))
            
            # 根据欢迎度作为权重来选择酒吧
            weights = [tol for _, tol in bar_tolerances]
            bar_ids = [bid for bid, _ in bar_tolerances]
            
            if sum(weights) > 0:
                chosen_bar = random.choices(bar_ids, weights=weights, k=1)[0]
                print(f"前5轮: Agent {self.unique_id} 基于欢迎度选择了酒吧 {chosen_bar}")
                return chosen_bar
            else:
                # 如果所有酒吧的欢迎度都为0，随机选择
                chosen_bar = random.choice(range(self.model.num_bars))
                print(f"前5轮: Agent {self.unique_id} 随机选择了酒吧 {chosen_bar}")
                return chosen_bar
        
        # 5轮之后基于记忆中的归属感选择酒吧
        # 计算每个酒吧的平均归属感
        avg_belongings = {}
        valid_bars = []
        
        for bar_id in range(self.model.num_bars):
            # 如果有足够的记忆，计算平均值
            if self.memory[bar_id]:
                avg_belonging = sum(self.memory[bar_id]) / len(self.memory[bar_id])
                avg_belongings[bar_id] = avg_belonging
                print(f"Agent {self.unique_id} 对酒吧 {bar_id} 的平均归属感: {avg_belonging:.2f}")
                
                # 平均归属感高于阈值的酒吧被视为有效
                if avg_belonging >= self.threshold:
                    valid_bars.append(bar_id)
        
        # 如果没有有效的酒吧，则退出系统
        if not valid_bars:
            print(f"Agent {self.unique_id} 没有找到有效的酒吧，阈值: {self.threshold}")
            self.status = "exited"
            return None
        
        # 基于归属感形成概率分布选择酒吧
        # 归属感越高，选择概率越大
        if valid_bars:
            # 只考虑有效酒吧的归属感
            valid_belongings = {bar_id: avg_belongings[bar_id] for bar_id in valid_bars}
            total_belonging = sum(valid_belongings.values())
            
            if total_belonging > 0:
                # 计算每个有效酒吧的选择概率
                probs = [valid_belongings[bar_id] / total_belonging for bar_id in valid_bars]
                # 基于概率选择酒吧
                chosen_bar = random.choices(valid_bars, weights=probs, k=1)[0]
                print(f"Agent {self.unique_id} 选择了酒吧 {chosen_bar}")
                return chosen_bar
            else:
                # 如果无法形成概率分布，随机选择
                chosen_bar = random.choice(valid_bars)
                print(f"Agent {self.unique_id} 随机选择了酒吧 {chosen_bar}")
                return chosen_bar
        
        # 默认返回None (不应该到达这里)
        return None
        
        # 基于归属感形成概率分布选择酒吧
        # 归属感越高，选择概率越大
        if valid_bars:
            # 只考虑有效酒吧的归属感
            valid_belongings = {bar_id: avg_belongings[bar_id] for bar_id in valid_bars}
            total_belonging = sum(valid_belongings.values())
            
            if total_belonging > 0:
                # 计算每个有效酒吧的选择概率
                probs = [valid_belongings[bar_id] / total_belonging for bar_id in valid_bars]
                # 基于概率选择酒吧
                chosen_bar = random.choices(valid_bars, weights=probs, k=1)[0]
                print(f"Agent {self.unique_id} 选择了酒吧 {chosen_bar}")
                return chosen_bar
            else:
                # 如果无法形成概率分布，随机选择
                chosen_bar = random.choice(valid_bars)
                print(f"Agent {self.unique_id} 随机选择了酒吧 {chosen_bar}")
                return chosen_bar
        
        # 默认返回None (不应该到达这里)
        return None
    
    def step(self):
        """每步行为"""
        if self.status == "exited":
            return
        
        # 选择酒吧
        chosen_bar_id = self.choose_bar()
        
        if chosen_bar_id is not None:
            # 更新当前酒吧
            self.current_bar = chosen_bar_id
            
            # 将自己添加到酒吧访客列表
            bar = self.model.bars[chosen_bar_id]
            bar.add_visitors([self.identity_group])
            
            # 计算并更新对该酒吧的归属感记忆
            belonging_score = self.calculate_belonging(bar)
            self.update_memory(chosen_bar_id, belonging_score)
            
            # 打印信息 (调试用)
            print(f"Agent {self.unique_id} ({self.identity_group}) 进入酒吧 {chosen_bar_id}，归属感: {belonging_score:.2f}")
        else:
            # 如果没有选择任何酒吧，则退出系统
            self.current_bar = None
            self.status = "exited"
            print(f"Agent {self.unique_id} ({self.identity_group}) 退出系统")
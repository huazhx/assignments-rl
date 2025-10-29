import numpy as np

class GridWorld5x5:
    """
    5x5网格世界求解器
    标准奖励设置：r_边界=-1, r_障碍=-1, r_目标=+1, r_其它=0
    折扣率：γ=0.9
    障碍可以进入，但代价为-1
    """
    
    def __init__(self):
        self.size = 5
        self.n_states = 25
        self.gamma = 0.9
        
    def state_to_coord(self, state):
        """状态编号转坐标 (row, col)"""
        return state // self.size, state % self.size
    
    def coord_to_state(self, row, col):
        """坐标转状态编号"""
        return row * self.size + col
    
    def is_boundary_hit(self, state, action):
        """判断动作是否会撞到边界"""
        row, col = self.state_to_coord(state)
        
        if action == 0 and col == self.size - 1:  # 右边界
            return True
        elif action == 1 and row == self.size - 1:  # 下边界
            return True
        elif action == 2 and col == 0:  # 左边界
            return True
        elif action == 3 and row == 0:  # 上边界
            return True
        return False
    
    def get_next_state(self, state, action):
        """获取执行动作后的下一个状态（障碍可进入）"""
        row, col = self.state_to_coord(state)
        
        # 动作：0=右, 1=下, 2=左, 3=上
        if action == 0:
            new_row, new_col = row, col + 1
        elif action == 1:
            new_row, new_col = row + 1, col
        elif action == 2:
            new_row, new_col = row, col - 1
        else:  # action == 3
            new_row, new_col = row - 1, col
        
        # 边界检查：撞边界则停在原地
        if new_row < 0 or new_row >= self.size or new_col < 0 or new_col >= self.size:
            return state
        
        next_state = self.coord_to_state(new_row, new_col)
        return next_state
    
    def compute_reward(self, state, action, next_state, obstacles, goal_states):
        """
        计算奖励
        r_边界 = -1, r_障碍 = -1, r_目标 = +1, r_其它 = 0
        """
        # 如果撞到边界（停在原地）
        if self.is_boundary_hit(state, action):
            return -1
        
        # 如果进入障碍区域
        if next_state in obstacles and next_state not in goal_states:
            return -1
        
        # 如果到达目标
        if next_state in goal_states:
            return 1
        
        # 其他情况（正常移动）
        return 0
    
    def build(self, obstacles, goal_states, action_map):
        """
        构建转移概率矩阵P和奖励向量R
        
        参数:
        obstacles: 障碍物状态集合，例如 {6, 7, 12}（可进入但奖励-1）
        goal_states: 终止状态集合，例如 {17}
        action_map: 策略
                   确定性: {state: action}
                   随机性: {state: [(action, prob), ...]}
        
        返回: (P, R)
        """
        P = np.zeros((self.n_states, self.n_states))
        R = np.zeros(self.n_states)
        
        for state in range(self.n_states):
            # 终止状态
            if state in goal_states:
                P[state, state] = 1.0
                R[state] = 0  # 终止状态奖励为0
                continue
            
            # 所有状态（包括障碍）都可以有策略
            if state not in action_map:
                continue
            
            actions = action_map[state]
            
            # 转换为列表格式
            if isinstance(actions, int):
                actions = [(actions, 1.0)]
            
            # 遍历所有可能的动作
            for action, prob in actions:
                next_state = self.get_next_state(state, action)
                reward = self.compute_reward(state, action, next_state, obstacles, goal_states)
                
                P[state, next_state] += prob
                R[state] += prob * reward
        
        return P, R
    
    def solve(self, P, R):
        """求解贝尔曼方程"""
        I = np.eye(self.n_states)
        A = I - self.gamma * P
        V = np.linalg.solve(A, R)
        return V
    
    def print_grid(self, V, obstacles, goal_states, title="状态值函数"):
        """打印网格形式的结果"""
        print(f"\n{title} (γ = {self.gamma}):\n")
        for row in range(self.size):
            row_str = []
            for col in range(self.size):
                state = row * self.size + col
                if state in goal_states:
                    row_str.append(f" [{V[state]:6.2f}]G")
                elif state in obstacles:
                    row_str.append(f" ({V[state]:6.2f}) ")  # 用括号表示障碍
                else:
                    row_str.append(f"  {V[state]:6.2f}  ")
            print("".join(row_str))
        
        print("\n图例：[value]G = 目标状态, (value) = 障碍区域, value = 正常区域")


# ==================== 示例1: 原图5x5网格 ====================

def example_original_5x5():
    """原图5x5网格世界"""
    print("=" * 70)
    print("示例1: 原图5x5网格世界（障碍可进入）")
    print("=" * 70)
    
    gw = GridWorld5x5()
    
    obstacles = {6, 7, 12, 16, 18, 21}
    goal_states = {17}
    
    action_map = {
        0: 0, 1: 0, 2: 0, 3: 1, 4: 1,
        5: 3, 6: 3, 7: 0, 8: 1, 9: 1,  # 障碍6,7也有策略
        10: 3, 11: 2, 12: 1, 13: 0, 14: 1,  # 障碍12也有策略
        15: 3, 16: 0, 18: 2, 19: 1,  # 障碍16,18也有策略
        20: 3, 21: 0, 22: 3, 23: 2, 24: 2  # 障碍21也有策略
    }
    
    P, R = gw.build(obstacles, goal_states, action_map)
    V = gw.solve(P, R)
    gw.print_grid(V, obstacles, goal_states)
    
    print("\n转移概率矩阵 P 的部分展示（前10个状态）:")
    print(P[:10, :10])
    
    print("\n奖励向量 R:")
    for i in range(25):
        if i % 5 == 0:
            print()
        marker = "G" if i in goal_states else ("O" if i in obstacles else " ")
        print(f"s{i:2d}{marker}: {R[i]:5.2f}", end="  ")
    print()
    
    return V


# ==================== 示例2: 最优策略（向目标移动）====================

def example_optimal_policy():
    """最优策略：所有状态都朝目标(3,2)移动"""
    print("\n" + "=" * 70)
    print("示例2: 最优策略示例（目标在(3,2)，即state 17）")
    print("=" * 70)
    
    gw = GridWorld5x5()
    
    obstacles = {6, 7, 12, 16, 18, 21}
    goal_states = {17}
    
    # 构建最优策略：每个状态选择最接近目标的方向
    action_map = {}
    goal_row, goal_col = 3, 2  # 目标在state 17
    
    for state in range(25):
        if state in goal_states:
            continue
            
        row, col = gw.state_to_coord(state)
        
        # 计算到目标的曼哈顿距离，选择能减少距离的方向
        # 优先垂直方向
        if row < goal_row:
            action = 1  # 下
        elif row > goal_row:
            action = 3  # 上
        elif col < goal_col:
            action = 0  # 右
        else:
            action = 2  # 左
        
        action_map[state] = action
    
    P, R = gw.build(obstacles, goal_states, action_map)
    V = gw.solve(P, R)
    gw.print_grid(V, obstacles, goal_states, "最优策略状态值函数")
    
    return V


# ==================== 示例3: 随机策略 ====================

def example_random_policy():
    """均匀随机策略"""
    print("\n" + "=" * 70)
    print("示例3: 均匀随机策略（4个方向等概率）")
    print("=" * 70)
    
    gw = GridWorld5x5()
    
    obstacles = {6, 7, 12, 16, 18, 21}
    goal_states = {17}
    
    # 均匀随机策略
    action_map = {}
    for state in range(25):
        if state in goal_states:
            continue
        action_map[state] = [(0, 0.25), (1, 0.25), (2, 0.25), (3, 0.25)]
    
    P, R = gw.build(obstacles, goal_states, action_map)
    V = gw.solve(P, R)
    gw.print_grid(V, obstacles, goal_states, "随机策略状态值函数")
    
    return V


# ==================== 示例4: 障碍区域的影响 ====================

def example_obstacle_comparison():
    """对比：有障碍 vs 无障碍的状态值"""
    print("\n" + "=" * 70)
    print("示例4: 对比障碍区域的影响")
    print("=" * 70)
    
    gw = GridWorld5x5()
    
    obstacles = {6, 7, 12, 16, 18, 21}
    goal_states = {17}
    
    # 简单策略：都向右下移动
    action_map = {}
    for state in range(25):
        if state in goal_states:
            continue
        
        row, col = gw.state_to_coord(state)
        if row < 4 and col < 4:
            action = [(0, 0.5), (1, 0.5)]  # 右或下
        elif row < 4:
            action = 1  # 下
        elif col < 4:
            action = 0  # 右
        else:
            action = 2  # 左（角落）
        
        action_map[state] = action
    
    P, R = gw.build(obstacles, goal_states, action_map)
    V = gw.solve(P, R)
    gw.print_grid(V, obstacles, goal_states, "有障碍情况")
    
    # 对比：无障碍情况
    print("\n" + "-" * 70)
    print("对比：无障碍情况")
    print("-" * 70)
    
    P_no_obs, R_no_obs = gw.build(set(), goal_states, action_map)
    V_no_obs = gw.solve(P_no_obs, R_no_obs)
    gw.print_grid(V_no_obs, set(), goal_states, "无障碍情况")
    
    # 显示差异
    print("\n状态值差异（无障碍 - 有障碍）:")
    diff = V_no_obs - V
    for row in range(5):
        row_str = []
        for col in range(5):
            state = row * 5 + col
            row_str.append(f"  {diff[state]:+6.2f}  ")
        print("".join(row_str))
    
    return V, V_no_obs


if __name__ == "__main__":
    # 运行所有示例
    example_original_5x5()
    example_optimal_policy()
    example_random_policy()
    example_obstacle_comparison()
    
    print("\n" + "=" * 70)
    print("所有示例求解完成！")
    print("奖励设置：r_边界=-1, r_障碍=-1（可进入）, r_目标=+1, r_其它=0")
    print("折扣率：γ=0.9")
    print("=" * 70)
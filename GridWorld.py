import numpy as np

class GridWorld:

    def __init__(self, size=5, n_states=25, action_map=None, gamma=0.9, obstacles=None, goal_states=None):
        self.size = size
        self.n_states = n_states
        self.gamma = gamma
        self.action_map = action_map 
        self.obstacles = obstacles if obstacles is not None else set()
        self.goal_states = goal_states if goal_states is not None else set()

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

        # 动作：0=右, 1=下, 2=左, 3=上, 4=停留
        if action == 0:
            new_row, new_col = row, col + 1
        elif action == 1:
            new_row, new_col = row + 1, col
        elif action == 2:
            new_row, new_col = row, col - 1
        elif action == 3:
            new_row, new_col = row - 1, col
        elif action == 4:
            new_row, new_col = row, col
        else:
            new_row, new_col = row, col  # 无效动作，停在原地

        # 边界检查：撞边界则停在原地
        if new_row < 0 or new_row >= self.size or new_col < 0 or new_col >= self.size:
            return state

        next_state = self.coord_to_state(new_row, new_col)
        return next_state

    def compute_reward(self, state, action, next_state):
        """
        计算奖励
        r_边界 = -1, r_障碍 = -1, r_目标 = +1, r_其它 = 0
        """
        # 如果撞到边界（停在原地）
        if self.is_boundary_hit(state, action):
            return -1

        # 如果进入障碍区域
        if next_state in self.obstacles and next_state not in self.goal_states:
            return -10

        # 如果到达目标
        if next_state in self.goal_states:
            return 1

        # 其他情况（正常移动）
        return 0

    def build(self):
        
        P = np.zeros((self.n_states, self.n_states))
        R = np.zeros(self.n_states)

        for state in range(self.n_states):

            # 所有状态（包括障碍）都可以有策略
            if state not in self.action_map:
                print(f"警告: 状态 {state} 没有定义动作，跳过该状态。")
                continue

            actions = self.action_map[state]

            # 转换为列表格式
            if isinstance(actions, int):
                actions = [(actions, 1.0)]

            # 遍历所有可能的动作
            for action, prob in actions:
                next_state = self.get_next_state(state, action)
                reward = self.compute_reward(
                    state, action, next_state)

                P[state, next_state] += prob
                R[state] += prob * reward

        return P, R

    def BOE_PR(self):
        """返回转移概率矩阵P和奖励矩阵R"""
        P = np.zeros((self.n_states, 5,self.n_states))
        R = np.zeros((self.n_states, 5), dtype=np.int32)  # 5个动作
        for s in range(self.n_states):
            # 所有状态（包括障碍）都可以有策略
            if s not in self.action_map:
                print(f"警告: 状态 {s} 没有定义动作，跳过该状态。")
                continue
            for a in range(5):  # 5个动作
                next_state = self.get_next_state(s, a)
                reward = self.compute_reward(s, a, next_state)
                P[s, a, next_state] = 1.0  # 确定性转移
                R[s, a] = reward
        return P, R


def print_grid(gamma, size, V, obstacles, goal_states, title="状态值函数"):
    """打印网格形式的结果"""
    print(f"\n{title} (γ = {gamma}):\n")
    
    # 打印列号
    print("    ", end="")
    for col in range(size):
        print(f"   col{col}   ", end="")
    print()
    
    for row in range(size):
        print(f"row{row} ", end="")
        row_str = []
        for col in range(size):
            state = row * size + col
            if state in goal_states:
                row_str.append(f"[{V[state]:6.2f}]G")
            elif state in obstacles:
                row_str.append(f"({V[state]:6.2f})X")  # X表示障碍
            else:
                row_str.append(f" {V[state]:6.2f}  ")
        print(" ".join(row_str))
    
    

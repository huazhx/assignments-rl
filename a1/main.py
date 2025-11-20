from GridWorld import GridWorld, print_grid
from Bellman import BellmanEquationSolver
from random import randint

def solve(obstacles, goal_states, action_map):
    gw = GridWorld(size=5, n_states=25, action_map=action_map, gamma=0.9,
                   obstacles=obstacles, goal_states=goal_states)
    P, R = gw.build()

    be = BellmanEquationSolver(P, R, gamma=0.9)
    V_iter, iterations = be.solve_iterative()
    V_close = be.solve_closed_form()
    print(f"迭代法求得： V (after {iterations} iterations)")
    print_grid(gamma=0.9, V=V_iter, size=5, obstacles=obstacles, goal_states=goal_states)

    print(f"解析式求解： V")
    print_grid(gamma=0.9, V=V_close, size=5, obstacles=obstacles, goal_states=goal_states)


obstacles = {6, 7, 12, 16, 18, 21}
goal_states = {17}

action_map = {
    0: 0, 1: 0, 2: 0, 3: 1, 4: 1,
    5: 3, 6: 3, 7: 0, 8: 1, 9: 1,  # 障碍6,7也有策略
    10: 3, 11: 2, 12: 1, 13: 0, 14: 1,  # 障碍12也有策略
    15: 3, 16: 0, 17:4, 18: 2, 19: 1,  # 障碍16,18也有策略
    20: 3, 21: 0, 22: 3, 23: 2, 24: 2  # 障碍21也有策略
}


action_map_2 = {
    # 第1行
    0: 0,   # (1,1) →
    1: 0,   # (1,2) →
    2: 0,   # (1,3) →
    3: 0,   # (1,4) →
    4: 1,   # (1,5) ↓
    
    # 第2行
    5: 3,   # (2,1) ↑
    6: 3,   # (2,2) ↑ 障碍
    7: 0,   # (2,3) → 障碍
    8: 0,   # (2,4) →
    9: 1,   # (2,5) ↓
    
    # 第3行
    10: 3,  # (3,1) ↑
    11: 2,  # (3,2) ←
    12: 1,  # (3,3) ↓ 障碍
    13: 0,  # (3,4) →
    14: 1,  # (3,5) ↓
    
    # 第4行
    15: 3,  # (4,1) ↑
    16: 0,  # (4,2) → 障碍
    17: 4,  # (4,3) 目标状态
    18: 2,  # (4,4) ← 障碍
    19: 1,  # (4,5) ↓
    
    # 第5行
    20: 3,  # (5,1) ↑
    21: 0,  # (5,2) → 障碍
    22: 3,  # (5,3) ↑
    23: 2,  # (5,4) ←
    24: 2,  # (5,5) ←
}

action_map_3 = {
    i: 0 for i in range(25)  # 所有状态都向右移动
}

# 随机移动
action_map_4 = {
    i: randint(0, 4) for i in range(25)
}
print("=" * 35 + "策略一" + "=" * 35)
solve(obstacles, goal_states, action_map)

print("\n" + "=" * 35 + "策略二" + "=" * 35)
solve(obstacles, goal_states, action_map_2)

print("\n" + "=" * 35 + "策略三：全部向右移动" + "=" * 35)
solve(obstacles, goal_states, action_map_3)

print("\n" + "=" * 35 + "策略四：随机移动" + "=" * 35)
solve(obstacles, goal_states, action_map_4)


print("\n" + "="*60)
print("图例说明：")
print("  [value]G  → 目标状态 (Goal)")
print("  (value)X  → 障碍区域 (Obstacle)")
print("   value    → 正常可达区域")
print("="*60)
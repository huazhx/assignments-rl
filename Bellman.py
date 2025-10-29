import numpy as np

class BellmanEquationSolver:
    """使用矩阵向量方式求解贝尔曼方程"""
    
    def __init__(self, P, R, gamma=0.9):
        """
        初始化求解器
        
        参数:
        P: 状态转移概率矩阵 (n_states, n_states)
           P[i,j] 表示从状态i转移到状态j的概率
        R: 奖励向量 (n_states,)
           R[i] 表示状态i的即时奖励
        gamma: 折扣因子 (0 <= gamma < 1)
        """
        self.P = np.array(P)
        self.R = np.array(R)
        self.gamma = gamma
        self.n_states = len(R)
        
    def solve_closed_form(self):
        """
        使用闭式解求解贝尔曼方程
        
        贝尔曼方程: V = R + γPV
        变换: V - γPV = R
              (I - γP)V = R
              V = (I - γP)^(-1)R
        
        返回: 状态值函数 V
        """
        I = np.eye(self.n_states)
        A = I - self.gamma * self.P
        V = np.linalg.solve(A, self.R)
        return V
    
    def solve_iterative(self, max_iter=1000, tol=1e-6, verbose=True):
        """
        使用迭代法求解贝尔曼方程
        
        迭代公式: V_{k+1} = R + γPV_k
        
        参数:
        max_iter: 最大迭代次数
        tol: 收敛阈值
        verbose: 是否打印收敛信息
        
        返回: (状态值函数 V, 迭代次数)
        """
        V = np.zeros(self.n_states)
        
        for i in range(max_iter):
            V_new = self.R + self.gamma * self.P @ V
            
            # 检查收敛
            delta = np.max(np.abs(V_new - V))
            if delta < tol:
                if verbose:
                    print(f"迭代法在第 {i+1} 次迭代后收敛，误差: {delta:.2e}")
                return V_new, i+1
            
            V = V_new
        
        if verbose:
            print(f"达到最大迭代次数 {max_iter}")
        return V, max_iter


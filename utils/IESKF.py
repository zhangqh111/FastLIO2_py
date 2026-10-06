import numpy as np

EPS = 1e-12

# 反对称矩阵
def skew(v):
    return np.array([[0.0, -v[2], v[1]],
                     [v[2], 0.0, -v[0]],
                     [-v[1], v[0], 0.0]])

# 对数映射(罗德里格斯公式) : 旋转向量 -> 旋转矩阵
def so3_exp(phi):
    theta = float(np.linalg.norm(phi))
    if theta < EPS:
        return np.eye(3) + skew(phi) 
    K = skew(phi / theta)
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)

# 对数映射 : 旋转矩阵 -> 旋转向量
def so3_log(R):
    c = min(1.0, max(-1.0, (np.trace(R) - 1.0) / 2.0))
    theta = float(np.arccos(c))
    v = np.array([R[2, 1] - R[1, 2],
                  R[0, 2] - R[2, 0],
                  R[1, 0] - R[0, 1]])
    if theta < EPS:
        return 0.5 * v                                   # 小角度一阶近似
    return v * (theta / (2.0 * np.sin(theta)))

# 世界系g的S2流形
def s2_basis(g):
    gu = g / np.linalg.norm(g)
    ref = np.array([0.0, 0.0, 1.0])
    if abs(gu @ ref) > 0.99:            # g 与 z 轴接近平行时换参考轴, 避免叉乘退化
        ref = np.array([1.0, 0.0, 0.0])
    b = np.cross(gu, ref)
    b = b / np.linalg.norm(b)
    c = np.cross(gu, b)
    return b, c

# 描述状态变量的esekfom类
class esekfom:
    IDX = {
        'pos':  slice(0, 3),
        'rot':  slice(3, 6),
        'R_LI': slice(6, 9),
        't_LI': slice(9, 12),
        'vel':  slice(12, 15),
        'bg':   slice(15, 18),
        'ba':   slice(18, 21),
        'grav': slice(21, 23),
    }

    def __init__(self, use_s2_gravity=True):
        self.use_s2_gravity = use_s2_gravity

        # ---- 名义状态 ----
        self.pos  = np.zeros(3)                        # 世界系位置
        self.rot  = np.eye(3)                          # IMU -> 世界 旋转 (SO3)        
        self.R_LI = np.eye(3)                              # LiDAR -> IMU 外参旋转 (SO3)
        self.t_LI = np.array([0.04165, 0.02326, -0.0284])  # LiDAR -> IMU 外参平移 (m)
        self.vel  = np.zeros(3)                        # 世界系速度
        self.bg   = np.zeros(3)                        # 陀螺零偏 (rad/s)
        self.ba   = np.zeros(3)                        # 加速度计零偏 (m/s^2)
        self.grav = np.array([0.0, 0.0, -9.81])        # 世界系重力 (|grav| = 9.81)

        # ---- 误差状态维数 (= 协方差 P / Q 的尺寸) ----
        self.dim = 23 if use_s2_gravity else 24

    def boxplus(self, delta):        
        delta = np.asarray(delta, dtype=np.float64)
        s = self.copy()
        s.pos  = self.pos + delta[0:3]
        s.rot  = self.rot @ so3_exp(delta[3:6])
        s.R_LI = self.R_LI @ so3_exp(delta[6:9])
        s.t_LI = self.t_LI + delta[9:12]
        s.vel  = self.vel + delta[12:15]
        s.bg   = self.bg + delta[15:18]
        s.ba   = self.ba + delta[18:21]        
        s.grav = self.grav
        return s
    
    def boxminus(self, other):        
        d = np.zeros(self.dim)
        d[0:3]   = self.pos - other.pos
        d[3:6]   = so3_log(other.rot.T @ self.rot)           # R_self = R_other * Exp(d)
        d[6:9]   = so3_log(other.R_LI.T @ self.R_LI)
        d[9:12]  = self.t_LI - other.t_LI
        d[12:15] = self.vel - other.vel
        d[15:18] = self.bg - other.bg
        d[18:21] = self.ba - other.ba    
        d[21:23] = 0,0
    
        return d

    def copy(self):
        """深拷贝一份状态 (boxplus / update_iekf 都要用)"""
        s = esekfom(self.use_s2_gravity)
        s.pos  = self.pos.copy()
        s.rot  = self.rot.copy()
        s.R_LI = self.R_LI.copy()
        s.t_LI = self.t_LI.copy()
        s.vel  = self.vel.copy()
        s.bg   = self.bg.copy()
        s.ba   = self.ba.copy()
        s.grav = self.grav.copy()
        if hasattr(self, 'cov'):
            s.cov = self.cov.copy()
        return s

    def __repr__(self):
        return (f"esekfom(dim={self.dim}, \npos={self.pos}, vel={self.vel}, "
                f"bg={self.bg}, ba={self.ba}, grav={self.grav},\n"
                f"===\trot\t===\n{self.rot},\n"
                f"===\tR_LI\t===\n{self.R_LI}, t_LI={self.t_LI})")
    
    def update_iekf(self, pl, normals, d_plane, sigma=0.05, max_iter=5, eps=1e-3):
        """
        用点面残差做迭代 EKF (IEKF) 观测更新, 结果原地写回 self。

        测量模型 (每个点一条一维观测, 观测量 z = 0):
            p_imu   = R_LI · p_L + t_LI
            P_world = R · p_imu + t
            h(x)    = nᵀ · P_world + d_plane          (平面: nᵀx + d_plane = 0)
            r       = z - h(x) = -h(x)

        参数:
            pl      : (M,3) 雷达系下的点
            normals : (M,3) 世界系单位法向量
            d_plane : (M,)  平面常数
            sigma   : 点面残差标准差 -> R = sigma^2 · I_M
            max_iter: 最大迭代次数(默认 5)
            eps     : 迭代增量阈值
        返回:
            info: {'iters', 'delta_norms', 'rms_before', 'rms_after'}
        """
        M = len(pl)
        n = self.dim
        P = self.cov                       # 先验协方差(迭代过程中保持不动)
        x_prior = self.copy()

        def rho_of(x):
            """点面残差 h(x)"""
            p_imu = pl @ x.R_LI.T + x.t_LI
            Pw = p_imu @ x.rot.T + x.pos
            return np.einsum('ij,ij->i', normals, Pw) + d_plane

        info = {'rms_before': float(np.sqrt(np.mean(rho_of(x_prior) ** 2))),
                'delta_norms': [], 'iters': 0}

        x = x_prior.copy()
        K = H = None
        for it in range(max_iter):
            # ---- 1. 在当前迭代点计算残差与雅可比 ----
            p_imu = pl @ x.R_LI.T + x.t_LI
            Pw = p_imu @ x.rot.T + x.pos
            r = -(np.einsum('ij,ij->i', normals, Pw) + d_plane)      # z - h(x), z = 0

            a = normals @ x.rot            # = (Rᵀn)ᵀ
            b = a @ x.R_LI                 # = ((R·R_LI)ᵀn)ᵀ
            H = np.zeros((M, n))
            H[:, 0:3]  = normals              # ∂r/∂pos
            H[:, 3:6]  = -np.cross(a, p_imu)  # ∂r/∂θ
            H[:, 6:9]  = -np.cross(b, pl)     # ∂r/∂R_LI (外参, 目前 P 为 0 不生效)
            H[:, 9:12] = a                    # ∂r/∂t_LI
            # vel / bg / ba / grav 列保持 0: 残差与它们无关

            # ---- 2. 卡尔曼增益: K = (H P Hᵀ + R)⁻¹ H P  ← FAST-LIO 的写法 ----
            S = H @ P @ H.T + np.eye(M) * (sigma ** 2)
            K = np.linalg.solve(S, H @ P).T                   # (n,M), 等价于 P Hᵀ S⁻¹

            # ---- 3. 迭代增量 + 状态更新 ----
            dvec = x.boxminus(x_prior)
            delta = K @ r - (np.eye(n) - K @ H) @ dvec
            x = x.boxplus(delta)
            info['delta_norms'].append(float(np.linalg.norm(delta)))
            info['iters'] = it + 1
            if info['delta_norms'][-1] < eps:
                break

        info['rms_after'] = float(np.sqrt(np.mean(rho_of(x) ** 2)))

        # ---- 4. 后验协方差 (Joseph 形式, 保证对称正定) ----
        IKH = np.eye(n) - K @ H
        P_post = IKH @ P @ IKH.T + K @ (np.eye(M) * (sigma ** 2)) @ K.T
        self.cov = 0.5 * (P_post + P_post.T)

        for f in ('pos', 'rot', 'R_LI', 't_LI', 'vel', 'bg', 'ba', 'grav'):
            setattr(self, f, getattr(x, f).copy())
        return info
# state = esekfom()
# print(state)
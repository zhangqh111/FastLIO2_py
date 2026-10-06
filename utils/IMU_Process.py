import numpy as np
from collections import deque
from IESKF import so3_exp, skew

G_m_s2 = 9.81

class Pose6D:
    def __init__(self, offset_time, acc_world, gyr, vel, pos, rot):
        self.offset_time = float(offset_time)              # 相对本帧点云起始时间(秒)
        self.acc_world = np.asarray(acc_world, dtype=np.float64)
        self.gyr = np.asarray(gyr, dtype=np.float64)       # 已去零偏的机体系角速度
        self.vel = np.asarray(vel, dtype=np.float64).copy()
        self.pos = np.asarray(pos, dtype=np.float64).copy()
        self.rot = np.asarray(rot, dtype=np.float64).copy()

    def __repr__(self):
        return (f"Pose6D(t={self.offset_time:.6f}, pos={np.round(self.pos, 5)}, "
                f"vel={np.round(self.vel, 5)})")

# 批量指数映射
def so3_exp_batch(w, dt):
    wn = float(np.linalg.norm(w))
    if wn < 1e-12:
        return np.eye(3)[None] + dt[:, None, None] * skew(w)[None]
    K = skew(w / wn)
    th = wn * dt
    return (np.eye(3)[None]
            + np.sin(th)[:, None, None] * K[None]
            + (1.0 - np.cos(th))[:, None, None] * (K @ K)[None])

class IMUProcess:
    def __init__(self):
        # ---- 初始化标志位: False 表示"还没初始化" ----
        self.Init = False

        # ---- 初始化阶段求出来的量 ----
        self.mean_acc = np.zeros(3)     # 第一帧加速度均值
        self.mean_gyr = np.zeros(3)     # 第一帧角速度均值 (rad/s)
        self.gravity  = np.zeros(3)     # 世界系重力向量, 对齐后 ≈ [0, 0, -9.81]
        self.bg       = np.zeros(3)     # 陀螺零偏估计 (rad/s)
        self.ba       = np.zeros(3)     # 加速度计零偏, 初始化阶段先取 0
        self.R0       = np.eye(3)       # 由重力方向得到的初始姿态 (可选, 见第 5 节)
        self.init_count = 0     
        self.last_imu = None    
        self.last_lidar_end_time = 0.0  # 上一帧点云中最后一个点的时间
        self.acc_scale = 1.0
        self.lidar_beg_time = 0.0
        self.imu_pose = []          # 本帧前向传播的状态快照列表 (每帧开头清空)
        self.pcl_undistorted = None 
        # ---- IMU 噪声 (与 FAST-LIO2 一致: 白噪声 + 零偏随机游走) ----
        self.noise_gyro = 0.1          # 陀螺仪白噪声
        self.noise_acc = 0.1           # 加速度计白噪声
        self.noise_gyro_bias = 0.0001  # 陀螺零偏随机游走
        self.noise_acc_bias = 0.0001   # 加速度零偏随机游走

    def imu_init(self, meas):        
        if self.Init:
            return False

        if len(meas.imu) == 0:
            return False                        # 没有 IMU, 不锁初始化, 等下一帧

        # ---- 2. 第一帧所有 IMU 取均值 ----
        acc_all = np.array([imu.linear_acceleration for imu in meas.imu])   # (N, 3)
        gyr_all = np.array([imu.angular_velocity    for imu in meas.imu])   # (N, 3)

        self.mean_acc = acc_all.mean(axis=0)
        self.mean_gyr = gyr_all.mean(axis=0)
        self.init_count = acc_all.shape[0]

        acc_norm = float(np.linalg.norm(self.mean_acc))
        self.acc_scale = G_m_s2 / acc_norm 
        if acc_norm < 1e-6:
            return False                        # 数据异常, 不锁初始化

        self.bg = self.mean_gyr.copy()
        self.gravity = -self.mean_acc * self.acc_scale
        self.ba = np.zeros(3)

        # ---- 4. 锁定初始化 ----
        self.Init = True
        print(f"[IMU init] 用 {self.init_count} 条 IMU 完成初始化")
        print(f"[IMU init] mean_gyr = {self.mean_gyr} rad/s  -> bg")
        print(f"[IMU init] mean_acc = {self.mean_acc}, |mean_acc| = {acc_norm:.4f}")
        print(f"[IMU init] gravity  = {self.gravity} m/s^2")
        return True
    
    # 反向传播
    def backward_undistortion(self, state, pcl):
        pcl_out = pcl.copy()
        if len(pcl_out) == 0 or len(self.imu_pose) < 2:
            return pcl_out
        
        R_e, p_e = state.rot, state.pos
        R_L, t_L = state.R_LI, state.t_LI

        t_pose = np.array([p.offset_time for p in self.imu_pose])        # (M,)
        t_pts = pcl_out['offset_time'].astype(np.float64) * 1e-9         # ns -> s, (N,)

        k_idx = np.searchsorted(t_pose, t_pts, side='left')
        np.clip(k_idx, 1, len(t_pose) - 1, out=k_idx)

        for k in np.unique(k_idx):
            m = (k_idx == k)
            head = self.imu_pose[k - 1]
            tail = self.imu_pose[k]

            dt = t_pts[m] - head.offset_time                             # (n,)
            # 该点时刻的姿态与位置 (批量外推)
            R_i = np.einsum('ij,njk->nik', head.rot, so3_exp_batch(tail.gyr, dt))
            p_i = (head.pos[None, :] + head.vel[None, :] * dt[:, None]
                   + 0.5 * tail.acc_world[None, :] * (dt ** 2)[:, None])

            P = np.column_stack([pcl_out['x'][m], pcl_out['y'][m], pcl_out['z'][m]])
            # ① 到本点时刻 IMU 系  ② 到世界(相对末端位置)  ③ 到末端雷达系
            W = np.einsum('nij,nj->ni', R_i, P @ R_L.T + t_L) + (p_i - p_e)
            P_end = (W @ R_e - t_L) @ R_L

            pcl_out['x'][m] = P_end[:, 0]
            pcl_out['y'][m] = P_end[:, 1]
            pcl_out['z'][m] = P_end[:, 2]

        return pcl_out

    # 前向传播 + 点云去畸变
    def undistort_pointcloud(self, state, imu_data, pcl=None):
        lidar_beg_time = self.lidar_beg_time
        
        v_imu = deque(imu_data)
        if self.last_imu is not None:
            v_imu.appendleft(self.last_imu)
        self.imu_pose = []
        if self.last_imu is not None:
            acc0 = (self.last_imu.linear_acceleration - state.ba) * self.acc_scale
            gyr0 = self.last_imu.angular_velocity - state.bg
            self.imu_pose.append(Pose6D(0.0,
                                        state.rot @ acc0 + state.grav,
                                        gyr0,
                                        state.vel, state.pos, state.rot))
        if len(v_imu) < 2:
            return state 
        
        for i in range(len(v_imu) - 1):
            head = v_imu[i]
            tail = v_imu[i + 1]
            if tail.timestamp < self.last_lidar_end_time:
                continue

            angvel_avr = (head.angular_velocity + tail.angular_velocity) * 0.5 - state.bg
            acc_avr    = (head.linear_acceleration + tail.linear_acceleration) * 0.5 - state.ba
            acc_avr    = acc_avr * self.acc_scale          # 原始单位 -> m/s^2
            if head.timestamp < self.last_lidar_end_time:                
                dt = tail.timestamp - self.last_lidar_end_time
            else:    
                dt = tail.timestamp - head.timestamp

            if dt <= 0.0:
                continue
            acc_world = state.rot @ acc_avr + state.grav
            self.predict_covariance(state, acc_avr, angvel_avr, dt)
            state.pos = state.pos + state.vel * dt + 0.5 * acc_world * dt * dt
            state.vel = state.vel + acc_world * dt
            state.rot = state.rot @ so3_exp(angvel_avr * dt)   # 机体系小量, 右乘

            self.imu_pose.append(Pose6D(tail.timestamp - lidar_beg_time,
                                        acc_world, angvel_avr,
                                        state.vel, state.pos, state.rot))
        
        if pcl is not None and len(pcl) > 0 and len(self.imu_pose) >= 2:
            pcl_end_time = float(pcl['offset_time'].max()) * 1e-9     # 相对本帧起始
            last_pose = self.imu_pose[-1]
            dt = pcl_end_time - last_pose.offset_time
            if dt > 1e-9:
                # 用最后一段区间的平均测量做零阶保持外推
                angvel_avr = last_pose.gyr
                acc_world = last_pose.acc_world
                acc_avr_ex = state.rot.T @ (acc_world - state.grav)   # 换回机体系
                self.predict_covariance(state, acc_avr_ex, angvel_avr, dt)
                state.pos = state.pos + state.vel * dt + 0.5 * acc_world * dt * dt
                state.vel = state.vel + acc_world * dt
                state.rot = state.rot @ so3_exp(angvel_avr * dt)
                self.imu_pose.append(Pose6D(pcl_end_time, acc_world, angvel_avr,
                                            state.vel, state.pos, state.rot))

        # ---- 5. 反向传播去畸变 (参考系 = 上面的帧末状态) ----
        if pcl is not None and len(pcl) > 0:
            self.pcl_undistorted = self.backward_undistortion(state, pcl)
        return state

    def imu_process(self, meas, state):
        if not self.Init:
            if self.imu_init(meas):                      # 复用你已有的初始化函数
                state.bg   = self.bg.copy()              # 角速度均值 -> 陀螺零偏
                state.grav = self.gravity.copy()         # 加速度均值算出的世界系重力
                self.last_imu = meas.imu[-1]
                self.last_lidar_end_time = meas.lidar_end_time
                print(f"[imu_process] 初始化写入 state: "
                      f"bg={state.bg}, grav={state.grav}")
                return True
            return False         

        self.lidar_beg_time = meas.lidar_beg_time
        self.undistort_pointcloud(state, meas.imu, meas.lidar)
        if len(meas.imu) > 0:
            self.last_imu = meas.imu[-1]
        self.last_lidar_end_time = meas.lidar_end_time
        return False
    
    def predict_covariance(self, state, acc_avr, angvel_avr, dt):
        """
        IMU 区间上的协方差传播: P <- Fx · P · Fx^T + Q
        (观测融合之前得到的就是这个先验协方差)

        必须在*更新名义状态之前*调用 —— Fx 里的 R 要用区间"起点"的姿态。

        参数:
            state      : 当前状态 (用到 state.rot / state.cov / state.dim)
            acc_avr    : 该区间已去零偏、已换算到 m/s^2 的机体系加速度
            angvel_avr : 该区间已去零偏的机体系角速度
            dt         : 区间时长
        返回: 新的 P (dim x dim), 同时写回 state.cov
        """
        n = state.dim
        R = state.rot
        s = self.acc_scale                 # 原始加速度 -> m/s^2 的换算系数

        # ---- Fx: 误差状态顺序 = (pos, rot, R_LI, t_LI, vel, bg, ba, grav) ----
        Fx = np.eye(n)
        Fx[0:3, 12:15]   = np.eye(3) * dt                            # dpos <- dvel
        Fx[0:3, 3:6]     = -0.5 * (R @ skew(acc_avr)) * dt * dt      # dpos <- dtheta
        Fx[0:3, 18:21]   = -0.5 * R * s * dt * dt                    # dpos <- dba
        E = so3_exp(-angvel_avr * dt)
        Fx[3:6, 3:6]     = E                                          # dtheta <- dtheta
        Fx[3:6, 15:18]   = -E * dt                                    # dtheta <- dbg
        Fx[12:15, 3:6]   = -(R @ skew(acc_avr)) * dt                  # dvel <- dtheta
        Fx[12:15, 18:21] = -R * s * dt                                # dvel <- dba
        # 其余块: R_LI/t_LI 常数, bg/ba 随机游走, grav 冻结 -> 都是单位阵

        # ---- Q: 过程噪声 (FAST-LIO 的离散形式) ----
        Q = np.zeros((n, n))
        Q[3:6, 3:6]     = np.eye(3) * self.noise_gyro * dt * dt
        Q[12:15, 12:15] = np.eye(3) * self.noise_acc * dt * dt
        Q[15:18, 15:18] = np.eye(3) * self.noise_gyro_bias * dt
        Q[18:21, 18:21] = np.eye(3) * self.noise_acc_bias * dt

        P_new = Fx @ state.cov @ Fx.T + Q
        P_new = 0.5 * (P_new + P_new.T)        # 强制对称, 抑制数值漂移
        state.cov = P_new
        return P_new
    
if __name__ == '__main__':
    t_pose = np.array([0, 1, 2, 3, 4])
    t_pts = np.array([0.1, 0.4, 0.9, 1.3, 1.6, 1.7, 2.1, 2.2, 2.4, 3.4, 3.5, 4.1])

    k_idx = np.searchsorted(t_pose, t_pts, side='left')
    print(f"k_idx : {k_idx}")
    # np.clip(k_idx, 1, len(t_pose) - 1, out=k_idx)

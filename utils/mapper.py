from collections import deque
from pathlib import Path
import numpy as np
from LiDAR2bin import load_lidar_frame_pair
from IMU2bin import load_imu_between
import open3d as o3d

LIDAR_BIN = Path(r'F:\SLAM\SLAM_demo\FastLIO2\data\lidar_data.bin')
IMU_BIN = Path(r'F:\SLAM\SLAM_demo\FastLIO2\data\imu_data.bin')

class IMUData:
    def __init__(self, timestamp, angular_velocity, linear_acceleration):
        self.timestamp = float(timestamp)
        self.angular_velocity = np.asarray(angular_velocity, dtype=np.float64)
        self.linear_acceleration = np.asarray(linear_acceleration, dtype=np.float64)
    
    def __repr__(self):
        return (f"IMUData(t={self.timestamp:.6f}, w={self.angular_velocity}, "
                f"a={self.linear_acceleration})")

class MeasureGroup:
    def __init__(self):
        self.lidar_beg_time = 0.0   # 本帧雷达开始时间(秒)
        self.lidar_end_time = 0.0   # 本帧最后一个雷达点的时间(秒)
        self.lidar = None           # 结构化数组, 字段 x, y, z, reflectivity, offset_time
        self.imu = deque()          # 元素为 IMUData, 按时间递增

    def __repr__(self):
        n = 0 if self.lidar is None else self.lidar.shape[0]
        return (f"MeasureGroup(beg={self.lidar_beg_time:.6f}, "
                f"end={self.lidar_end_time:.6f}, points={n}, imu={len(self.imu)})")

def sync_packages(frame_index, lidar_bin=LIDAR_BIN, imu_bin=IMU_BIN,
                  imu_time_lower=None, include_end=False):
    ts_cur, points, _ts_next = load_lidar_frame_pair(lidar_bin, frame_index)
    meas = MeasureGroup()
    meas.lidar = points
    meas.lidar_beg_time = float(ts_cur)

    if points.shape[0] > 0:
        last_offset_ns = int(points['offset_time'][-1])   # 点云按时间递增排列, 末点即最晚
        meas.lidar_end_time = meas.lidar_beg_time + last_offset_ns * 1e-9
    else:
        meas.lidar_end_time = meas.lidar_beg_time 
    t_lower = meas.lidar_beg_time if imu_time_lower is None else float(imu_time_lower)
    imu_ts, imu_gyro, imu_accel = load_imu_between(
        imu_bin, t_lower, meas.lidar_end_time, include_end=include_end
    )

    meas.imu = deque(
        IMUData(imu_ts[i], imu_gyro[i], imu_accel[i])
        for i in range(imu_ts.shape[0])
    )

    return meas

def _extract_xyz(pcl):    
    arr = np.asarray(pcl)
    if arr.dtype.names is not None:                      # 结构化数组(带 x/y/z 字段)
        return np.column_stack([arr['x'], arr['y'], arr['z']]).astype(np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"点云形状应为 (N,3), 实际为 {arr.shape}")
    return arr.astype(np.float64)

def voxel_filter(pcl, voxel_size):
    pts = _extract_xyz(pcl)
    if len(pts) == 0:
        return np.zeros((0, 3), dtype=np.float64)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd_down = pcd.voxel_down_sample(voxel_size=float(voxel_size))
    return np.asarray(pcd_down.points, dtype=np.float64)
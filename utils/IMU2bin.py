# 将IMU数据转换为bin格式保存起来用于离线调试
# IMU数据格式 : 每条IMU数据7条数据 (时间戳 * 1 + 角速度 * 3 + 加速度 * 3)
# 时间戳使用float64, 角速度/加速度使用float32

import os
import numpy as np
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore

def extract_imu_to_bin(bag_path, output_bin, imu_topic='/livox/imu'):
    typestore = get_typestore(Stores.ROS1_NOETIC)
    timestamps = []   # 秒 (float64)
    gyro_list = []    # 角速度 (N,3) float32
    accel_list = []   # 线加速度 (N,3) float32

    with Reader(bag_path) as reader:
        for connection, timestamp, rawdata in reader.messages():
            if connection.topic == imu_topic:
                # 反序列化消息
                msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
                
                # 时间戳：rosbags 返回的是纳秒 (int)，转换为秒 (float)
                st = msg.header.stamp
                ts_sec = st.sec + st.nanosec * 1e-9
                timestamps.append(ts_sec)
                
                # 角速度和线加速度
                gyro_list.append([
                    msg.angular_velocity.x,
                    msg.angular_velocity.y,
                    msg.angular_velocity.z
                ])
                accel_list.append([
                    msg.linear_acceleration.x,
                    msg.linear_acceleration.y,
                    msg.linear_acceleration.z
                ])

    # 转换为 NumPy 数组
    timestamps = np.array(timestamps, dtype=np.float64)
    gyro = np.array(gyro_list, dtype=np.float32)     # 形状 (N,3)
    accel = np.array(accel_list, dtype=np.float32)   # 形状 (N,3)

    dtype = np.dtype([
        ('ts', np.float64),          # 8 字节
        ('gyro', np.float32, (3,)),  # 12 字节
        ('accel', np.float32, (3,))  # 12 字节
    ])

    assert dtype.itemsize == 32, f"Unexpected itemsize: {dtype.itemsize}"

    data = np.zeros(len(timestamps), dtype=dtype)
    data['ts'] = timestamps
    data['gyro'] = gyro
    data['accel'] = accel

    data.tofile(output_bin)
    print(f"已保存 {len(timestamps)} 条 IMU 数据到 {output_bin}")
    print(f"文件大小: {len(timestamps) * 32} 字节")

def load_imu_from_bin(bin_path, index):

    dtype = np.dtype([
        ('ts', np.float64),
        ('gyro', np.float32, (3,)),
        ('accel', np.float32, (3,))
    ])
    record_size = dtype.itemsize
    file_size = os.path.getsize(bin_path)
    total_records = file_size // record_size

    if index < 0 or index >= total_records:
        raise IndexError(f"索引越界: 请求 index={index}，但文件有效范围为 0~{total_records - 1}")
    try:
        with open(bin_path, 'rb') as f:
            f.seek(index * record_size)
            raw = f.read(record_size)
            if len(raw) < record_size:
                raise IndexError(f"索引 {index} 超出文件记录范围")
            record = np.frombuffer(raw, dtype=dtype, count=1)[0]
    except FileNotFoundError:
        raise FileNotFoundError(f"文件不存在: {bin_path}")
    
    timestamp = record['ts']
    gyro = record['gyro']
    accel = record['accel']
    return timestamp, gyro, accel

LIDAR_TIME_BASE = 946685437.499511
_IMU_CACHE = {}
def _load_imu_all(bin_path):
    key = (str(bin_path), os.path.getsize(bin_path))
    data = _IMU_CACHE.get(key)
    if data is None:
        dtype = np.dtype([
            ('ts', np.float64),
            ('gyro', np.float32, (3,)),
            ('accel', np.float32, (3,))
        ])
        data = np.fromfile(bin_path, dtype=dtype)
        _IMU_CACHE[key] = data
    return data

def load_imu_between(bin_path, t_start, t_end, time_base=LIDAR_TIME_BASE, include_end=True):
    if t_end < t_start:
        t_start, t_end = t_end, t_start
    data = _load_imu_all(bin_path)
    ts_rel = data['ts'] - time_base          # 平移到与 LiDAR 相同的相对时间轴
    i0 = int(np.searchsorted(ts_rel, t_start, side='left'))
    i1 = int(np.searchsorted(ts_rel, t_end, side='right' if include_end else 'left'))
    return ts_rel[i0:i1], data['gyro'][i0:i1].copy(), data['accel'][i0:i1].copy()

if __name__ == "__main__":
    bag_path = r'F:\dataset\Retail_Street.bag'
    output_bin = r'F:\SLAM\SLAM_demo\FastLIO2\data\imu_data.bin'
    # extract_imu_to_bin(bag_path, output_bin)
    ts_cur = 0
    ts_next = 0.1
    imu_ts, imu_gyro, imu_accel = load_imu_between(output_bin, ts_cur, ts_next)
    print(f"    区间 [{ts_cur:.6f}, {ts_next:.6f}] 内 IMU 条数 = {imu_ts.shape[0]}")
    # ts, gyro, accel = load_imu_from_bin(output_bin, 0)
    # print(f"时间戳: {ts:.6f} s")
    # print(f"角速度: {gyro} rad/s")
    # print(f"加速度: {accel} g")
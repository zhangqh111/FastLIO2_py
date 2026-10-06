import numpy as np
import open3d as o3d
import os
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore, get_types_from_msg

# 检查Lidar数据能否正常从bag包中读取
def inspect_lidar_message(bag_path, topic='/livox/lidar'):
    typestore = get_typestore(Stores.ROS1_NOETIC)
    with Reader(bag_path) as reader:
        target_connection = None
        for connection in reader.connections:
            if connection.topic == topic:
                target_connection = connection
                print(f"话题: {target_connection.topic}")
                print(f"消息类型: {target_connection.msgtype}")
                if target_connection.msgtype == 'livox_ros_driver/msg/CustomMsg':
                    print(f"注册Livox雷达的消息类型")
                    custom_types = get_types_from_msg(target_connection.msgdef.data, target_connection.msgtype)
                    typestore.register(custom_types) 
                msgdef = typestore.get_msgdef(target_connection.msgtype)
                print(msgdef)
                break

        if target_connection is None:
            print(f"未找到话题 {topic}")
            return

# 看下header.stamp和timebase的区别    
def inspect_lidar_times(bag_path, topic='/livox/lidar', frame_indices=None):
    if frame_indices is None:
        frame_indices = [0, 1, 2, 3, 4]
    elif isinstance(frame_indices, int):
        frame_indices = [frame_indices]
    else:
        frame_indices = list(frame_indices)
    target_indices = set(frame_indices)

    typestore = get_typestore(Stores.ROS1_NOETIC)

    with Reader(bag_path) as reader:
        target_conn = None
        for connection in reader.connections:
            if connection.topic == topic:
                target_conn = connection
                break
        
        if target_conn is None:
            print(f"未找到话题 {topic}")
            return
        
        try:
            custom_types = get_types_from_msg(target_conn.msgdef.data, target_conn.msgtype)
            typestore.register(custom_types)
        except Exception as e:
            print(f"注册自定义消息类型失败: {e}")

        print(f"话题: {target_conn.topic}")
        print(f"消息类型: {target_conn.msgtype}\n")

        frame_count = 0
        max_frame = max(target_indices) if target_indices else -1

        for connection, timestamp, rawdata in reader.messages():
            if connection.topic != topic:
                continue

            if frame_count in target_indices:
                msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
                stamp = msg.header.stamp
                if hasattr(stamp, 'sec') and hasattr(stamp, 'nanosec'):
                    header_sec = stamp.sec + stamp.nanosec * 1e-9
                    header_raw = f"{stamp.sec}.{stamp.nanosec:09d}"
                else:
                    header_raw = str(stamp)
                    header_sec = float(stamp) * 1e-9 if isinstance(stamp, (int, float)) else float('nan')
                timebase = msg.timebase
                timebase_sec = timebase * 1e-9

                print(f"Frame {frame_count}:")
                print(f"  header.stamp = {header_raw}  ({header_sec:.9f} s)")
                print(f"  timebase     = {timebase}  ({timebase_sec:.9f} s)")
                print(f"  差值 (header - timebase) = {(header_sec - timebase_sec)*1e9:.0f} ns\n")

            frame_count += 1
            # 如果已经处理完所有目标帧，提前退出
            if max_frame >= 0 and frame_count > max_frame:
                break

        if frame_count == 0:
            print("没有读取到该话题的消息")

# 将雷达提取成bin文件
def extract_lidar_to_bin(bag_path, output_bin, topic='/livox/lidar'):
    typestore = get_typestore(Stores.ROS1_NOETIC)
    with Reader(bag_path) as reader:
        target_conn = None
        for connection in reader.connections:
            if connection.topic == topic:
                target_conn = connection
                break
        if target_conn is None:
            print(f"未找到话题 {topic}")
            return
        
        try:
            custom_types = get_types_from_msg(target_conn.msgdef.data, target_conn.msgtype)
            typestore.register(custom_types)
        except Exception as e:
            print(f"注册自定义消息类型失败: {e}")
            return
        print(f"开始转换话题: {target_conn.topic} (类型: {target_conn.msgtype})")
        point_dtype = np.dtype([
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
            ('reflectivity', np.uint8),
            ('offset_time', np.uint32)
        ])
        frame_count = 0
        total_points = 0

        with open(output_bin, 'wb') as f:
            f.write(np.uint32(0).tobytes())
            for connection, timestamp, rawdata in reader.messages():
                if connection.topic != topic:
                    continue
                msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
                stamp = msg.header.stamp
                ts_sec = stamp.sec + stamp.nanosec * 1e-9
                point_num = msg.point_num
                if point_num <= 0:
                    continue
                f.write(np.float64(ts_sec).tobytes())
                f.write(np.uint32(point_num).tobytes())
                # 提取点数据并构造结构化数组
                points = msg.points
                # 预先分配数组
                point_array = np.empty(point_num, dtype=point_dtype)
                # 使用列表推导提取各字段（注意：字段名需与 CustomPoint 一致）
                point_array['x'] = [p.x for p in points]
                point_array['y'] = [p.y for p in points]
                point_array['z'] = [p.z for p in points]
                point_array['reflectivity'] = [p.reflectivity for p in points]
                point_array['offset_time'] = [p.offset_time for p in points]
                f.write(point_array.tobytes())
                frame_count += 1
                total_points += point_num
                if frame_count % 100 == 0:
                    print(f"已处理 {frame_count} 帧，累计点数 {total_points}")
            # 回填总帧数
            f.seek(0)
            f.write(np.uint32(frame_count).tobytes())
    print(f"转换完成：共 {frame_count} 帧，总点数 {total_points}")
    print(f"输出文件: {output_bin}")

def load_lidar_frame(bin_path, frame_index):
    with open(bin_path, 'rb') as f:
        f.seek(0)
        total_frames = np.fromfile(f, dtype=np.uint32, count=1)[0]
        if frame_index >= total_frames:
            raise IndexError(f"帧索引 {frame_index} 超出范围 (0~{total_frames-1})")
        
        f.seek(4)
        for i in range(frame_index):
            # 读取时间戳和点数
            ts = np.fromfile(f, dtype=np.float64, count=1)[0]
            point_num = np.fromfile(f, dtype=np.uint32, count=1)[0]
            f.seek(point_num * 17, 1)  # 跳过点数据

        ts = np.fromfile(f, dtype=np.float64, count=1)[0]
        point_num = np.fromfile(f, dtype=np.uint32, count=1)[0]
        point_dtype = np.dtype([
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
            ('reflectivity', np.uint8),
            ('offset_time', np.uint32)
        ])
        point_data = np.fromfile(f, dtype=point_dtype, count=point_num)
        return ts, point_data

_LIDAR_INDEX_CACHE = {}
# 建立bin文件每帧的索引
def _build_lidar_index(bin_path):
    key = (str(bin_path), os.path.getsize(bin_path))
    if key in _LIDAR_INDEX_CACHE:
        return _LIDAR_INDEX_CACHE[key]
    entries = []
    with open(bin_path, 'rb') as f:
        total_frames = int(np.fromfile(f, dtype=np.uint32, count=1)[0])
        for _ in range(total_frames):
            ts = float(np.fromfile(f, dtype=np.float64, count=1)[0])
            point_num = int(np.fromfile(f, dtype=np.uint32, count=1)[0])
            entries.append((f.tell(), ts, point_num))
            f.seek(point_num * 17, 1)
    _LIDAR_INDEX_CACHE[key] = entries
    # print(entries[0][1])
    return entries

#  读取 lidar_data.bin 的第 frame_index 帧点云和时间戳, 并返回下一帧的时间戳
def load_lidar_frame_pair(bin_path, frame_index):
    entries = _build_lidar_index(bin_path)
    if frame_index < 0 or frame_index >= len(entries):
        raise IndexError(f"帧索引越界: {frame_index}, 有效范围 0~{len(entries) - 1}")
    offset, ts_cur, point_num = entries[frame_index]
    ts_cur = ts_cur - entries[0][1]
    point_dtype = np.dtype([('x', np.float32), ('y', np.float32), ('z', np.float32),
                            ('reflectivity', np.uint8), ('offset_time', np.uint32)])
    with open(bin_path, 'rb') as f:
        f.seek(offset)
        points = np.fromfile(f, dtype=point_dtype, count=point_num)
    ts_next = entries[frame_index + 1][1] - entries[0][1] if frame_index + 1 < len(entries) else None
    return ts_cur, points, ts_next

# 通过bin文件读取特定雷达帧的点云
def visualize_lidar_frame(point_data, stamp = None, use_intensity_color=True):
    xyz = np.column_stack((point_data['x'], point_data['y'], point_data['z']))
    xyz = xyz.astype(np.float64)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if use_intensity_color:
        # 将反射强度归一化到 [0,1] 并映射为颜色（例如使用蓝色到红色）
        intensity = point_data['reflectivity'].astype(np.float32)
        intensity_max = intensity.max() if intensity.size > 0 else 1
        intensity_norm = intensity / intensity_max if intensity_max > 0 else intensity
        # 创建颜色映射（简单示例：蓝色低强度，红色高强度）
        colors = np.zeros((len(intensity), 3))
        colors[:, 0] = intensity_norm          # R
        colors[:, 2] = 1.0 - intensity_norm    # B
        # 绿色通道设为0，得到蓝-红渐变
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    else:
        # 统一颜色（白色）
        pcd.paint_uniform_color([1.0, 1.0, 1.0])

    # 可视化
    o3d.visualization.draw_geometries([pcd], window_name=f"Livox Point Cloud (stamp={stamp:.3f}s)")

if __name__ == "__main__":
    bag_path = r'F:\dataset\Retail_Street.bag'
    output_bin = r'F:\SLAM\SLAM_demo\FastLIO2\data\lidar_data.bin'
    # inspect_lidar_message(bag_path, '/livox/lidar')   # 看下livox_ros_driver/msg/CustomMsg的数据类型
    # extract_lidar_to_bin(bag_path, output_bin, topic='/livox/lidar')  # 将bag中的点云存储为bin文件
    # inspect_lidar_times(bag_path, '/livox/lidar', frame_indices=0)    # 看下header.stamp和timebase的区别
    # stamp, point_data = load_lidar_frame(output_bin, 500)
    # visualize_lidar_frame(point_data)
    ts_cur, points_cur, ts_next = load_lidar_frame_pair(output_bin, 0) 
    print(f"当前帧时间戳 : {ts_cur}, 下一帧时间戳 : {ts_next}")
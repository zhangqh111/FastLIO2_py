from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore
import numpy as np
from pathlib import Path
import time
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent / 'utils'))
from LiDAR2bin import load_lidar_frame_pair
from LiDAR2bin import load_lidar_frame_pair, _build_lidar_index
from mapper import sync_packages, voxel_filter
from IMU_Process import IMUProcess
from IESKF import esekfom
from visualize import visualize_position, save_visualization, reset_history, visualize_pointcloud, RealtimeVisualizer
import open3d as o3d
from IKDTree import IKDTree, Node, fit_plane_residual

LIDAR_BIN = Path(r'F:\SLAM\SLAM_demo\FastLIO2\data\lidar_data.bin')
IMU_BIN = Path(r'F:\SLAM\SLAM_demo\FastLIO2\data\imu_data.bin')
LIDAR_TIME_BASE = _build_lidar_index(LIDAR_BIN)[0][1]

fps = 100
interval = 1.0 / fps 

frame = 1
frame_max = 1000
prev_end = None
imu_proc = IMUProcess()
state = esekfom()
state.cov = np.eye(state.dim) * 1e-4      # 初始不确定度, 按你自己的初始化精度取
state.cov[21:23, :] = 0.0                 # 重力冻结、不估计 -> 保持 0
state.cov[:, 21:23] = 0.0
state.cov[6:12, :] = 0.0                  # 外参当常数 -> 保持 0
state.cov[:, 6:12] = 0.0
map_point_num = 0

reset_history()
VOXEL_SIZE = 0.5
ikd_tree = IKDTree()
rt = RealtimeVisualizer(window_name='FastLIO2 realtime  (blue=map, red=scan, green=traj)')
ikd_tree.depth_debug = 5
while True:    
    start = time.perf_counter()
    print(f"\n{'-' * 100}")
    
    meas = sync_packages(frame - 1, imu_time_lower=prev_end)
    prev_end = meas.lidar_end_time
    print(f"    MeasureGroup: beg={meas.lidar_beg_time:.6f} end={meas.lidar_end_time:.6f} "
          f"点数={meas.lidar.shape[0]} IMU={len(meas.imu)}")
    
    initialized = imu_proc.imu_process(meas, state)      

    undistorted_cloud = getattr(imu_proc, 'pcl_undistorted', None)
    if undistorted_cloud is not None:        
        pts_down = voxel_filter(undistorted_cloud, VOXEL_SIZE)
        
        R_wl = state.rot @ state.R_LI
        t_wl = state.pos + state.rot @ state.t_LI
        pts_world = pts_down @ R_wl.T + t_wl

        if frame == 2:            
            data = pts_world.copy()                 
            ikd_tree.root = Node(np.zeros(3))
            ikd_tree.build_tree(ikd_tree.root, data, 0, len(data) - 1, 0)
            map_point_num = len(data)
            print(f"[map] 第2帧建图: {map_point_num} 点入树 (体素后 {len(pts_down)} 点)")
        
        else:
            K_NEAREST = 5
            MAX_DIST_SQ = 5.0
            valid, valid_num, neighbors = ikd_tree.SearchCloudVec(pts_world, K_NEAREST, MAX_DIST_SQ)
            
            idx_keep = []; nrm = []; dpl = []
            for i in np.where(valid)[0]:
                nb = neighbors[i].sorted_items()
                p = pts_world[i]
                r = fit_plane_residual(p, nb, plane_threshold=0.1, residual_ratio=1.0 / 9.0,
                                       range_ref=float(np.linalg.norm(p - t_wl)))
                if r is None:
                    continue
                idx_keep.append(i); nrm.append(r[2]); dpl.append(r[3])
            idx_keep = np.array(idx_keep, dtype=int)
            nrm = np.array(nrm); dpl = np.array(dpl)
            pl = pts_down[idx_keep]
            print(f"[plane] 参与点面残差: {len(idx_keep)} 点")
            print(f"第{frame}帧处理结束")
            
            t4 = time.perf_counter()
            info = state.update_iekf(pl, nrm, dpl, sigma=0.05, max_iter=5, eps=1e-3)
            t5 = time.perf_counter()
            print(f"[iekf] 迭代 {info['iters']} 次, 增量 {['%.2e' % v for v in info['delta_norms']]}")
            print(f"[iekf] 残差 RMS {info['rms_before']:.5f} -> {info['rms_after']:.5f} m, 用时 {t5-t4:.3f} s")
            print(f"[iekf] 更新后 pos = {np.round(state.pos, 6)}")

            # ---- 用迭代更新后的位姿把当前帧投影进全局地图, 并整体体素降采样 ----
            R_wl_u = state.rot @ state.R_LI
            t_wl_u = state.pos + state.rot @ state.t_LI
            pts_map = pts_down @ R_wl_u.T + t_wl_u            # 注意用*更新后*的位姿

            t6 = time.perf_counter()
            n_before = map_point_num
            map_point_num = ikd_tree.add_map_points(pts_map, VOXEL_SIZE, voxel_fn=voxel_filter)
            t7 = time.perf_counter()
            print(f"[map+] {n_before}(旧) + {len(pts_map)}(当前帧) -> {map_point_num} 点"
                  f" (体素 {VOXEL_SIZE} m), 用时 {t7 - t6:.4f} s")
        
        # ---- 实时可视化(非阻塞): 全局地图 + 当前帧世界系点云 + IMU 轨迹 ----
        rt.update(map_pts=ikd_tree._collect_points(),
                  scan_pts=pts_world,          # 当前帧世界系稀疏点云
                  pos=state.pos,
                  rot=state.rot)

    # visualize_position(state, frame=frame - 1)
    # print(f"位置 : (x, y, z) = ({state.pos[0]}, {state.pos[1]}, {state.pos[2]})\n")
    print(f"{'-' * 100}\n")

    frame += 1
    elapsed = time.perf_counter() - start
    sleep_time = interval - elapsed
    if sleep_time > 0:
        time.sleep(sleep_time)
    if frame > frame_max:        
        # ---- 全局地图点 (世界系) ----
        map_pts = ikd_tree._collect_points()
        if len(map_pts) == 0:
            print("[map] 地图为空, 没有可显示的点")
        else:
            # visualize_pointcloud 要的是带 x/y/z 字段的结构化数组
            map_vis = np.zeros(len(map_pts), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
            map_vis['x'] = map_pts[:, 0]
            map_vis['y'] = map_pts[:, 1]
            map_vis['z'] = map_pts[:, 2]
            print(f"[map] 全局地图 {len(map_pts)} 点 (世界系)")
            visualize_pointcloud(map_vis, None, frame=frame - 1, point_size=2.0,
                                 window_name=f'global map ({len(map_pts)} pts)')
        print(f"第{frame_max}帧处理结束,退出")
        rt.close()
        break
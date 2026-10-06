from pathlib import Path
import csv
import numpy as np
import matplotlib
# matplotlib.use('Agg')                      # 只存文件不开窗口, 必须在 pyplot 之前
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import open3d as o3d

SAVE_DIR = Path(r'F:\SLAM\SLAM_demo\data\visualize')

_history = []

def reset_history():
    _history.clear()

# 记录位置数据
def visualize_position(state, frame=None, save_dir=SAVE_DIR, save_each_frame=False):
    pos = np.asarray(state.pos, dtype=np.float64)
    if frame is None:
        frame = len(_history)
    _history.append((int(frame), float(pos[0]), float(pos[1]), float(pos[2])))
    print(f"[record] frame {frame}: pos = ({pos[0]:+.6f}, {pos[1]:+.6f}, {pos[2]:+.6f}) m")
    return pos

def _plot_trajectory(frames, arr, label):
    fig = plt.figure(figsize=(15, 4.5))
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.plot(arr[:, 0], arr[:, 1], '-o', ms=1, lw=1.2)
    ax1.plot(arr[0, 0], arr[0, 1], 'go', ms=3, label='start')
    ax1.plot(arr[-1, 0], arr[-1, 1], 'rs', ms=3, label='end')
    ax1.set_xlabel('x [m]'); ax1.set_ylabel('y [m]')
    ax1.set_title(f'Top view (XY), {label}')
    ax1.grid(True); ax1.legend(loc='best')
    if arr.shape[0] > 1:
        ax1.set_aspect('equal', adjustable='datalim')
    else:                                   # 只有一个点时手工给范围, 避免空坐标轴
        ax1.set_xlim(arr[0, 0] - 1e-3, arr[0, 0] + 1e-3)
        ax1.set_ylim(arr[0, 1] - 1e-3, arr[0, 1] + 1e-3)

    # (2) x / y / z 随帧变化
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.plot(frames, arr[:, 0], '-o', ms=1, label='x')
    ax2.plot(frames, arr[:, 1], '-o', ms=1, label='y')
    ax2.plot(frames, arr[:, 2], '-o', ms=1, label='z')
    ax2.set_xlabel('frame'); ax2.set_ylabel('position [m]')
    ax2.set_title('Position vs frame')
    ax2.grid(True); ax2.legend(loc='best')

    # (3) 3D 轨迹
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    ax3.plot(arr[:, 0], arr[:, 1], arr[:, 2], '-o', ms=3)
    ax3.set_xlabel('x [m]'); ax3.set_ylabel('y [m]'); ax3.set_zlabel('z [m]')
    ax3.set_title('3D trajectory')

    fig.tight_layout()
    return fig

# 保存图片
def save_visualization(save_dir=SAVE_DIR, save_each_frame=False):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if len(_history) == 0:
        print("[visualize] history 为空, 没有可保存的数据")
        return
    
    frames = np.array([h[0] for h in _history])
    arr = np.array([[h[1], h[2], h[3]] for h in _history])      # (N, 3)

    # ---- 1. 写 CSV ----
    csv_path = save_dir / 'onlyIMU_positions.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['frame', 'x', 'y', 'z'])
        for fr, x, y, z in _history:
            writer.writerow([fr, f'{x:.9f}', f'{y:.9f}', f'{z:.9f}'])

    # ---- 2. 总轨迹图 ----
    fig = _plot_trajectory(frames, arr, f'frame {frames[0]}~{frames[-1]}')
    fig.savefig(save_dir / 'onlyIMU_trajectory.png', dpi=120)
    plt.close(fig)

    print(f"[visualize] 共 {len(_history)} 帧位置已保存:")
    print(f"            数据: {csv_path}")
    print(f"            图片: {save_dir / 'onlyIMU_trajectory.png'}"
          + (f"  (+ {len(frames)} 张逐帧快照)" if save_each_frame else ""))

def visualize_pointcloud(raw_pcl, undist_pcl=None, frame=None, window_name=None,
                         point_size=None, bg_color=None):
    """
    用 Open3D 交互窗口查看原始点云(浅蓝)与去畸变后的点云(红)。不保存任何文件。

    参数:
        raw_pcl     : 原始点云 (结构化数组, 字段 x, y, z, ...)
        undist_pcl  : 去畸变后的点云; 传 None 就只显示原始点云
        frame       : 帧号, 只用于窗口标题
        window_name : 窗口标题; None 时自动生成
        point_size  : 点大小; None 用 Open3D 默认(窗口里也能改)
        bg_color    : 背景色 (3,) 取值 0~1; None 用默认

    说明:
        窗口是阻塞的: 鼠标左键拖动=旋转, 滚轮=缩放, 中键拖动=平移,
        关掉窗口后 main.py 才会继续往下执行。
    """
    if raw_pcl is None or len(raw_pcl) == 0:
        print("[visualize] 点云为空, 跳过")
        return

    def _to_pcd(pcl, color):
        xyz = np.column_stack([pcl['x'], pcl['y'], pcl['z']]).astype(np.float64)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.paint_uniform_color(color)
        return pcd, xyz

    geometries = []
    # 先加原始点云(浅蓝), 再加去畸变点云(红, 会画在上层)
    pcd_raw, raw_xyz = _to_pcd(raw_pcl, [0.5294, 0.8078, 0.9804])   # lightskyblue
    geometries.append(pcd_raw)

    if undist_pcl is not None and len(undist_pcl) > 0:
        pcd_und, und_xyz = _to_pcd(undist_pcl, [1.0, 0.0, 0.0])     # red
        geometries.append(pcd_und)
        if len(undist_pcl) == len(raw_pcl):
            shift = np.linalg.norm(und_xyz - raw_xyz, axis=1) * 1000.0
            print(f"[visualize] 去畸变位移: 平均 {shift.mean():.3f} mm / 最大 {shift.max():.3f} mm")

    if window_name is None:
        window_name = 'raw (light blue) vs undistorted (red)'
        if frame is not None:
            window_name += f' - frame {frame}'

    if point_size is None and bg_color is None:
        # 默认简易模式
        o3d.visualization.draw_geometries(geometries,
                                          window_name=window_name,
                                          width=1280, height=800)
    else:
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=window_name, width=1280, height=800)
        for geom in geometries:
            vis.add_geometry(geom)

        # 获取渲染控制器
        opt = vis.get_render_option()

        # 1. 消除小方块感：将点尺寸缩小为细腻的像素点 (推荐 1.0 或 2.0)
        if point_size is not None:
            opt.point_size = float(point_size)
        else:
            opt.point_size = 1.0  # 默认 1.0 像素，点会最细锐

        # 2. 设置背景色 (例如 [0, 0, 0] 纯黑更能突出细点)
        if bg_color is not None:
            opt.background_color = np.asarray(bg_color, dtype=np.float64)

        vis.run()
        vis.destroy_window()
class RealtimeVisualizer:
    """
    非阻塞实时可视化: 一个 Open3D 窗口里同时显示
        全局地图(浅蓝) / 当前帧世界系点云(红) / IMU 位姿轨迹(绿) / 当前位姿坐标轴

    关键点(踩过的坑):
      1) 只调用 poll_events()/update_renderer(), 绝不调用 run(), 所以不阻塞主循环;
      2) **每帧新建几何体, 用 remove + add 替换**, 不用 update_geometry ——
         legacy Visualizer 对"点数变化"的缓冲重传不可靠, 会出现窗口全黑;
      3) 只有在"第一次加非空地图"时才 reset_bounding_box=True(把相机对准数据),
         之后都是 False, 视角不会每帧乱跳。
    """

    def __init__(self, window_name="FastLIO2 realtime", enable=True,
                 map_point_size=1.5, scan_point_size=3.0, show_axes=True, max_traj=20000,
                 map_color=(0.53, 0.81, 0.98), scan_color=(1.0, 0.0, 0.0),
                 traj_color=(0.0, 1.0, 0.0), bg_color=(0.05, 0.05, 0.05),
                 width=1280, height=800, verbose=True):
        self.window_name = window_name
        self.enable = bool(enable)
        self.map_point_size = float(map_point_size)
        self.scan_point_size = float(scan_point_size)
        self.show_axes = bool(show_axes)
        self.max_traj = int(max_traj)
        self.map_color = list(map_color); self.scan_color = list(scan_color)
        self.traj_color = list(traj_color); self.bg_color = list(bg_color)
        self.width = int(width); self.height = int(height)
        self.verbose = bool(verbose)

        self._vis = None
        self.enabled = False
        self._open_failed = not self.enable
        self._view_fitted = False
        self._map_pcd = None; self._scan_pcd = None
        self._axes = None;    self._traj = None
        self.pos_history = []
        self.frame_count = 0

    # ---------------- 内部 ----------------
    def _ensure_window(self):
        if self._vis is not None:
            return True
        if self._open_failed or not self.enable:
            return False
        try:
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name=self.window_name, width=self.width, height=self.height)
            opt = vis.get_render_option()
            opt.point_size = self.map_point_size
            opt.background_color = np.asarray(self.bg_color, dtype=np.float64)
            self._vis = vis
            self.enabled = True
            if self.verbose:
                print("[viz] 窗口已创建")
            return True
        except Exception as e:
            print(f"[viz] 无法创建窗口({e}); 已关闭实时显示, 主流程继续")
            self._open_failed = True
            self._vis = None
            self.enabled = False
            return False

    def _swap(self, old, new, fit=False):
        """用新几何体替换旧的 (remove + add), 避免点数变化导致的缓冲问题"""
        vis = self._vis
        if old is not None:
            vis.remove_geometry(old, reset_bounding_box=False)
        vis.add_geometry(new, reset_bounding_box=fit)
        return new

    # ---------------- 每帧调用 ----------------
    def update(self, map_pts=None, scan_pts=None, pos=None, rot=None):
        self.frame_count += 1
        if pos is not None:
            self.pos_history.append(np.asarray(pos, dtype=np.float64).copy())

        map_arr = None if map_pts is None else np.asarray(map_pts, dtype=np.float64)
        scan_arr = None if scan_pts is None else np.asarray(scan_pts, dtype=np.float64)

        if not self._ensure_window():
            return
        vis = self._vis
        if not self.enabled:
            return

        # ---- 全局地图 ----
        if map_arr is not None and len(map_arr):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(map_arr)
            pcd.paint_uniform_color(self.map_color)
            fit = not self._view_fitted                 # 首帧把相机对准地图
            self._map_pcd = self._swap(self._map_pcd, pcd, fit=fit)
            if fit:
                self._view_fitted = True
                if self.verbose:
                    print(f"[viz] 首帧渲染: 地图 {len(map_arr)} 点, "
                          f"包围盒 min={np.round(map_arr.min(axis=0), 2)} "
                          f"max={np.round(map_arr.max(axis=0), 2)}")

        # ---- 当前帧世界系点云 ----
        if scan_arr is not None and len(scan_arr):
            spcd = o3d.geometry.PointCloud()
            spcd.points = o3d.utility.Vector3dVector(scan_arr)
            spcd.paint_uniform_color(self.scan_color)
            self._scan_pcd = self._swap(self._scan_pcd, spcd)

        # ---- IMU 轨迹曲线 ----
        pts = np.asarray(self.pos_history[-self.max_traj:], dtype=np.float64)
        if len(pts) >= 2:
            lines = np.column_stack([np.arange(len(pts) - 1), np.arange(1, len(pts))]).astype(np.int32)
            ls = o3d.geometry.LineSet()
            ls.points = o3d.utility.Vector3dVector(pts)
            ls.lines = o3d.utility.Vector2iVector(lines)
            ls.colors = o3d.utility.Vector3dVector(np.tile(self.traj_color, (len(lines), 1)))
            self._traj = self._swap(self._traj, ls)

        # ---- 当前位姿坐标轴 ----
        if self.show_axes and pos is not None and rot is not None:
            ax = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
            T = np.eye(4)
            T[:3, :3] = np.asarray(rot, dtype=np.float64)
            T[:3, 3] = np.asarray(pos, dtype=np.float64)
            ax.transform(T)
            self._axes = self._swap(self._axes, ax)

        # ---- 非阻塞刷新 ----
        self.enabled = bool(vis.poll_events())
        vis.update_renderer()
        if not self.enabled and self.verbose:
            print("[viz] 窗口已关闭, 后续帧不再刷新(主流程继续)")

    def reset_view(self):
        """把相机重新对准数据 (视角跑飞了可以调它)"""
        if self._vis is not None and self.enabled:
            self._vis.reset_view_point(True)
            self._view_fitted = True

    def close(self):
        if self._vis is not None and self.enabled:
            try:
                self._vis.destroy_window()
            except Exception:
                pass
        self.enabled = False
        self._vis = None
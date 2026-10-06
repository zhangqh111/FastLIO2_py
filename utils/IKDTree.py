import numpy as np

def squared_distance(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return dx * dx + dy * dy + dz * dz

def fit_plane_residual(point, neighbors, plane_threshold=0.1,
                       residual_ratio=1.0 / 9.0, range_ref=None):
    
    pts = np.asarray([nb[1] for nb in neighbors], dtype=np.float64)
    if len(pts) < 3:                       # 至少 3 个点才能定一个平面(也才有 1 个自由度判断平面性)
        return None
    p = np.asarray(point, dtype=np.float64)

    # ---- 1. 最小二乘拟合平面: 质心 + 协方差最小特征向量 = 法向 ----
    centroid = pts.mean(axis=0)
    q = pts - centroid                     # 相对质心, 平面必过质心
    cov = q.T @ q / len(pts)               # 3x3
    eig_val, eig_vec = np.linalg.eigh(cov) # 特征值升序
    normal = eig_vec[:, 0]                 # 最小特征值对应的方向
    normal = normal / np.linalg.norm(normal)
    d = -float(normal @ centroid)          # 平面: normal·x + d = 0

    # ---- 2. 所有近邻点到平面的距离都要在阈值内 ----
    dist_nb = np.abs(q @ normal)           # = |normal·p_i + d|, 用相对质心的坐标更省一次加法
    if dist_nb.max() > plane_threshold:
        return None

    # ---- 3. 当前点的点面残差 + 相对门限 ----
    residual = float(normal @ p + d)
    ref = float(np.linalg.norm(p)) if range_ref is None else float(range_ref)
    if abs(residual) >= residual_ratio * ref:
        return None

    return p, residual, normal, d

class MaxHeap:
    def __init__(self, key=None, capacity=None):
        self._data = []                          # 内部数组, 元素为 (key, item) 二元组
        self._key = key if key is not None else (lambda x: x)
        self.capacity = capacity                 # None 表示不限容量

    def __len__(self):
        return len(self._data)
    
    def is_empty(self):
        return len(self._data) == 0

    def top(self):        
        return self._data[0][1] if self._data else None
    
    def top_key(self):        
        return self._data[0][0] if self._data else None
    
    def push(self, item):        
        self._data.append((self._key(item), item))
        self._sift_up(len(self._data) - 1)
        if self.capacity is not None and len(self._data) > self.capacity:
            return self.pop()
        return None
    
    def pop(self):        
        data = self._data
        if not data:
            return None
        top_item = data[0][1]
        last = data.pop()
        if data:
            data[0] = last
            self._sift_down(0)
        return top_item
    
    def push_k(self, item, k):        
        if k <= 0:
            return None
        if len(self._data) < k:
            self.push(item)
            return None
        item_key = self._key(item)
        if item_key < self._data[0][0]:
            evicted = self._data[0][1]
            self._data[0] = (item_key, item)
            self._sift_down(0)
            return evicted
        return None
    
    def items(self):        
        return [item for _, item in self._data]

    def sorted_items(self):        
        return [item for _, item in sorted(self._data, key=lambda kv: kv[0])]

    def __repr__(self):
        return f"MaxHeap(size={len(self._data)}, top_key={self.top_key()})"
    
    def _sift_up(self, i):
        data = self._data
        while i > 0:
            parent = (i - 1) // 2
            if data[i][0] > data[parent][0]:
                data[i], data[parent] = data[parent], data[i]
                i = parent
            else:
                break

    def _sift_down(self, i):
        data = self._data
        n = len(data)
        while True:
            left = 2 * i + 1
            right = left + 1
            largest = i
            if left < n and data[left][0] > data[largest][0]:
                largest = left
            if right < n and data[right][0] > data[largest][0]:
                largest = right
            if largest == i:
                break
            data[i], data[largest] = data[largest], data[i]
            i = largest

class Node:
    __slots__ = ('point', 'axis', 'value', 'left', 'right',
                 'subtree_size', 'invalid_num', 'bbox',
                 'lazy_deleted', 'lazy_size')
    
    def __init__(self, point, axis=0, value=0.0):
        self.point = point              # (3,) float64: 本节点存的那个点
        self.axis = axis                # 分割轴 0/1/2
        self.value = value              # 分割值: 左子树该轴坐标 <= value, 右子树 > value
        self.left = None                # 左子树
        self.right = None               # 右子树
        self.subtree_size = 1           # 子树点数 (含被惰性删除的)
        self.invalid_num = 0            # 子树中被惰性删除的点数 (含自身)
        self.bbox = np.vstack((point, point)).astype(np.float64)   # (2,3) = [min; max], 搜索剪枝用        
        self.lazy_deleted = False       # 本节点是否被惰性标记删除
        self.lazy_size = 0.0            # 标记时用的体素尺寸 (供 push_down 用)

    
class IKDTree:
    def __init__(self):        
        self.root = None
        self._flat = None
        self.depth_debug = 5
    
    def build_tree(self, node, points, left, right, d):
        self._flat = None
        if left > right:
            return -1
        if right >= len(points):
            raise IndexError(f"right 超出点云范围: right={right}, len(points)={len(points)}")
        
        seg = points[left:right + 1]
        span = seg.max(axis=0) - seg.min(axis=0)
        axis = int(np.argmax(span)) 
        mid = (left + right) // 2
        kth = mid - left                               
        order = np.argpartition(seg[:, axis], kth)     
        seg[:] = seg[order]
        # node.point = seg[kth].astype(np.float64).copy()    
        node.point = tuple(seg[kth].tolist())     # Python 元组: 距离计算快 2 倍
        node.axis = axis
        node.value = float(seg[kth, axis])
        node.left = None                               
        node.right = None                
        # if d < self.depth_debug : print('\t' * d + f"轴 {axis}, 当前点 {node.point}, 层数 {d}, left {left}, mid {mid}")
        # if d < self.depth_debug : print('\t' * d + f"当前点 {node.point}, 层数 {d}")

        if left <= mid - 1:
            node.left = Node(np.zeros(3))
            # if d < self.depth_debug : print('\t' * d + f"开始划分左子树")
            self.build_tree(node.left, points, left, mid - 1, d+1)
        if mid + 1 <= right:
            node.right = Node(np.zeros(3))
            # if d < self.depth_debug : print('\t' * d + f"开始划分右子树")
            self.build_tree(node.right, points, mid + 1, right, d+1)

        return mid
        
    def _collect_points(self):
        """把树里所有点收集成 (M,3) 数组并缓存 (建图/插入/删除后必须置 self._flat = None)"""
        if self._flat is None:
            pts = []
            stack = [self.root]
            while stack:
                nd = stack.pop()
                if nd is None:
                    continue
                pts.append(nd.point)
                stack.append(nd.left)
                stack.append(nd.right)
            self._flat = np.asarray(pts, dtype=np.float64) if pts else np.zeros((0, 3))
        return self._flat

    def SearchCloudVec(self, points, k_nearst=5, max_dist_sq=5.0, chunk=512):
        n = len(points)
        valid = np.zeros(n, dtype=bool)
        neighbors = [None] * n
        valid_num = 0

        M = self._collect_points()
        if len(M) == 0:
            return valid, valid_num, neighbors

        k = int(k_nearst)
        m2 = np.einsum('ij,ij->i', M, M)                     # |m|^2
        Q = np.asarray(points, dtype=np.float64)

        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            q = Q[s:e]
            q2 = np.einsum('ij,ij->i', q, q)
            # |q-m|^2 = |q|^2 + |m|^2 - 2 q·m   (矩阵乘法走 BLAS, 比广播快且省内存)
            d2 = q2[:, None] + m2[None, :] - 2.0 * (q @ M.T)
            np.maximum(d2, 0.0, out=d2)                      # 抹掉浮点负零

            idx = np.argpartition(d2, k - 1, axis=1)[:, :k]  # 每行最小的 k 个(无序)
            sub = np.take_along_axis(d2, idx, axis=1)
            order = np.argsort(sub, axis=1)                  # 行内按距离升序
            idx_s = np.take_along_axis(idx, order, axis=1)
            sub_s = np.take_along_axis(sub, order, axis=1)

            ok = sub_s[:, -1] <= max_dist_sq                 # 第 k 近邻的判据
            valid[s:e] = ok
            valid_num += int(ok.sum())

            for r in range(e - s):                           # 仍组装成 MaxHeap, 接口不变
                heap = MaxHeap(key=lambda kv: kv[0])
                for j in range(k):
                    heap.push((float(sub_s[r, j]), M[idx_s[r, j]]))
                neighbors[s + r] = heap

        return valid, valid_num, neighbors    

    @staticmethod
    def _voxel_downsample(points, voxel_size):
        """体素降采样: 每个体素取质心 (内置实现, 用于不想引入 open3d 的场合)"""
        keys = np.floor(points / voxel_size).astype(np.int64)
        uniq, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
        sums = np.zeros((len(uniq), 3), dtype=np.float64)
        np.add.at(sums, inverse, points)
        return sums / counts[:, None]

    def add_map_points(self, points, voxel_size=0.5, voxel_fn=None):
        """
        把当前帧点云并入全局地图: 与现有地图点合并 -> 整体体素降采样 -> 重建 ikd-tree。

        典型用法(迭代 EKF 更新之后):
            R_wl = state.rot @ state.R_LI; t_wl = state.pos + state.rot @ state.t_LI
            pts_map = pts_down @ R_wl.T + t_wl          # 用*更新后*的位姿投影到世界系
            n = ikd_tree.add_map_points(pts_map, 0.5, voxel_fn=voxel_filter)

        参数:
            points     : (N,3) 世界系下的点
            voxel_size : 体素边长(米); None 表示不降采样
            voxel_fn   : 体素降采样函数 fn(points, voxel_size)->points;
                         None 时用内置 numpy 实现。**建议传 mapper.voxel_filter**,
                         这样地图和查询点云用的是同一套体素网格约定
        返回:
            合并降采样后的地图点数
        """
        new_pts = np.asarray(points, dtype=np.float64)
        old_pts = self._collect_points()                     # 现有地图点(有缓存, 热调用不重新遍历)
        all_pts = np.vstack([old_pts, new_pts]) if len(old_pts) else new_pts

        if voxel_size is not None and len(all_pts):
            fn = voxel_fn if voxel_fn is not None else IKDTree._voxel_downsample
            all_pts = fn(all_pts, float(voxel_size))

        # ---- 重建树 (顺带让 _flat 缓存失效) ----
        self.root = None
        self._flat = None
        if len(all_pts) == 0:
            return 0
        data = all_pts.copy()                                # build_tree 会原地重排
        self.root = Node(np.zeros(3))
        self.build_tree(self.root, data, 0, len(data) - 1, 0)
        return len(data)

if __name__ == '__main__':    
    pass
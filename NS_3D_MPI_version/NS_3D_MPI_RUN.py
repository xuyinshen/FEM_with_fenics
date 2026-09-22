#!/usr/bin/env python
# coding: utf-8

# # <center>有限元分析——从二维扩散方程到NS方程</center>

# # 1. 引入必要的库

# In[1]:


import numpy as np
from dolfin import *
from meshio import write
import meshio
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import gmsh
import sys, time, os, datetime, re, csv
import traceback
from PIL import Image
from scipy.sparse.linalg import eigs
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import inv
from petsc4py import PETSc
import dolfin.cpp as cpp
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False
import yaml
import argparse

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

sys.stdout.flush()
sys.stderr.flush()

set_log_level(LogLevel.ERROR)

def mpi_exception_handler(exc_type, exc_value, exc_traceback):
    """捕获所有未处理异常，广播给所有进程"""
    error_msg = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    
    # 所有进程打印错误
    print(f"\n❌ [Rank {rank}] 发生错误:\n{error_msg}", file=sys.stderr)
    sys.stderr.flush()
    
    # 通知其他进程退出
    comm.Abort(1)
    sys.exit(1)

# 安装异常处理器
sys.excepthook = mpi_exception_handler

# parameters["form_compiler"]["optimize"] = True
# parameters["form_compiler"]["cpp_optimize"] = True
# parameters["form_compiler"]["representation"] = "uflacs"


# In[2]:


class ProgressBar:
    def __init__(self, total, prefix="", width=100, min_show_eta_percent=0.001, 
                 refresh_interval=0.2, comm=None, window_size=3):
        """
        通用型进度条（支持 MPI / 多线程 / 纯循环）
        :param total: 总步数/总任务数
        :param prefix: 进度条前缀
        :param width: 进度条字符宽度
        :param min_show_eta_percent: 进度达到多少比例后开始显示 ETA（防初期抖动）
        :param refresh_interval: 终端刷新间隔（秒），防 I/O 阻塞
        :param comm: MPI 通信器（传 None 则为单进程模式）
        :param window_size: 滑动窗口大小（默认10步），用于计算近期平均速度
        """
        self.comm = comm
        self.rank = comm.Get_rank() if comm else 0
        self.total = total
        self.prefix = prefix
        self.width = width
        self.min_show_eta = min_show_eta_percent
        self.refresh_interval = refresh_interval
        self.window_size = window_size

        self.start_time = time.time()
        self.last_refresh_time = 0.0
        self.history = []  # 存储最近 window_size 个 (step, timestamp)

    def update(self, num):
        # MPI 隔离：非主进程直接返回，零开销
        if self.rank != 0:
            return

        current_time = time.time()
        self._record_history(num, current_time)  # 始终记录，保证速度计算准确

        rate = float(num) / self.total if self.total > 0 else 0

        # 节流：未到刷新间隔且未完成时跳过终端 I/O
        if num < self.total and (current_time - self.last_refresh_time < self.refresh_interval):
            return
        self.last_refresh_time = current_time

        # 计算 ETA
        eta_str = ""
        elapsed = current_time - self.start_time
        if num > 0 and rate >= self.min_show_eta and elapsed > 0.3:
            speed = self._get_recent_speed()
            if speed > 0:
                remaining = self.total - num
                eta_sec = int(remaining / speed)
                eta_str = self._format_eta(eta_sec)

        # 绘制进度条
        filled = int(self.width * rate)
        bar = '[{}{}]'.format('*' * filled, ' ' * (self.width - filled))
        sys.stdout.write('\r{}{} {:.1f}% {}'.format(self.prefix, bar, rate * 100, eta_str))
        sys.stdout.flush()

        if num >= self.total:
            sys.stdout.write('\n')
            sys.stdout.flush()

    def _record_history(self, num, t):
        """维护滑动窗口历史"""
        self.history.append((num, t))
        if len(self.history) > self.window_size:
            self.history.pop(0)

    def _get_recent_speed(self):
        """基于滑动窗口计算近期平均速度（步/秒）"""
        if len(self.history) < 2:
            # 数据不足时回退到累计平均（仅前1~2步生效）
            dt = time.time() - self.start_time
            return self.history[-1][0] / dt if dt > 1e-6 else 0.0

        oldest_num, oldest_t = self.history[0]
        latest_num, latest_t = self.history[-1]

        dt = latest_t - oldest_t
        dn = latest_num - oldest_num

        if dt <= 1e-6 or dn <= 0:
            return 0.0
        return dn / dt

    @staticmethod
    def _format_eta(seconds):
        if seconds < 60:
            return f"预计还需 {seconds}秒"
        elif seconds < 3600:
            m, s = divmod(seconds, 60)
            return f"预计还需 {m}分{s}秒"
        else:
            h, rem = divmod(seconds, 3600)
            m, s = divmod(rem, 60)
            return f"预计还需 {h}小时{m}分"

def Format_eta(seconds):
        if seconds < 60:
            return f"预计还需 {round(seconds, 3)}秒"
        elif seconds < 3600:
            m, s = divmod(seconds, 60)
            return f"预计还需 {round(m)}分{round(s, 3)}秒"
        else:
            h, rem = divmod(seconds, 3600)
            m, s = divmod(rem, 60)
            return f"预计还需 {round(h)}小时{round(m)}分"

##============加载参数配置=============

if rank == 0:
    print("正在加载参数...\n")

def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', help='YAML 配置文件路径')
    args, _ = parser.parse_known_args()  # 忽略 mpiexec 注入的额外参数

    # 默认配置兜底
    default = {
        'geometry': {'D':0.1, 'L_factor':21.0, 'H_factor':14.0, 'W_factor':4.0, 'xc_factor':5.0, 'yc_factor':7.0, 'rc_factor':0.5},
        'mesh': {'coarse':0.05, 'fine':0.02, 'R_inner':0.3, 'R_outer':0.6},
        'physics': {'nu':1e-5, 'v0':1.0, 'T':0.1, 'dt_auto':True, 'dt_value':None, 'theta':0.5},
        'solver': {'max_nonlinear_iter':5, 'tol_nonlinear':1e-5, 'solver_u':'bicgstab', 'precond_u':'default', 'solver_p':'bicgstab', 'precond_p':'default'},
        'output': {'base_dir':'/mnt/d/paraview_data', 'parent_folder':'Data_NS_3D', 'save_freq':5}
    }

    if os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            user_cfg = yaml.safe_load(f)
        for sec, vals in user_cfg.items():
            if sec in default: default[sec].update(vals)
            else: default[sec] = vals
    return default

CONFIG = load_config()

# 解包为全局变量（替换原代码中的硬编码）
D = CONFIG['geometry']['D']
L = CONFIG['geometry']['L_factor'] * D
H = CONFIG['geometry']['H_factor'] * D
W = CONFIG['geometry']['W_factor'] * D
xc = CONFIG['geometry']['xc_factor'] * D
yc = H/2 # CONFIG['geometry']['yc_factor'] * D
rc = CONFIG['geometry']['rc_factor'] * D

mesh_size = CONFIG['mesh']['coarse']
mesh_size_fine = CONFIG['mesh']['fine']
R_inner = CONFIG['mesh']['R_inner']
R_outer = CONFIG['mesh']['R_outer']

nu = CONFIG['physics']['nu']
v0 = CONFIG['physics']['v0']
T = 300 * mesh_size_fine # CONFIG['physics']['T']
theta = CONFIG['physics']['theta']

num_nnlin_iter = CONFIG['solver']['max_nonlinear_iter']
tol_nl = CONFIG['solver']['tol_nonlinear']
solver_u, prec_u = CONFIG['solver']['solver_u'], CONFIG['solver']['precond_u']
solver_p, prec_p = CONFIG['solver']['solver_p'], CONFIG['solver']['precond_p']
save_freq = CONFIG['output']['save_freq']
base_dir = CONFIG['output']['base_dir']
parent_folder = CONFIG['output']['parent_folder']

if rank == 0:
    print(f"📌 加载参数完成 | nu={nu}, v0={v0}, T={T}, save_freq={save_freq}")




# # 2. 构建网格

# In[3]:

mesh_size_coarse = mesh_size

if rank == 0:
    gmsh.initialize()
    gmsh.model.add("cylinder_3d")

# Create 3D box
    box = gmsh.model.occ.addBox(0, 0, 0, L, H, W)

# Create cylinder (along Z-axis)
    cylinder = gmsh.model.occ.addCylinder(xc, yc, 0, 0, 0, W, rc)

# Cut cylinder from box
    gmsh.model.occ.cut([(3, box)], [(3, cylinder)])
    gmsh.model.occ.synchronize()

    gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)

# --- Set to use first-order (linear) elements ---
    gmsh.option.setNumber("Mesh.ElementOrder", 1)  # 关键：避免生成10节点高阶四面体

    surfaces = gmsh.model.getEntities(dim=2)
    boundary_surfaces = {}

    tol = 1e-6

    for dim, tag in surfaces:
        bbox = gmsh.model.getBoundingBox(dim, tag)
        xmin, ymin, zmin, xmax, ymax, zmax = bbox

        if abs(xmin) < tol and abs(xmax) < tol:
            name = "left"
        elif abs(xmin - L) < tol and abs(xmax - L) < tol:
            name = "right"
        elif abs(ymin) < tol and abs(ymax) < tol:
            name = "bottom"
        elif abs(ymin - H) < tol and abs(ymax - H) < tol:
            name = "top"
        elif abs(zmin) < tol and abs(zmax) < tol:
            name = "front"
        elif abs(zmin - W) < tol and abs(zmax - W) < tol:
            name = "back"
        else:
            name = "cylinder"  # remaining is cylinder wall

        boundary_surfaces[name] = tag

# Add physical groups for boundaries
    boundnames = {"left": 1, "right": 2, "bottom": 3, "top": 4, "front": 5, "back": 6, "cylinder": 7}
    for name, tag in boundary_surfaces.items():
        physical_tag = boundnames[name]
        gmsh.model.addPhysicalGroup(2, [tag], physical_tag)
        gmsh.model.setPhysicalName(2, physical_tag, name)

# Add physical volume
    volume_entities = gmsh.model.getEntities(dim=3)
    gmsh.model.addPhysicalGroup(3, [ve[1] for ve in volume_entities], tag=10)
    gmsh.model.setPhysicalName(3, 10, "fluid")

# Distance field from cylinder surface
    distance_field = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(distance_field, "FacesList", [boundary_surfaces["cylinder"]])

    threshold_field = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(threshold_field, "IField", distance_field)
    gmsh.model.mesh.field.setNumber(threshold_field, "LcMin", mesh_size_fine)
    gmsh.model.mesh.field.setNumber(threshold_field, "LcMax", mesh_size_coarse)
    gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", R_inner)
    gmsh.model.mesh.field.setNumber(threshold_field, "DistMax", R_outer)

    gmsh.model.mesh.field.setAsBackgroundMesh(threshold_field)

# Generate 3D mesh
    gmsh.model.mesh.generate(3)

    gmsh.write("cylinder_3d.msh")
    

    gmsh.finalize()



# In[4]:


if rank == 0:
    print("🔄 正在读取 Gmsh 网格并转换为 XDMF...")
    msh = meshio.read("cylinder_3d.msh")
    
    # ==========================================
    # 1. 处理体积网格 (Tetrahedra)
    # ==========================================
    tetra_cells = []
    tetra_data = []
    
    for i, cell in enumerate(msh.cells):
        if cell.type == "tetra":
            tetra_cells.append(cell.data)
            # ✅ 修正：这里应该是 msh.cell_data
            if "gmsh:physical" in msh.cell_data:
                tetra_data.append(msh.cell_data["gmsh:physical"][i])
    
    if tetra_cells:
        tetra_cells_merged = np.vstack(tetra_cells)
        tetra_data_merged = np.concatenate(tetra_data) if tetra_data else None
        
        meshio.write("mesh.xdmf", meshio.Mesh(
            points=msh.points,
            cells={"tetra": tetra_cells_merged},
            cell_data={"gmsh:physical": [tetra_data_merged]} if tetra_data_merged is not None else {},
            field_data=msh.field_data
        ))
        print("✅ 体积网格 mesh.xdmf 生成成功")
    else:
        print("❌ 未找到四面体单元！")

    # ==========================================
    # 2. 处理边界网格 (Triangles)
    # ==========================================
    triangle_cells = []
    triangle_data = []
    
    for i, cell in enumerate(msh.cells):
        if cell.type == "triangle":
            triangle_cells.append(cell.data)
            # ✅ 修正：这里也应该是 msh.cell_data
            if "gmsh:physical" in msh.cell_data:
                triangle_data.append(msh.cell_data["gmsh:physical"][i])
    
    if triangle_cells:
        triangle_cells_merged = np.vstack(triangle_cells)
        triangle_data_merged = np.concatenate(triangle_data) if triangle_data else None
        
        meshio.write("mesh_boundary.xdmf", meshio.Mesh(
            points=msh.points,
            cells={"triangle": triangle_cells_merged},
            cell_data={"gmsh:physical": [triangle_data_merged]} if triangle_data_merged is not None else {},
            field_data=msh.field_data
        ))
        print("✅ 边界网格 mesh_boundary.xdmf 生成成功")
    else:
        print("⚠️ 未找到三角形单元")

# 所有进程等待 rank 0 写完文件


from dolfin import XDMFFile

print(f"[Rank {rank}] 开始加载网格...\n")

# 1. 加载体积网格
mesh = Mesh()
with XDMFFile("mesh.xdmf") as infile:
    infile.read(mesh)

# 2. 加载边界标记 (MeshValueCollection)
# 注意：dim=2 表示这是表面网格 (三角形)
mvc = MeshValueCollection("size_t", mesh, 2)

try:
    with XDMFFile("mesh_boundary.xdmf") as infile:
        infile.read(mvc, "gmsh:physical")
    
    # 转换为 MeshFunction (兼容不同版本的 FEniCS)
    try:
        boundaries = cpp.mesh.MeshFunctionSizet(mesh, mvc)
    except:
        boundaries = MeshFunction("size_t", mesh, mvc)
        
    if rank == 0: print("✅ 边界标记加载成功")
    
except Exception as e:
    if rank == 0: 
        print(f"⚠️ 边界文件加载失败: {e}")
        print("⚠️ 将尝试使用自动标记或全 0 标记")
    boundaries = MeshFunction("size_t", mesh, 2, 0)

# 定义测度
ds = Measure('ds', domain=mesh, subdomain_data=boundaries)



if rank == 0:
    print("\n" + "="*40)
    print(f"🎉 网格加载完成！")
    print(f"顶点数: {mesh.num_vertices()}")
    print(f"单元数: {mesh.num_cells()}")
    print("="*40 + "\n")

# # 3. 定义函数空间

# In[6]:


# 3D 向量空间（速度有 u,v,w 三个分量）
V = VectorFunctionSpace(mesh, "P", 2)  # P2-P1 Taylor-Hood
Q = FunctionSpace(mesh, "P", 1)

if rank == 0:
    print("函数空间定义完成！\n")


# # 4. 设置边界条件

# In[7]:


# 物理组编号（与 Gmsh 一致）
class LeftBound(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0.0) and on_boundary

class RightBound(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], L) and on_boundary

class BottomBound(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], 0.0) and on_boundary

class TopBound(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], H) and on_boundary

class FrontBound(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[2], 0.0) and on_boundary

class BackBound(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[2], W) and on_boundary

class CylinderBound(SubDomain):
    def inside(self, x, on_boundary):
        dist = sqrt((x[0] - xc)**2 + (x[1] - yc)**2)
        return on_boundary and near(dist, rc, 1e-2)

# 实例化
left = LeftBound()
right = RightBound()
top = TopBound()
bottom = BottomBound()
front = FrontBound()
back = BackBound()
cylinder = CylinderBound()

# 创建边界标记函数
boundaries = MeshFunction("size_t", mesh, 2)  # dim=2 表面
boundaries.set_all(0)

left.mark(boundaries, 1)
right.mark(boundaries, 2)
top.mark(boundaries, 3)
bottom.mark(boundaries, 4)
front.mark(boundaries, 5)
back.mark(boundaries, 6)
cylinder.mark(boundaries, 7)

# 定义积分测度
ds = Measure('ds', domain=mesh, subdomain_data=boundaries)
dS = Measure('dS', domain=mesh)  # 内部面（可选）

dx = dx(metadata={'quadrature_degree': 4})
ds = ds(metadata={'quadrature_degree': 4})

if rank == 0:
    print("边界条件设置完成！\n")


# # 5. 定义实验空间和测试空间

# In[8]:


# 测试/试探函数
u = TrialFunction(V)
v = TestFunction(V)
p = TrialFunction(Q)
q = TestFunction(Q)

# 函数
u_n = Function(V)   # 上一时间步速度
p_n = Function(Q)   # 上一时间步压力
u1 = Function(V)    # 当前速度
p1 = Function(Q)    # 当前压力


# # 6. 定义弱方程

# In[9]:


# 参数
# v0 = 1.0      # 入口速度
# nu = 1e-5     # 运动粘度
# dt = 0.5 * mesh.hmin()  # 时间步长

if CONFIG['physics']['dt_auto']:
    dt = 0.5 * mesh_size_fine
else:
    dt = CONFIG['physics']['dt_value']

num_steps = int(T / dt)

# 入口速度 (x方向)
uin = Expression(('v0', '0.0', '0.0'), v0=v0, degree=2)

# 初始场
u0 = interpolate(uin, V)
p0 = interpolate(Constant(0.0), Q)

u_n.assign(u0)
p_n.assign(p0)

# 边界条件
bcu = [
    DirichletBC(V, uin, boundaries, 1),      # left, 入口
#    DirichletBC(V, Constant((0, 0, 0)), boundaries, 2),  # right
    DirichletBC(V, Constant((0, 0, 0)), boundaries, 3),  # top
    DirichletBC(V, Constant((0, 0, 0)), boundaries, 4),  # bottom
    DirichletBC(V, Constant((0, 0, 0)), boundaries, 5),  # front
    DirichletBC(V, Constant((0, 0, 0)), boundaries, 6),  # back
    DirichletBC(V, Constant((0, 0, 0)), boundaries, 7),  # cylinder
]

bcp = [
    DirichletBC(Q, Constant(0.0), boundaries, 2),  # outlet 压力为 0
]

bcs = [bcu, bcp]


# In[10]:


# 稳定化参数
h = CellDiameter(mesh)
u_mag = sqrt(dot(u1, u1))
d1 = 1.0 / sqrt(1.0/dt**2 + (u_mag/h)**2 + 1e-10)  # 防止除零
d2 = h * u_mag

# 中间速度
u_mid = theta*u + (1 - theta)*u_n
u_mid1 = theta*u1 + (1 - theta)*u_n

# SUPG 项
F_SUPG = d1 * inner(
    (u - u_n)/dt + dot(grad(u_mid), u_mid1) + grad(p_n) - nu*div(grad(u_mid)),
    dot(grad(v), u_mid1)
) * dx \
         + d2 * div(u_mid) * div(v) * dx

# PSPG 项
F_PSPG = d1 * inner(div(u_mid1), q) * dx

# 动量方程（预测步）
Fu = (
    (inner((u - u_n)/dt, v)
    + inner(dot(grad(u_mid), u_mid1), v)
    - p_n*div(v)
    + nu*inner(grad(u_mid), grad(v)))*dx
    + F_SUPG
)

# 压力方程（校正步）
Fp = (
    (inner(grad(p), grad(q))
    - inner(grad(p_n), grad(q)))*dx + F_PSPG
)

# 提取双线性形式
auv1 = lhs(Fu)
Lv1 = rhs(Fu)
auv2 = lhs(Fp)
Lv2 = rhs(Fp)

if rank == 0:
    print("弱方程定义完成！\n")


# # 7. 计算

# 对于扩散方程：
# 
# $$
# \frac{\partial {u}}{\partial t} - \nu\nabla^2 u = 0
# $$
# 
# 扩散速度为 $x \sim \sqrt{\nu T}$。因此需要更大的 $\nu$ 来看看传播效果。

# In[12]:


   
num_steps = int(T/dt)
# num_steps = int(3)
# num_pic = int(70)
# freq_pic = int(num_steps/num_pic)

u_n.assign(u0)
p_n.assign(p0)
ulist = [u0]
plist = [p0]
dtlist = [dt]

u_old = Function(V)

num_nnlin_iter = 5
prec = "amg" if has_krylov_solver_preconditioner("amg") else "default"
tol_nl = 1e-5

# ============================================================================
pvd_dir = None
img_dir = None

if rank == 0:
    base_dir = os.path.join("/mnt/d/paraview_data")
    parent_dir = os.path.join(base_dir, "Data_NS_3D")
    os.makedirs(parent_dir, exist_ok=True)

    current_date = datetime.datetime.now().strftime("%Y%m%d")

# 构建固定前缀（不含日期和序号）
    base_prefix = f"nu_{nu}_v0_{v0}_dt_{dt}_T_{T:.3f}_hmesh_{mesh_size_fine:.3f}"

# 🌟 扫描已有文件夹，寻找该参数组合下的最大序号
    max_idx = 0
    pattern = re.compile(rf"^{re.escape(base_prefix)}_\d{{8}}_(\d+)$")
    if os.path.isdir(parent_dir):
        for item in os.listdir(parent_dir):
            match = pattern.match(item)
            if match:
                idx = int(match.group(1))
                if idx > max_idx:
                    max_idx = idx

    next_idx = max_idx + 1
    folder_name = f"{base_prefix}_{current_date}_{next_idx:06d}"
    full_path = os.path.join(parent_dir, folder_name)

# 创建子目录
    pvd_dir = os.path.join(full_path, "paraview")
    img_dir = os.path.join(full_path, "diagram")
    os.makedirs(pvd_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    print(f"📁 文件夹已创建：{os.path.abspath(full_path)}")

# 📊 生成/追加 CSV 运行日志
    log_file = os.path.join(parent_dir, "run_log.csv")
    log_headers = ["创建时间", "序号", "nu", "v0", "dt", "T", "hmesh", "文件夹名"]
    file_exists = os.path.isfile(log_file)

    with open(log_file, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=log_headers)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
#        "日期": current_date,
            "文件夹名": folder_name,
            "序号": next_idx,
            "创建时间": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "nu": nu, "v0": v0, "dt": dt, "T": T, "hmesh": mesh_size_fine
        })
if rank == 0:
    print("已创建数据存储所需文件夹！\n")

pvd_dir = comm.bcast(pvd_dir, root=0)
img_dir = comm.bcast(img_dir, root=0)

if rank == 0:
    print("已广播文件夹位置！\n")

xdmf_u = XDMFFile(comm, os.path.join(pvd_dir, "velocity.xdmf"))
xdmf_p = XDMFFile(comm, os.path.join(pvd_dir, "pressure.xdmf"))
xdmf_u.parameters["flush_output"] = True  # 立即刷盘，防崩溃丢数据
xdmf_p.parameters["flush_output"] = True

#pbar = ProgressBar(num_steps-1, prefix="求解中: ", comm=comm)
save_freq = 5
resttime = None
if rank == 0:
    start_t = []
    end_t = []
#pbar.update(0)
for j in range(num_steps):
    
    if rank == 0:
        start_t.append(time.time())
    for k in range(10):
        A1 = assemble(auv1)
        b1 = assemble(Lv1)
        
        [bc.apply(A1, b1) for bc in bcu]
        [bc.apply(u1.vector()) for bc in bcu]
        
        # solve(A1, u1.vector(), b1, "gmres", "ilu")
        solve(A1, u1.vector(), b1, solver_u, prec_u)
        # u1.rename("Velocity", "velocity field")
	
        A2 = assemble(auv2)
        b2 = assemble(Lv2)
        
        [bc.apply(A2, b2) for bc in bcp]
        [bc.apply(p1.vector()) for bc in bcp]
        
        # solve(A2, p1.vector(), b2, "gmres", "ilu")
        solve(A2, p1.vector(), b2, solver_p, prec_p)
            
        p_n.assign(p1)
    
        #if k > 0:
        #    residual = u1.vector() - u_old.vector()
        #    if residual.norm("l2") < tol_nl:
        #        break
        # u_old.assign(u1)
        # p1.rename("Pressure", "pressure field")
        # file_p << p1
        # file_u << u1
    
    u_n.assign(u1)
    p_n.assign(p1)
    
#    dt = CFL*hmesh/np.max(np.abs(u_n.vector().get_local()))
    
    if j % save_freq == 0:
        xdmf_u.write(u1, j * dt)
        xdmf_p.write(p1, j * dt)
    if rank == 0:
        end_t.append(time.time())
        if j < 3:
            resttime = (end_t[j] - start_t[0])/(j+1) * (num_steps - j - 1)
        elif j < 13:
            resttime = (end_t[j] - start_t[j-3])/(4) * (num_steps - j - 1)
        else:
            resttime = (end_t[j] - start_t[j-10])/(11) * (num_steps - j - 1)
        sys.stdout.write(f"{Format_eta(resttime)}\n")
        sys.stdout.flush()
        sys.stderr.flush()
    if j == num_steps - 1:
        break

#    pbar.update(j)
    # if j % freq_pic == 0:
    #     plotInnerU(ulist, j)
    #     plt.savefig('./nu_{}_v0_{}_dh_{}/u/u_{}.png'.format(nu, v0, hmesh, j))
    #     plt.close(1)

    #     plotU(plist, j)
    #     plt.savefig('./nu_{}_v0_{}_dh_{}/p/p_{}.png'.format(nu, v0, hmesh, j))
    #     plt.close(1)
    # process_bar(j+1,num_steps)

if rank == 0:
    sys.stdout.write("正在关闭流式存储...\n")
    sys.stdout.flush()
    sys.stderr.flush()

try:
    xdmf_u.close()
    xdmf_p.close()
except Exception as e:
    sys.stdout.write(f"[Rank {rank}] 关闭文件警告: {e}")
    sys.stdout.flush()
    sys.stderr.flush()


if rank == 0:
    sys.stdout.write("✅ 计算完成，数据已流式保存！\n")
    sys.stdout.flush()
    sys.stderr.flush()

    
sys.stdout.flush()
sys.stderr.flush()

comm.Abort(0)





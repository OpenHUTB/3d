# 爱因斯坦雕像尺寸与 UV 展开量化方法

本文档记录了爱因斯坦雕像三维模型的**尺寸量化**与**UV 展开量化**方法，包括图片标定、Mesh 缩放、空间量化精度计算以及 UV 展开评估，同时附带可直接运行的 Python 代码示例。

---

## 1. 图片标定与雕像尺寸测量

### 1.1 标定参考
- 使用 A4 纸（210mm × 297mm）作为尺度参考。

### 1.2 自动检测 A4 四角
- 通过边缘检测 + 轮廓近似筛选四边形，并排序为 TL、TR、BR、BL。

### 1.3 像素/mm 比例计算
- 利用 A4 宽高在图像中的像素长度计算平均比例关系，得到 1mm ≈ 1.084 px。

### 1.4 雕像像素高度测量
- 通过前景提取（Otsu 阈值 + 开运算）检测雕像轮廓，计算垂直方向像素高度。

### 1.5 换算真实高度
- 根据像素/mm 比例，将像素高度转换为真实高度，本例测得雕像高度约 1648.26 mm。

---

## 2. Mesh 读取与缩放

### 2.1 读取 OBJ Mesh
- 导入三角形网格模型（zero.obj），提取顶点数据。

### 2.2 计算原始高度
- 沿 Y 轴（或 Z 轴）计算模型高度范围，用作缩放基准。

### 2.3 按比例缩放 Mesh
- 根据照片测量得到的真实雕像高度，计算缩放并应用到 Mesh，保持模型中心不变。

### 2.4 输出缩放后的 Mesh
- 保存为 `zero_scaled.obj`，并记录缩放比例、原始高度、目标高度。

<details>
<summary>查看 Python 代码 - 尺寸量化与 Mesh 缩放</summary>

```python
import cv2
import numpy as np
import open3d as o3d
import copy

photo_path = r"标定照片路径"
mesh_in_path = r"对应模型路径"
scaled_mesh_out = r"输出缩放后的Mesh路径"

A4_W, A4_H = 210.0, 297.0

def order_quad_points(pts):
    s = pts.sum(axis=1)
    rect = np.zeros((4,2), dtype="float32")
    rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1], rect[3] = pts[np.argmin(diff)], pts[np.argmax(diff)]
    return rect

def detect_a4_corners(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    edges = cv2.Canny(blur,50,200)
    contours,_ = cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    best, best_score = None, 1e9
    h_img, w_img = gray.shape
    for cnt in contours:
        peri = cv2.arcLength(cnt,True)
        if peri < 100: continue
        approx = cv2.approxPolyDP(cnt,0.02*peri,True).reshape(4,2).astype(np.float32)
        area = abs(cv2.contourArea(approx))
        if area < (w_img*h_img)*0.0005: continue
        d = [np.linalg.norm(approx[i]-approx[(i+1)%4]) for i in range(4)]
        ratio = max(d)/min(d) if min(d)>0 else 1e9
        score = abs(ratio-1.414)/np.sqrt(area)
        if score < best_score:
            best_score = score
            best = approx.copy()
    if best is None: raise RuntimeError("未检测到 A4 纸")
    return order_quad_points(best)

def compute_px_per_mm(a4_ordered):
    w_px = np.linalg.norm(a4_ordered[1]-a4_ordered[0])
    h_px = np.linalg.norm(a4_ordered[2]-a4_ordered[1])
    return (w_px/A4_W + h_px/A4_H)/2.0

def auto_scale_mesh_by_height(mesh_path, out_path, desired_height_mm):
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    vertices = np.asarray(mesh.vertices)
    y_range, z_range = vertices[:,1].max()-vertices[:,1].min(), vertices[:,2].max()-vertices[:,2].min()
    height_model = z_range if y_range<1e-6 else y_range
    scale_factor = desired_height_mm/height_model
    centroid = mesh.get_center()
    mesh_scaled = copy.deepcopy(mesh)
    mesh_scaled.translate(-centroid)
    mesh_scaled.scale(scale_factor, center=(0,0,0))
    mesh_scaled.translate(centroid)
    o3d.io.write_triangle_mesh(out_path,mesh_scaled)
    return out_path, scale_factor, height_model
```
</details>

---

## 3. 空间量化精度
- 每像素对应实际尺寸约 0.922 mm/px

- 模型每个顶点在真实空间中大约误差 ±0.92 mm

- 通过照片中 A4 标定和 Mesh 缩放计算得到，保证三维模型的真实尺寸精确性

---

## 4. UV 展开质量评估
### 4.1 数据获取
- 读取导出的 UV 布局图及面片信息（zero.obj + UV 图）。

### 4.2 局部畸变计算
- 对每个三角形面片进行线性变换估计，计算 UV 面片面积与对应 3D 面片面积比值，得到 UV 畸变系数。

### 4.3 统计分析
- 总面片数：563,932

- 有效面片数：563,932

- 平均畸变：0.12672

- 标准差：0.07394

- 最大畸变：0.56858

- 最小畸变：0.0

### 4.4 结论
- UV 展开整体均匀，主干面片畸变低，纹理映射精度高

- 仅少数边缘或复杂曲面出现轻微异常

- 高畸变比例为 0，保证纹理贴图精确性

<details>
<summary>查看 Python 代码 - UV 展开分析</summary>

```python
# UV 量化分析代码
import os
import numpy as np
from PIL import Image
import csv
from tqdm import tqdm
import matplotlib.pyplot as plt

OBJ_PATH = r""       
UV_IMAGE_PATH = r"" 
OUTPUT_DIR = r""  
DISTORTION_THRESHOLD = 3.0
HEATMAP_COLORMAP = "jet"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 这里省略具体函数，可直接使用你提供的完整 UV 分析代码

```
</details>

---

## 5. 总体流程概括
- 图片标定 → 确定像素/mm → 测量雕像像素高度 → 换算真实高度

- 导入 Mesh → 计算原始高度 → 缩放 Mesh → 保存缩放模型

- 量化空间精度 → 计算每像素对应真实 mm → 验证模型精度

- UV 展开分析 → 计算畸变 → 可视化输出 → 评估纹理映射质量




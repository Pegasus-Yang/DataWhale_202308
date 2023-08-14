
## 创建项目

- 创建学习项目和对应的虚拟环境，用于保存后续学习代码
- 进入学习项目目录，在根目录下创建`checkpoint outputs data 图表 Zihao-Configs`几个目录

## 安装依赖

#### 安装pytorch相关

- 由于是arm芯片，不支持CUDA操作，所以直接安装最新版本的pytorch相关依赖即可

```
pip install torch torchvision
```

#### 安装mmcv

- 安装mmcv，命令如下，其中安装mmcv时会有一个比较长时间的building过程，不要急躁，耐心等待一会。

```
pip install -U openmim
mim install mmengine
mim install mmcv
```

#### 安装工具包

- `pip install opencv-python pillow matplotlib seaborn tqdm pytorch-lightning mmdet -i https://pypi.tuna.tsinghua.edu.cn/simple`
- 相比教程，去掉了mmdet的版本要求，理论上安装时会安装最新版本的mmdet，可以在安装之后通过pip list查看一下具体的安装版本

#### 安装MMSegmentation

- 最新的MMSegmentation版本已经来到了v1.1.1，可以在学习项目的上级目录中执行`git clone https://github.com/open-mmlab/mmsegmentation.git -b v1.1.1`将对应的代码拉取下来
- 进入学习项目的虚拟环境中，将命令行所在目录切换到上一步骤拉取的mmsegmentation项目文件夹内，执行命令`pip install -e .`将项目安装到之后需要使用的虚拟环境内。注意，如果对mmsegmentation的项目版本不确定，可以进入项目下的`mmseg/version.py`文件中查看版本号信息

## 检查环境安装情况

- 在项目中编写python文件并运行，查看输出结果是否正常

```python
import torch, torchvision
import mmcv
from mmcv.ops import get_compiling_cuda_version, get_compiler_version
import mmseg
from mmseg.utils import register_all_modules
from mmseg.apis import inference_model, init_model

print('Pytorch 版本', torch.__version__)
print('MPS 是否可用', torch.has_mps)
print('MMCV版本', mmcv.__version__)
print('CUDA版本', get_compiling_cuda_version())
print('编译器版本', get_compiler_version())
print('mmsegmentation版本', mmseg.__version__)
```

- 其中可能会发现有部分import语句倒入的包/模块没有被调用，主要是为了确认导入 OK，也就是包正常存在
- 因为是arm版本的MAC，新版本pytorch已经支持MPS加速，所以修改CUDA确认，改成了MPS相关

## Matplotlib中文字体设置

- 教程中给出了一套直接将字体放入依赖默认位置的修改方式，如果环境不会经常挪动的话还是挺好用的，但是项目更换环境之后就需要重新操作一次，比较麻烦，偷懒的我就找了另一个方法，每次运行之前主动添加字体，这样代码分享出去之后其他人就不需要做改动可以直接运行，两种方法各有利弊，大家可以自行取舍。
- 首先在项目**根目录**下创建fonts文件夹，下载[字体文件](https://zihao-openmmlab.obs.cn-east-3.myhuaweicloud.com/20220716-mmclassification/dataset/SimHei.ttf)到文件夹中，之后在项目根目录创建如下python文件：

```python
import os.path
import matplotlib.font_manager as font_manager

root_dir = os.path.dirname(os.path.abspath(__file__))


def add_custom_fonts():
    font_dirs = [os.sep.join([root_dir, 'fonts'])]
    font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
    for font_file in font_files:
        font_manager.fontManager.addfont(font_file)
```

- 之后涉及到显示Matplotlib图像的语句之前导入并运行`add_custom_fonts()`，这样就可以将字体加入到Matplotlib的字体管理器中了，例如:

```python
import matplotlib
import matplotlib.pyplot as plt

from font_manage import add_custom_fonts

# 将自定义字体加入字体管理器  
add_custom_fonts()
# 中文字体  
matplotlib.rc("font", family='SimHei')
plt.plot([1, 2, 3], [100, 500, 300])
plt.title('matplotlib中文字体测试', fontsize=25)
plt.xlabel('X轴', fontsize=15)
plt.ylabel('Y轴', fontsize=15)
plt.show()
```
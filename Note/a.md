
## 安装

- 已经进行过前一篇环境准备的同学，可以跳过安装步骤直接进行下一步
-
未进行环境准备或者只想尝试一下MMSegmentation的同学可以按照[官方教程](https://mmsegmentation.readthedocs.io/zh_CN/latest/get_started.html)
进行安装准备

## 验证

#### 下载配置和模型文件

- 我是使用一个python项目来记录整个学习进程的，所以文件我也是下载到了项目中对应的目录中，切换到目标目录后，执行命令如下

```
mim download mmsegmentation --config pspnet_r50-d8_4xb2-40k_cityscapes-512x1024 --dest .
```

- 其中的mim是之前安装mmcv是使用的openmim库提供的命令，如果没有的话可以`pip install -U openmim`安装一下

#### 准备数据

- 通过下载的模型文件可以得知，数据主要是针对的**街景**这个场景，所以准备数据时也要按照这个方向来准备，效果才会比较好
- 可以直接自行搜索相关的图片和视频保存到本地，也可以按照教程中给出的数据下载(同济子豪兄教程提供):
    - [伦敦街景图片](https://zihao-openmmlab.obs.cn-east-3.myhuaweicloud.com/20220713-mmdetection/images/street_uk.jpeg)
    - [上海街景视频](https://zihao-download.obs.cn-east-3.myhuaweicloud.com/detectron2/traffic.mp4)
    - [街拍视频](https://zihao-openmmlab.obs.cn-east-3.myhuaweicloud.com/20220713-mmdetection/images/street_20220330_174028.mp4)

#### 运行demo代码

- 首先推荐通过python文件运行，方便记录和修改
- 从[官方教程](https://mmsegmentation.readthedocs.io/zh_CN/latest/get_started.html)步骤2中找到demo代码，保存到本地的py文件内

```python
from mmseg.apis import inference_model, init_model, show_result_pyplot
import mmcv

config_file = 'pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py'
checkpoint_file = 'pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth'

# 根据配置文件和模型文件建立模型
model = init_model(config_file, checkpoint_file, device='cuda:0')

# 在单张图像上测试并可视化
# demo/demo.png 修改为准备好的图片数据路径
img = 'demo/demo.png'  # or img = mmcv.imread(img), 这样仅需下载一次
result = inference_model(model, img)
# 在新的窗口可视化结果
show_result_pyplot(model, img, result, show=True)
# 或者将可视化结果保存到图像文件夹中
# 您可以修改分割 map 的透明度 (0, 1].
show_result_pyplot(model, img, result, show=True, out_file='result.jpg', opacity=0.5)
# 在一段视频上测试并可视化分割结果
# video.mp4修改为准备好的视频数据路径
video = mmcv.VideoReader('video.mp4')
for frame in video:
    result = inference_segmentor(model, frame)
    show_result_pyplot(model, result, wait_time=1)
```

- 修改运行设备，如果没有CUDA支持的话，device参数改为'cpu'，虽然慢，但是可以跑。。有CUDA的童鞋就不用动了

```
model = init_model(config_file, checkpoint_file, device='cuda:0')
```

- 修改图片源，改为之前准备的街景图片数据，写相对路径或者绝对路径都行

```
img = 'demo/demo.png'
```

- 修改视频源，改为之前准备的街景视频数据

```
video = mmcv.VideoReader('video.mp4')
```

-
BUG修复，MMSegmentation版本在1.X的时候，有一些API进行了修改，详见[官方介绍](https://mmsegmentation.readthedocs.io/zh_CN/latest/migration/package.html#id4)

```python
# 函数调用错误
result = inference_segmentor(model, frame)
# 需要改为
result = inference_model(model, frame)
```

```python
# 参数错误
show_result_pyplot(model, result, wait_time=1)
# 需要改为
show_result_pyplot(model, frame, result, wait_time=1)
```

- 改动之后可以按照需求分别注释掉视频/图片的处理部分语句来调试，如果想要连续运行，需要在函数`show_result_pyplot`
  的参数中增加`wait_time=1`的参数，避免代码运行到这一步的时候无限等待，导致下面的语句不能被运行
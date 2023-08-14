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
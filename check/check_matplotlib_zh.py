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

import os.path
import matplotlib.font_manager as font_manager

root_dir = os.path.dirname(os.path.abspath(__file__))


def add_custom_fonts():
    font_dirs = [os.sep.join([root_dir, 'fonts'])]
    font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
    for font_file in font_files:
        font_manager.fontManager.addfont(font_file)

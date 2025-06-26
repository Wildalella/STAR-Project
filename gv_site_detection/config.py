#Stores all tunable constants, paths, thresholds, and filenames in one place.
import numpy as np

IMAGE_PATH = 'data/image.png'
SCALE_BAR_MM = 2.0
SCALE_BAR_ROI_COORDS = (315, 330, 945, 1150)
LOWER_PURPLE_HSV = np.array([125, 100, 10])
UPPER_PURPLE_HSV = np.array([160, 255, 255])
MIN_DOT_AREA_PX = 10
MAX_DOT_AREA_PX = 1000
HEATMAP_GRID_CELL_SIZE_MM = 1.0
DOT_DENSITY_HEATMAP_IMAGE_PATH = 'data/output/heatmaps/dot_density_heatmap.png'

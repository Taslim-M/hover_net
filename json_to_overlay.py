from misc.viz_utils import visualize_instances_dict
from misc.wsi_handler import get_file_handler
import numpy as np
import json
from skimage import io
import cv2
import zarr
import glob


# load the wsi file
wsi_list = glob.glob("dataset/wsicrc" + "/*")
wsi_file = wsi_list[0]
wsi_ext = '.svs'

# load the json file
json_path_wsi = 'test_output/wsicrc/' + 'json/' + 'TCGA2' + '.json'


bbox_list_wsi = []
centroid_list_wsi = []
contour_list_wsi = [] 
type_list_wsi = []

print("Reading JSON file")
# add results to individual lists
with open(json_path_wsi) as json_file:
    data = json.load(json_file)
    mag_info = data['mag']
    nuc_info = data['nuc']
    for inst in nuc_info:
        inst_info = nuc_info[inst]
        inst_centroid = inst_info['centroid']
        centroid_list_wsi.append(inst_centroid)
        inst_contour = inst_info['contour']
        contour_list_wsi.append(inst_contour)
        inst_bbox = inst_info['bbox']
        bbox_list_wsi.append(inst_bbox)
        inst_type = inst_info['type']
        type_list_wsi.append(inst_type)

# let's generate a tile from the WSI
# define the region to select
x_tile = 0
y_tile = 0
w_tile = 100
h_tile = 100

# load the wsi object and read region
wsi_obj = get_file_handler(wsi_file, wsi_ext)
print("get_file_handler complete")
wsi_obj.prepare_reading(read_mag=mag_info)
print("prepare_reading complete")
w_tile, h_tile = wsi_obj.get_dimensions(read_mag=mag_info)
print("get_dimensions complete")

print(w_tile)
print(h_tile)

w_tile = int(w_tile/3)
h_tile = int(h_tile/3)

print("tile dimens: ")
print(w_tile)
print(h_tile)

wsi_tile = wsi_obj.read_region((x_tile,y_tile), (w_tile,h_tile))
print("read_region complete")

# only consider results that are within the tile

coords_xmin = x_tile
coords_xmax = x_tile + w_tile
coords_ymin = y_tile
coords_ymax = y_tile + h_tile

tile_info_dict = {}
count = 0
for idx, cnt in enumerate(contour_list_wsi):
    cnt_tmp = np.array(cnt)
    cnt_tmp = cnt_tmp[(cnt_tmp[:,0] >= coords_xmin) & (cnt_tmp[:,0] <= coords_xmax) & (cnt_tmp[:,1] >= coords_ymin) & (cnt_tmp[:,1] <= coords_ymax)] 
    label = str(type_list_wsi[idx])
    if cnt_tmp.shape[0] > 0:
        cnt_adj = np.round(cnt_tmp - np.array([x_tile,y_tile])).astype('int')
        tile_info_dict[idx] = {'contour': cnt_adj, 'type':label}
        count += 1

# plot the overlay

# the below dictionary is specific to PanNuke checkpoint - will need to modify depeending on categories used
type_info = {
    "0" : ["nolabe", [0  ,   0,   0]], 
    "1" : ["neopla", [255,   0,   0]], 
    "2" : ["inflam", [0  , 255,   0]], 
    "3" : ["connec", [0  ,   0, 255]], 
    "4" : ["necros", [255, 255,   0]], 
    "5" : ["no-neo", [255, 165,   0]],
	"6" : ["epithe", [155, 71,  203]],
	"7" : ["spindl", [80, 212,  226]],
	"8" : ["Miscel", [138, 145,  38]] 
} #Blk, R, G, B, Y, O, V, C, CG

overlaid_img = visualize_instances_dict(wsi_tile, tile_info_dict, type_colour=type_info)
save_path = "test_output/wsicrc/overlay/tr1.tiff"
cv2.imwrite(save_path, cv2.cvtColor(overlaid_img, cv2.COLOR_RGB2BGR))

# f = open('C:/Users/VSAP Lab/Desktop/hover_net-master/test_output/wsicrc/json/TCGA2.json’)
# data = json.load(f)
# type_info = 
# {
#     "0" : ["nolabe", [0  ,   0,   0]], 
#     "1" : ["neopla", [255,   0,   0]], 
#     "2" : ["inflam", [0  , 255,   0]], 
#     "3" : ["connec", [0  ,   0, 255]], 
#     "4" : ["necros", [255, 255,   0]], 
#     "5" : ["no-neo", [255, 165,   0]],
# 	"6" : ["epithe", [155, 71,  203]],
# 	"7" : ["spindl", [80, 212,  226]],
# 	"8" : ["Miscel", [138, 145,  38]] 
# } #Blk, R, G, B, Y, O, V, C, CG

# for i in data['nuc']:
# 	contour = data['nuc'][str(i)]['contour']
# 	x_s = [a_tuple[0] for a_tuple in contour]
# 	y_s = [a_tuple[1] for a_tuple in contour]
# 	coords = np.empty((len(x_s),2))
# 	coords[:,0] = x_s
# 	coords[:,1] = y_s
#     label = str(data['nuc'][str(i)]['type'])
# 	tile_info_dict[i] = {'contour': np.int0(coords), 'type':label}

# # shape of the max mag level in svs
# # r_mask = np.zeros((67343, 143424), np.uint8)
# # overlaid_output = visualize_instances_dict(r_mask, tile_info_dict, type_colour=type_info,line_thickness=cv2.FILLED)
# overlaid_output = visualize_instances_dict(r_mask, tile_info_dict, type_colour=type_info,line_thickness=cv2.FILLED)
# zarr.save('example.zarr',overlaid_output)

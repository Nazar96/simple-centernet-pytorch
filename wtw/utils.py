from typing import List, Tuple, Dict, Optional
import xmltodict
from collections import OrderedDict
import numpy as np
import cv2
from collections import defaultdict


def load_wtw_annotation(file_path: str) -> OrderedDict:
    with open(file_path, 'r') as f:
        data = ''.join(f.readlines())
    data = xmltodict.parse(data)
    return data['annotation']


def _get_cell_dict(bndbox):
    p1 = float(bndbox['x1']), float(bndbox['y1'])
    p2 = float(bndbox['x2']), float(bndbox['y2'])
    p3 = float(bndbox['x3']), float(bndbox['y3'])
    p4 = float(bndbox['x4']), float(bndbox['y4'])
    tmp = {
        'vertices': [p1, p2, p3, p4],
        'center': (
            (p1[0] + p2[0] + p3[0] + p4[0]) / 4,
            (p1[1] + p2[1] + p3[1] + p4[1]) / 4
        ),
        'size': (
            abs(float(bndbox['xmax']) - float(bndbox['xmin'])),
            abs(float(bndbox['ymax']) - float(bndbox['ymin']))
        ),
    }
    return tmp


def get_cells(data):
    result = []
    cells = data['object']
    if not isinstance(cells, list):
        cells = cells['item']

    for cell in cells:
        bndbox = cell['bndbox']
        cell_dict = _get_cell_dict(bndbox)
        result.append(cell_dict)
    return result


def get_keypoints(cells, h, w, pad=2):
    labels = []
    tmp_keypoints = []
    for i, cell in enumerate(cells):
        for j, v in enumerate(cell['vertices']):
            tmp_keypoints.append(v)
            labels.append(('vert', i, j))
        tmp_keypoints.append(cell['center'])
        labels.append(('center', i))

    keypoints = []
    for x, y in tmp_keypoints:
        x, y = max(pad, min(x, w - pad)), max(pad, min(y, h - pad))
        keypoints.append((x, y))
    return keypoints, labels


def keypoints2cells(kpoints):
    tmp_cells = defaultdict(list)
    for i, p in enumerate(kpoints):
        idx = i//5
        tmp_cells[idx].append(p)
    tmp_cells = list(tmp_cells.values())

    tr_cells = []
    for cell in tmp_cells:
        if len(cell) == 5:
            tr_cell = {}
            tr_cell['vertices'] = cell[:4]
            tr_cell['center'] = cell[4]
            tr_cells.append(tr_cell)
    return tr_cells


class WTW:
    def __init__(self):
        pass

#     @staticmethod
#     def __get_cells(
#             data: OrderedDict,
#             size: Optional[Tuple[int, int]] = None,
#     ) -> List[Dict]:
#         width = int(data['size']['width'])
#         height = int(data['size']['height'])

#         kx, ky = 1, 1
#         if size is not None:
#             kx, ky = size[0]/width, size[1]/height

#         result = []
#         cells = data['object']
#         if not isinstance(cells, list):
#             cells = cells['item']
            
#         for cell in cells:
#             bndbox = cell['bndbox']
#             cell_dict = _get_cell_dict(bndbox, kx, ky)
#             result.append(cell_dict)
#         return result

    @staticmethod
    def __get_vert2center(cells: List[Dict], size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        w, h = size
        v2c = {}
        for cell_v in cells:
            for vert in cell_v['vertices']:
                if vert not in v2c:
                    centers = []
                    for cell_c in cells:
                        if vert in cell_c['vertices']:
                            centers.append(cell_c['center'])
                    v2c[vert] = centers

        vertices, centers = [], []
        for vertice, center in v2c.items():
            xv, yv = vertice
            center = [((xv-xc)/w, (yv-yc)/h) for xc, yc in center]
            vertices.append(vertice)
            centers.append(center)

        for c in centers:
            n = len(c)
            if n < 4:
                c.extend([(0, 0) for _ in range(4 - n)])
        centers = [c[0] + c[1] + c[2] + c[3] for c in centers]
        return np.asarray(vertices), np.asarray(centers)

    @staticmethod
    def __get_center2vert(cells: List[Dict], size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        w, h = size
        _vertices = [c['vertices'] for c in cells]
        centers = [c['center'] for c in cells]

        vertices = []
        for vertice, center in zip(_vertices, centers):
            xc, yc = center
            vertice = [((xc-xv)/w, (yc-yv)/h) for xv, yv in vertice]
            vertices.append(vertice)

        vertices = [v[0] + v[1] + v[2] + v[3] for v in vertices]
        return np.asarray(vertices), np.asarray(centers)

    @staticmethod
    def __get_heatmap(points: np.ndarray, size: Tuple[int, int], kernel_size: int = 3, k: int = 4) -> np.ndarray:
        h, w = size
        h, w = h//k, w//k
        heatmap = np.zeros((h, w))
        for x, y in points:
            heatmap = cv2.circle(heatmap, (int(x/k), int(y/k)), 1, 1, -1)
        heatmap = cv2.blur(heatmap, (kernel_size, kernel_size))
        heatmap -= heatmap.min()
        heatmap /= heatmap.max()
        return heatmap

#     @staticmethod
#     def __get_dimension_mask(points, box_sizes, size: Tuple[int, int], pad: int = 2, k: int = 4) -> np.ndarray:
#         h, w = size
#         h, w = h//k, w//k
#         mask = np.zeros((2, h, w))
#         for (x, y), box_size in zip(points, box_sizes):
#             x, y = int(x//k), int(y//k)
#             for i in range(len(box_size)):
#                 x_min, x_max = max(0, x - pad), min(size[1] - 1, x + pad)
#                 y_min, y_max = max(0, y - pad), min(size[0] - 1, y + pad)
#                 mask[i, y_min:y_max, x_min:x_max] = box_size[i]
#         return mask

    @staticmethod
    def __get_coord_mask(points, coords, size: Tuple[int, int], pad: int = 2, k: int = 4) -> np.ndarray:
        h, w = size
        h, w = h//k, w//k
        mask = np.zeros((8, h, w))
        for point, coord in zip(points, coords):
            x, y = point
            x, y = int(x/k), int(y/k)
            for i in range(len(coord)):
                x_min, x_max = max(0, x - pad), min(size[1] - 1, x + pad)
                y_min, y_max = max(0, y - pad), min(size[0] - 1, y + pad)
                mask[i, y_min:y_max, x_min:x_max] = coord[i]
        return mask

    def __call__(self, cells, size=(512, 512), k=4) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:        

        vertices = []
        for c in cells:
            vertices += c['vertices']
        vertices = np.asarray(vertices)
        centers = np.asarray([c['center'] for c in cells])

        vertice_heatmap = self.__get_heatmap(vertices, size, k=k)
        center_heatmap = self.__get_heatmap(centers, size, k=k)

        v, c = self.__get_vert2center(cells, size)
        v2c_heatmap = self.__get_coord_mask(v, c, size, k=k)

        v, c = self.__get_center2vert(cells, size)
        c2v_heatmap = self.__get_coord_mask(c, v, size, k=k)

        heatmap = np.asarray([vertice_heatmap, center_heatmap])

        return heatmap, v2c_heatmap, c2v_heatmap

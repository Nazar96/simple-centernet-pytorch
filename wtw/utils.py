from typing import List, Tuple, Dict, Optional
import xmltodict
from collections import OrderedDict
import numpy as np
import cv2


def load_wtw_annotation(file_path: str) -> OrderedDict:
    with open(file_path, 'r') as f:
        data = ''.join(f.readlines())
    data = xmltodict.parse(data)
    return data['annotation']


class WTW:
    def __init__(self, file: str):
        self.data = load_wtw_annotation(file)

    @staticmethod
    def __get_cells(
            data: OrderedDict,
            size: Optional[Tuple[int, int]] = None,
    ) -> List[Dict]:
        width = int(data['size']['width'])
        height = int(data['size']['height'])

        kx, ky = 1, 1
        if size is not None:
            kx, ky = size[0]/width, size[1]/height

        result = []
        for cell in data['object']:
            bndbox = cell['bndbox']
            p1 = float(bndbox['x1'])*kx, float(bndbox['y1'])*ky
            p2 = float(bndbox['x2'])*kx, float(bndbox['y2'])*ky
            p3 = float(bndbox['x3'])*kx, float(bndbox['y3'])*ky
            p4 = float(bndbox['x4'])*kx, float(bndbox['y4'])*ky
            tmp = {
                'vertices': [p1, p2, p3, p4],
                'center': (
                    (p1[0] + p2[0] + p3[0] + p4[0]) / 4,
                    (p1[1] + p2[1] + p3[1] + p4[1]) / 4
                ),
                'size': (
                    abs(float(bndbox['xmax']) - float(bndbox['xmin']))*kx,
                    abs(float(bndbox['ymax']) - float(bndbox['ymin']))*ky
                ),
            }

            result.append(tmp)
        return result

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
    def __get_heatmap(points: np.ndarray, size: Tuple[int, int], kernel_size: int = 3) -> np.ndarray:
        heatmap = np.zeros(size)
        for x, y in points:
            heatmap = cv2.circle(heatmap, (int(x), int(y)), 2, 1, -1)
        heatmap = cv2.blur(heatmap, (kernel_size, kernel_size))
        heatmap -= heatmap.min()
        heatmap /= heatmap.max()
        return heatmap

    @staticmethod
    def __get_dimension_mask(points, box_sizes, size: Tuple[int, int]) -> np.ndarray:
        mask = np.zeros((2, size[0], size[1]))
        for (x, y), box_size in zip(points, box_sizes):
            x = size[1] - 1 if x >= size[1] else x
            y = size[0] - 1 if y >= size[0] else y
            mask[:, int(y), int(x)] = box_size
        return mask

    @staticmethod
    def __get_coord_mask(points, coords, size: Tuple[int, int]) -> np.ndarray:
        mask = np.zeros((8, size[0], size[1]))
        for point, coord in zip(points, coords):
            x, y = point
            x = size[1] - 1 if x >= size[1] else x
            y = size[0] - 1 if y >= size[0] else y
            mask[:, int(y), int(x)] = coord
        return mask

    def __call__(self, size: Optional[Tuple[int, int]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cells = self.__get_cells(self.data, size)

        vertices = []
        for c in cells:
            vertices += c['vertices']
        vertices = np.asarray(vertices)
        centers = np.asarray([c['center'] for c in cells])
        sizes = np.asarray([c['size'] for c in cells])

        if size is None:
            size = (int(self.data['size']['height']), int(self.data['size']['width']))

        vertice_heatmap = self.__get_heatmap(vertices, size)
        center_heatmap = self.__get_heatmap(centers, size)

        v, c = self.__get_vert2center(cells, size)
        v2c_heatmap = self.__get_coord_mask(v, c, size)

        v, c = self.__get_center2vert(cells, size)
        c2v_heatmap = self.__get_coord_mask(c, v, size)

        heatmap = np.asarray([vertice_heatmap, center_heatmap])
        dimension_mask = self.__get_dimension_mask(centers, sizes, size)

        return heatmap, dimension_mask, v2c_heatmap, c2v_heatmap

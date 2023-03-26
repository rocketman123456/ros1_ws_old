
import os
from struct import unpack
from typing import List
import numpy as np
import open3d as o3d
from tqdm import tqdm
from classes import Plane, Result, through_crop


class IOHelper:
    def __init__(self, cloud_path: str, gt_path: str, algo_path: str) -> None:
        self._path_cloud = cloud_path
        self._path_gt = gt_path
        self._path_algo = algo_path
        self.dataset_path, _ = gt_path.rsplit('/', 1)
        _, self.dataset, self.method = self._path_algo.rsplit('/', 2)

    def read_gt(self) -> List[Plane]:
        return self._read(self._path_gt)

    def read_algo(self) -> List[Plane]:
        return self._read(self._path_algo)

    def _read(self, path: str) -> List[Plane]:
        if os.path.isdir(path):
            for file in os.listdir(path):
                if 'time' in file:
                    continue
                if file.endswith('.geo'):
                    return self.read_planes_geo(os.path.join(path, file))
                with open(os.path.join(path, file), 'r') as f:
                    if len(f.readline().split(' ')) > 1:
                        return self.read_planes_xyz_from_folder(path)
                    else:
                        return self.read_planes_i_from_folder(path)
            return []
        elif path.endswith('.geo'):
            return self.read_planes_geo(path)
        elif path.endswith('asc'):
            return [Plane.xyzfrom_txt(path)]
        with open(path, 'r') as f:
            if len(f.readline().split(' ')) > 1:
                return [Plane.xyzfrom_txt(path)]

        l: List[Plane] = [self.read_plane_i(path)]
        return l

    def read_planes_geo(self, filename: str) -> List[Plane]:
        planes: List[Plane] = []
        l = []
        with open(filename, "rb") as file:
            file.read(8)  # numcircles
            num_planes = int(file.read(8)[0])
            print(f'{num_planes = }')
            for i in range(num_planes):
                p: Plane
                color = [unpack('f', file.read(4)) for _ in range(3)]
                center = [unpack('f', file.read(4)) for _ in range(3)]
                normal = [unpack('f', file.read(4))[0] for _ in range(3)]
                basisu = [unpack('f', file.read(4)) for _ in range(3)]
                basisv = [unpack('f', file.read(4)) for _ in range(3)]
                num_in = unpack('N', file.read(8))[0]
                p = Plane()
                p.indices = []
                p.normal = normal
                for inl in range(num_in):
                    point = unpack('N', file.read(8))[0]
                    p.indices.append(point)
                p.name = filename.split('.')[0]
                p.set_indices = set(p.indices)
                planes.append(p)
                l.append(num_in)
        return planes

    def read_plane_xyz(self, filename: str):
        p = Plane.xyzfrom_txt(filename)
        return p

    def read_planes_xyz_from_folder(self, path: str) -> List[Plane]:
        planes = []
        for file in os.listdir(path):
            if 'time' not in file:
                planes.append(self.read_plane_xyz(os.path.join(path, file)))
        return planes

    def read_plane_i(self, filename: str) -> Plane:
        p = Plane.i_from_txt(filename)
        return p

    def read_planes_i_from_folder(self, path: str) -> List[Plane]:
        planes = []
        for file in os.listdir(path):
            if 'time' in file: 
                continue
            planes.append(self.read_plane_i(os.path.join(path, file)))
        return planes

    def _xyzfrom_bytes(self, b):
        x = unpack('f', b[:4])[0]
        y = unpack('f', b[4:8])[0]
        z = unpack('f', b[8:12])[0]
        # p = Point(x,y,z)
        return [x, y, z]

    def read_pc_pcl(self) -> np.ndarray:
        points: List[List[float]] = []
        with open(self._path_cloud, "rb") as file:
            size = unpack('N', file.read(8))[0]
            mode = unpack('N', file.read(8))[0]
            print(f'{size = }, {mode = }')
            for _ in tqdm(range(size)):
                c = self._xyzfrom_bytes(file.read(12))
                points.append(c)
                if mode & 1:
                    file.read(12)
                if mode & 2:
                    file.read(4)
                if mode & 4:
                    file.read(12)
                if mode & 8:
                    file.read(4)
                if mode & 16:
                    file.read(4)
        return np.array(points)

    def read_pc_xyz(self, path=None) -> np.ndarray:
        points: np.ndarray = np.loadtxt(
            path or self._path_cloud, dtype=float,usecols=(0, 1, 2)).tolist()
        return points

    def save_results(self, p: float, r: float, f1: float, found_planes: int, all_planes: int, time_total: float, time_per_plane: float, time_per_sample: float, time="") -> None:
        result = Result(p, r, f1, found_planes, all_planes, self.dataset,
                        self.method, time_total, time_per_plane, time_per_sample)

        output_folder = os.path.join(self.dataset_path, 'results')
        if 'results' not in os.listdir(self.dataset_path):
            os.mkdir(output_folder)
        if time != "":
            output_file = os.path.join(
                output_folder, f'{self.dataset}_{self.method}_{time}.out')
        else:
            output_file = os.path.join(
                output_folder, f'{self.dataset}_{self.method}.out')
        result.to_file(output_file)

    def get_times(self, path=""):
        if path != "":
            return np.loadtxt(path, skiprows=1, dtype=float)
        for file in os.listdir(self._path_algo):
            if 'time' in file:
                path = os.path.join(self._path_algo, file)
                break
        else:
            print(
                f'no time results found for {self.method} and {self.dataset}!')
            return 0, 0, 0
        return np.loadtxt(path, dtype=float).reshape(3)

    def read_cloud(self, path=None) -> np.ndarray:
        print('Reading Point cloud')
        path = path or self._path_cloud
        if path.endswith('.pcl'):
            return self.read_pc_pcl()
        elif path.endswith('.txt') or path.endswith('.asc'):
            return self.read_pc_xyz(path)
        else:  # pcd file
            return np.empty(0)

    def read_pcd(self, path=None) -> o3d.geometry.PointCloud:
        """if path is not set, use self._path_cloud"""
        return o3d.io.read_point_cloud(path or self._path_cloud)

    def read_kht(self, s_level: int):
        planes = []
        for file in os.listdir(self._path_algo):
            if 'plane' not in file:
                continue
            if len(a:=file.split('-')) > 1 and int(a[1][0]) == s_level:
                planes.append(Plane.xyzfrom_txt(os.path.join(self._path_algo, file)))
        return planes

    def save_kht_results(self, p: float, r: float, f1: float, found_planes: int, all_planes: int, time_total: float, time_per_plane: float, time_per_sample: float, s_l):
        result = Result(p, r, f1, found_planes, all_planes, self.dataset,
                        self.method, time_total, time_per_plane, time_per_sample)

        output_folder = os.path.join(self.dataset_path, 'results')
        if 'results' not in os.listdir(self.dataset_path):
            os.mkdir(output_folder)
        output_file = os.path.join(
            output_folder, f'{self.dataset}_{s_l}_{self.method}.out')

        result.to_file(output_file)

    def get_frames(self, path: str):
        self.frame_path = path
        frames = []
        for file in reversed(sorted(os.listdir(path))):
            if file.endswith('.pcd') and 'nope_' not in file:
                seconds, micro, _ = file.split('.')
                frames.append(f'{seconds[-4:]}.{micro[:4]}')
        return frames

    def get_frame_data(self, frame:str, voxel_grid, pointcloud):
        """returns the cloud, gt and test data for given frame"""
        cloud = gt = algo = None
        gt = self.read_gt()
        for file in os.listdir(self.frame_path):
            if frame in file and file.endswith('.pcd'):
                cloud = self.read_pcd(os.path.join(self.frame_path,file))
                break
        algo = []
        for file in os.listdir(self._path_algo):
            if frame in file and 'time' not in file:
                if self.method == 'RSPD':
                    algo = self._read(os.path.join(self._path_algo, file))
                    break
                else:
                    for p in self._read(os.path.join(self._path_algo, file)):
                        algo.append(p)
        cropped_gt = list(map(lambda plane: through_crop(plane,cloud, voxel_grid,pointcloud), gt))
        cropped_gt = [plane for plane in cropped_gt if plane is not None]
        return cloud, cropped_gt, algo

def create_txt(path: str):
    pc = o3d.io.read_point_cloud(path)
    with open(path.replace('.pcd', '.txt'), 'w') as file:
        for p in pc.points:
            file.write(f'{p[0]} {p[1]} {p[2]}\n')


def create_pcd(filepath: str):
    """OPS expects PCL file format so we are quickly going to convert the .txt file"""
    with open(filepath, "r") as inf, open(filepath.replace('.txt', '.pcd'), "w") as of:
        of.write("# .PCD v0.7 - Point Cloud Data file format\n")
        of.write("VERSION 0.7\nFIELDS x y z\n")
        of.write("SIZE 4 4 4\nTYPE F F F\n")
        of.write("COUNT 1 1 1\n")
        xyz = []
        points = 0
        for line in inf.readlines():
            l = line.split(" ")
            xyz.append(" ".join(l[:3]))
            points += 1
        of.write(f"WIDTH {points}\nHEIGHT 1\n")
        of.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        of.write(f"POINTS {points}\nDATA ascii\n")
        for coords in xyz:
            of.write(f"{coords}\n")

if __name__ == '__main__':
    path = "FIN-Dataset/auditorium/1664003770.004746437.pcd"
    create_txt(path)
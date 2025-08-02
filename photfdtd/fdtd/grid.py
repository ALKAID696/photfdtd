""" The FDTD Grid

The grid is the core of the FDTD Library. It is where everything comes
together and where the biggest part of the calculations are done.

"""

## Imports

# standard library
import os
from os import path, makedirs, chdir, remove
from subprocess import check_call, CalledProcessError
from glob import glob
from datetime import datetime

import numpy as np
# 3rd party
from tqdm import tqdm
from numpy import savez, sqrt

# typing
from .typing_ import Tuple, Number, Tensorlike

# relative
from .backend import backend as bd
from . import constants as const
from .conversions import *

# plot
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation


## Functions


## FDTD Grid Class
class Grid:
    """The FDTD Grid

    The grid is the core of the FDTD Library. It is where everything comes
    together and where the biggest part of the calculations are done.

    """

    from .visualization import visualize

    def __init__(
            self,
            shape: Tuple[Number, Number, Number],
            grid_spacing: float = 155e-9,
            grid_spacing_x: float = None,
            grid_spacing_y: float = None,
            grid_spacing_z: float = None,
            x_coordinates=None,
            y_coordinates=None,
            z_coordinates=None,
            permittivity: float = 1.0,
            permeability: float = 1.0,
            courant_number: float = None,
            folder: str = None,
    ):
        """
        Args:
            shape: shape of the FDTD grid.
            grid_spacing: distance between the grid cells.
            permittivity: the relative permittivity of the background.
            permeability: the relative permeability of the background.
            courant_number: the courant number of the FDTD simulation.
                Defaults to the inverse of the square root of the number of
                dimensions > 1 (optimal value). The timestep of the simulation
                will be derived from this number using the CFL-condition.
        """
        # save the grid spacing
        # Currently self.grid_spacing
        self.background_index = permittivity ** 0.5
        self.grid_spacing = float(grid_spacing or 1.0)

        # 读取网格尺寸
        self.Nx, self.Ny, self.Nz = self._handle_tuple(shape)

        # 处理每个方向的局部 cell 宽度和边界坐标
        # ———— X 方向 ————
        if x_coordinates is not None:
            # 用户给出的是节点坐标数组，长度必须为 Nx+1
            assert len(x_coordinates) == self.Nx + 1, "x_coordinates 长度必须等于 Nx+1（节点数）"
            # 直接当作节点坐标
            self.x_coordinates = np.array(x_coordinates, dtype=float)
            # 计算每个单元宽度 dx（长度为 Nx）
            self.dx = self.x_coordinates[1:] - self.x_coordinates[:-1]
        else:
            # 均匀网格：优先使用单向指定的 spacing，否则用 self.grid_spacing
            dx_val = float(grid_spacing_x or self.grid_spacing)
            self.dx = np.full(self.Nx, dx_val)
            # 从 dx 累加生成节点坐标（0 到 Nx*dx_val，总共 Nx+1 点）
            self.x_coordinates = np.concatenate(([0.0], np.cumsum(self.dx)))

        # ———— Y 方向 ————
        if y_coordinates is not None:
            assert len(y_coordinates) == self.Ny + 1, "y_coordinates 长度必须等于 Ny+1（节点数）"
            self.y_coordinates = np.array(y_coordinates, dtype=float)
            self.dy = self.y_coordinates[1:] - self.y_coordinates[:-1]
        else:
            dy_val = float(grid_spacing_y or self.grid_spacing)
            self.dy = np.full(self.Ny, dy_val)
            self.y_coordinates = np.concatenate(([0.0], np.cumsum(self.dy)))

        # ———— Z 方向 ————
        if z_coordinates is not None:
            assert len(z_coordinates) == self.Nz + 1, "z_coordinates 长度必须等于 Nz+1（节点数）"
            self.z_coordinates = np.array(z_coordinates, dtype=float)
            self.dz = self.z_coordinates[1:] - self.z_coordinates[:-1]
        else:
            dz_val = float(grid_spacing_z or self.grid_spacing)
            self.dz = np.full(self.Nz, dz_val)
            self.z_coordinates = np.concatenate(([0.0], np.cumsum(self.dz)))

        # 全局最小间距用于 Courant 条件
        self.grid_spacing_x = float(self.dx.min())
        self.grid_spacing_y = float(self.dy.min())
        self.grid_spacing_z = float(self.dz.min())

        # 模拟维度 D
        self.D = int(self.Nx > 1) + int(self.Ny > 1) + int(self.Nz > 1)

        # Courant 数
        max_cn = float(self.D) ** (-0.5)
        if courant_number is None:
            self.courant_number = 0.99 * max_cn
        elif courant_number > max_cn:
            raise ValueError(f"courant_number {courant_number} 超过 {self.D}D 稳定上限")
        else:
            self.courant_number = float(courant_number)


        # 时间步长，用最小局部 spacing 保证稳定
        self.time_step = self.courant_number * min(
            float(np.min(self.dx)),
            float(np.min(self.dy)),
            float(np.min(self.dz))
        ) / const.c

        # 初始化场
        self.E = bd.zeros((self.Nx, self.Ny, self.Nz, 3))
        self.H = bd.zeros((self.Nx, self.Ny, self.Nz, 3))

        # 介电和磁导逆张量
        if bd.is_array(permittivity) and permittivity.ndim == 3:
            permittivity = permittivity[:, :, :, None]
        self.inverse_permittivity = bd.ones((self.Nx, self.Ny, self.Nz, 3)) / bd.array(
            permittivity, dtype=bd.float
        )
        if bd.is_array(permeability) and permeability.ndim == 3:
            permeability = permeability[:, :, :, None]
        self.inverse_permeability = bd.ones((self.Nx, self.Ny, self.Nz, 3)) / bd.array(
            permeability, dtype=bd.float
        )

        # 物理域真实长度
        self.Lx = float(self.x_coordinates.sum())
        self.Ly = float(self.y_coordinates.sum())
        self.Lz = float(self.z_coordinates.sum())

        # Priority matrix of the grid, default to be a all-zero matrix, indicates the background of the sim region.
        self.priority = bd.zeros((self.Nx, self.Ny, self.Nz))

        # save current time index
        self.time_steps_passed = 0

        # dictionary containing the sources:
        self.sources = []

        # dictionary containing the boundaries
        self.boundaries = []

        # dictionary containing the detectors
        self.detectors = []

        # dictionary containing the objects in the grid
        self.objects = []

        # folder path to store the simulation
        self.folder = folder

    def _handle_distance(self, distance: Number, axis: "x") -> int:
        """transform a distance to an integer number of gridpoints"""
        if axis == "x":
            if not isinstance(distance, int):
                return int(float(distance) / self.grid_spacing_x + 0.5)
            return distance

        if axis == "y":
            if not isinstance(distance, int):
                return int(float(distance) / self.grid_spacing_y + 0.5)
            return distance

        if axis == "z":
            if not isinstance(distance, int):
                return int(float(distance) / self.grid_spacing_z + 0.5)
            return distance

    def curl_E(self, E: Tensorlike) -> Tensorlike:
        """Transforms an E-type field into an H-type field by performing a curl
        operation

        Args:
            E: Electric field to take the curl of (E-type field located on the
               edges of the grid cell [integer gridpoints])

        Returns:
            The curl of E (H-type field located on the faces of the grid [half-integer grid points])
        du∇ × E[m, n, p]
        """
        curl = bd.zeros(E.shape, dtype=E.dtype)

        curl[:, :-1, :, 0] += (E[:, 1:, :, 2] - E[:, :-1, :, 2])
        curl[:, :, :-1, 0] -= (E[:, :, 1:, 1] - E[:, :, :-1, 1])

        curl[:, :, :-1, 1] += (E[:, :, 1:, 0] - E[:, :, :-1, 0])
        curl[:-1, :, :, 1] -= (E[1:, :, :, 2] - E[:-1, :, :, 2])

        curl[:-1, :, :, 2] += (E[1:, :, :, 1] - E[:-1, :, :, 1])
        curl[:, :-1, :, 2] -= (E[:, 1:, :, 0] - E[:, :-1, :, 0])

        return curl

    def curl_E_with_nonuniform_grid(self, E: Tensorlike) -> Tensorlike:
        """Transforms an E-type field into an H-type field by performing a curl
        operation

        Args:
            E: Electric field to take the curl of (E-type field located on the
               edges of the grid cell [integer gridpoints])

        Returns:
            The curl of E (H-type field located on the faces of the grid [half-integer grid points])
        ∇ × E[m, n, p]
        """
        curl = bd.zeros(E.shape, dtype=E.dtype)

        # ∂E_z/∂y at H_x faces
        curl[:, :-1, :, 0] += (E[:, 1:, :, 2] - E[:, :-1, :, 2]) / self.dy[:-1][None, :, None]
        # ∂E_y/∂z at H_x faces
        curl[:, :, :-1, 0] -= (E[:, :, 1:, 1] - E[:, :, :-1, 1]) / self.dz[:-1][None, None, :]

        # ∂E_x/∂z at H_y faces
        curl[:, :, :-1, 1] += (E[:, :, 1:, 0] - E[:, :, :-1, 0]) / self.dz[:-1][None, None, :]
        # ∂E_z/∂x at H_y faces
        curl[:-1, :, :, 1] -= (E[1:, :, :, 2] - E[:-1, :, :, 2]) / self.dx[:-1][:, None, None]

        # ∂E_y/∂x at H_z faces
        curl[:-1, :, :, 2] += (E[1:, :, :, 1] - E[:-1, :, :, 1]) / self.dx[:-1][:, None, None]
        # ∂E_x/∂y at H_z faces
        curl[:, :-1, :, 2] -= (E[:, 1:, :, 0] - E[:, :-1, :, 0]) / self.dy[:-1][None, :, None]

        return curl

    def curl_H(self, H: Tensorlike) -> Tensorlike:
        """Transforms an H-type field into an E-type field by performing a curl
        operation

        Args:
            H: Magnetic field to take the curl of (H-type field located on half-integer grid points)

        Returns:
            The curl of H (E-type field located on the edges of the grid [integer grid points])
        du∇ × H[m, n, p]
        """
        curl = bd.zeros(H.shape, dtype=H.dtype)

        curl[:, 1:, :, 0] += (H[:, 1:, :, 2] - H[:, :-1, :, 2])
        curl[:, :, 1:, 0] -= (H[:, :, 1:, 1] - H[:, :, :-1, 1])

        curl[:, :, 1:, 1] += (H[:, :, 1:, 0] - H[:, :, :-1, 0])
        curl[1:, :, :, 1] -= (H[1:, :, :, 2] - H[:-1, :, :, 2])

        curl[1:, :, :, 2] += (H[1:, :, :, 1] - H[:-1, :, :, 1])
        curl[:, 1:, :, 2] -= (H[:, 1:, :, 0] - H[:, :-1, :, 0])

        return curl

    def curl_H_with_nonuniform_grid(self, H: Tensorlike) -> Tensorlike:
        """Transforms an H-type field into an E-type field by performing a curl
        operation

        Args:
            H: Magnetic field to take the curl of (H-type field located on half-integer grid points)

        Returns:
            The curl of H (E-type field located on the edges of the grid [integer grid points])
        ∇ × H[m, n, p]
        """
        curl = bd.zeros(H.shape, dtype=H.dtype)

        # ∂H_z/∂y at E_x edges
        curl[:, 1:, :, 0] += (H[:, 1:, :, 2] - H[:, :-1, :, 2]) / self.dy[:-1][None, :, None]
        # ∂H_y/∂z at E_x edges
        curl[:, :, 1:, 0] -= (H[:, :, 1:, 1] - H[:, :, :-1, 1]) / self.dz[:-1][None, None, :]

        # ∂H_x/∂z at E_y edges
        curl[:, :, 1:, 1] += (H[:, :, 1:, 0] - H[:, :, :-1, 0]) / self.dz[:-1][None, None, :]
        # ∂H_z/∂x at E_y edges
        curl[1:, :, :, 1] -= (H[1:, :, :, 2] - H[:-1, :, :, 2]) / self.dx[:-1][:, None, None]

        # ∂H_y/∂x at E_z edges
        curl[1:, :, :, 2] += (H[1:, :, :, 1] - H[:-1, :, :, 1]) / self.dx[:-1][:, None, None]
        # ∂H_x/∂y at E_z edges
        curl[:, 1:, :, 2] -= (H[:, 1:, :, 0] - H[:, :-1, :, 0]) / self.dy[:-1][None, :, None]

        return curl

    def _handle_time(self, time: Number) -> int:
        """transform a time value to an integer number of timesteps"""

        if not isinstance(time, int):
            return int(float(time) / self.time_step + 0.5)
        return time

    def _handle_tuple(
            self, shape: Tuple[Number, Number, Number]
    ) -> Tuple[int, int, int]:
        """validate the grid shape and transform to a length-3 tuple of ints"""
        if len(shape) != 3:
            raise ValueError(
                f"invalid grid shape {shape}\n"
                f"grid shape should be a 3D tuple containing floats or ints"
            )
        x, y, z = shape
        x = self._handle_distance(x, "x")
        y = self._handle_distance(y, "y")
        z = self._handle_distance(z, "z")
        return x, y, z

    def _handle_slice(self, s: slice, axis: "x") -> slice:
        """validate the slice and transform possibly float values to ints"""
        start = (
            s.start
            if not isinstance(s.start, float)
            else self._handle_distance(s.start, axis)
        )
        stop = (
            s.stop if not isinstance(s.stop, float) else self._handle_distance(s.stop, axis)
        )
        step = (
            s.step if not isinstance(s.step, float) else self._handle_distance(s.step, axis)
        )
        return slice(start, stop, step)

    def _handle_single_key(self, key, axis="x"):
        """transform a single index key to a slice or list"""
        try:
            len(key)
            return [self._handle_distance(k, axis) for k in key]
        except TypeError:
            if isinstance(key, slice):
                return self._handle_slice(key, axis)
            else:
                return [self._handle_distance(key, axis)]
        return key

    @property
    def x(self) -> float:
        return self.Lx

    @property
    def y(self) -> float:
        return self.Ly

    @property
    def z(self) -> float:
        return self.Lz

    @property
    def shape(self) -> Tuple[int, int, int]:
        """get the shape of the FDTD grid"""
        return (self.Nx, self.Ny, self.Nz)

    @property
    def time_passed(self) -> float:
        """get the total time passed"""
        return self.time_steps_passed * self.time_step

    def run(self, total_time: Number = None, progress_bar: bool = True, interval: int = 100):
        """run an FDTD simulation.

        Args:
            total_time: the total time for the simulation to run.
            progress_bar: choose to show a progress bar during
                simulation

        """
        if isinstance(total_time, float):
            total_time /= self.time_step
        self.total_time = int(total_time)
        time = range(0, self.total_time, 1)
        if progress_bar:
            time = tqdm(time)
        if self.animate:
            if os.path.exists(self.folder + "/frames"):
                for file_name in os.listdir(self.folder + "/frames"):
                    os.remove(self.folder + "/frames/" + file_name)
            else:
                os.makedirs(self.folder + "/frames")
            self.folder_frames = self.folder + "/frames"
        for det in self.detectors:
            det.__init_h5file__()
        for _ in time:
            self.step(interval=interval)

    def step(self, interval=100):
        """do a single FDTD step by first updating the electric field and then
        updating the magnetic field
        """
        self.update_E()
        self.update_H()
        if self.animate and self.time_steps_passed % interval == 0:
            self.save_frame()
        self.time_steps_passed += 1

    def save_frame(self, axis="y", axis_index=0):
        # TODO: for 3d simulation
        """save frames for animation"""

        if "self._Epol" not in locals():
            self._Epol = 'xyz'.index(self.sources[0].polarization)
        if "self.max_abs" not in locals():
            # self.max_abs = 1
            self.max_abs = np.max(simE_to_worldE(np.abs(self.E[:, :, :, self._Epol])))

        fig, ax = plt.subplots()
        if self.Nx == 1:
            axis = "x"
        elif self.Ny == 1:
            axis = "y"
        elif self.Nz == 1:
            axis = "z"
        else:
            # 3d仿真，自动绘制grid中心面上的场分布。3D simulation, plot the field distribution on the center plane of the grid.
            axis_index = int(self.E.shape[letter_to_number(axis)] / 2)
        if axis == "x":
            im = ax.imshow(simE_to_worldE(np.transpose(self.E[axis_index, :, :, self._Epol])), cmap="RdBu", interpolation="nearest", aspect="auto",
                           origin="lower", vmin=-self.max_abs, vmax=self.max_abs)
            ax.set_xlabel("y")
            ax.set_ylabel("z")
        elif axis == "y":
            im = ax.imshow(simE_to_worldE(np.transpose(self.E[:, axis_index, :, self._Epol])), cmap="RdBu", interpolation="nearest", aspect="auto",
                           origin="lower", vmin=-self.max_abs, vmax=self.max_abs)
            ax.set_xlabel("x")
            ax.set_ylabel("z")
        elif axis == "z":
            im = ax.imshow(simE_to_worldE(np.transpose(self.E[:, :, axis_index, self._Epol])), cmap="RdBu", interpolation="nearest", aspect="auto",
                           origin="lower", vmin=-self.max_abs, vmax=self.max_abs)
            ax.set_xlabel("x")
            ax.set_ylabel("y")

        cbar = plt.colorbar(im)
        cbar.set_label(f"E{number_to_letter(self._Epol)} V/m")
        plt.title(f"{self.time_steps_passed} time steps")
        plt.savefig(f"{self.folder_frames}/E_{self.time_steps_passed}.png")
        plt.close(fig)  # 自动关闭图形

    def update_E(self):
        """update the electric field by using the curl of the magnetic field"""

        # update boundaries: step 1
        for boundary in self.boundaries:
            if hasattr(boundary, "loc") and isinstance(boundary.loc, tuple):
                sx, sy, sz = boundary.loc[:3]

                # 从主 Grid 中裁剪对应 slice 的 dx/dy/dz
                dx_local = self.dx[sx] if self.dx is not None else None
                dy_local = self.dy[sy] if self.dy is not None else None
                dz_local = self.dz[sz] if self.dz is not None else None

                boundary.update_phi_E(dx=dx_local, dy=dy_local, dz=dz_local)
            else:
                # fallback，如果没有 loc 则传整个
                boundary.update_phi_E(dx=self.dx, dy=self.dy, dz=self.dz)

        curl = self.curl_H_with_nonuniform_grid(self.H)
        # Before: self.E += self.courant_number * self.inverse_permittivity * curl
        self.E += const.c * self.time_step * self.inverse_permittivity * curl

        # update objects
        # for obj in self.objects:
        #     # 在添加波导后，fdtd会把波导区域中grid.inverse_permittivity设为0，这样波导区域内电磁场的更新就被放在了这里
        #       Edited in 2023/12/25 by Tao Jia. Now the inverse permittivity of objects is added into the grid.
        #     obj.update_E(curl)

        # update boundaries: step 2
        for boundary in self.boundaries:
            boundary.update_E()

        # add sources to grid:
        for src in self.sources:
            src.update_E()

        # detect electric field
        for det in self.detectors:
            det.detect_E()

    def update_H(self):
        """update the magnetic field by using the curl of the electric field"""

        # update boundaries: step 1
        for boundary in self.boundaries:
            if hasattr(boundary, "loc") and isinstance(boundary.loc, tuple):
                sx, sy, sz = boundary.loc[:3]
                dx_local = self.dx[sx] if self.dx is not None else None
                dy_local = self.dy[sy] if self.dy is not None else None
                dz_local = self.dz[sz] if self.dz is not None else None
                boundary.update_phi_H(dx=dx_local, dy=dy_local, dz=dz_local)
            else:
                boundary.update_phi_H(dx=self.dx, dy=self.dy, dz=self.dz)

        curl = self.curl_E_with_nonuniform_grid(self.E)
        # Before: self.H -= self.courant_number * self.inverse_permeability * curl
        # self.H -= self.time_step * self.inverse_permeability * curl / sqrt(const.mu0)
        self.H -= const.c * self.time_step * self.inverse_permeability * curl

        # # update objects
        # for obj in self.objects:
        #     obj.update_H(curl)

        # update boundaries: step 2
        for boundary in self.boundaries:
            boundary.update_H()

        # add sources to grid:
        for src in self.sources:
            src.update_H()

        # detect electric field
        for det in self.detectors:
            det.detect_H()

    def reset(self):
        """reset the grid by setting all fields to zero"""
        self.H *= 0.0
        self.E *= 0.0
        self.time_steps_passed *= 0

    def add_source(self, name, source):
        """add a source to the grid"""
        source._register_grid(self)
        self.sources[name] = source
        # if not hasattr(self, "source_profile"):
        #     self.source_profile = {}
        # self.source_profile += {name: bd.empty((3))}

    def add_boundary(self, name, boundary):
        """add a boundary to the grid"""
        boundary._register_grid(self)
        self.boundaries[name] = boundary

    def add_detector(self, name, detector):
        """add a detector to the grid"""
        detector._register_grid(self)
        self.detectors[name] = detector

    def add_object(self, name, obj):
        """add an object to the grid"""
        obj._register_grid(self)
        self.objects[name] = obj

    def promote_dtypes_to_complex(self):
        self.E = self.E.astype(bd.complex)
        self.H = self.H.astype(bd.complex)
        [boundary.promote_dtypes_to_complex() for boundary in self.boundaries]

    def __setitem__(self, key, attr):
        if not isinstance(key, tuple):
            x, y, z = key, slice(None), slice(None)
        elif len(key) == 1:
            x, y, z = key[0], slice(None), slice(None)
        elif len(key) == 2:
            x, y, z = key[0], key[1], slice(None)
        elif len(key) == 3:
            x, y, z = key
        else:
            raise KeyError("maximum number of indices for the grid is 3")

        attr._register_grid(
            grid=self,
            x=self._handle_single_key(x, "x"),
            y=self._handle_single_key(y, "y"),
            z=self._handle_single_key(z, "z"),
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(shape=({self.Nx},{self.Ny},{self.Nz}), "
            f"grid_spacing_x={self.grid_spacing_x:.2e}, grid_spacing_y={self.grid_spacing_y:.2e}, "
            f"grid_spacing_z={self.grid_spacing_z:.2e}, courant_number={self.courant_number:.2f})"
        )

    def __str__(self):
        """string representation of the grid

        lists all the components and their locations in the grid.
        """
        s = repr(self) + "\n"
        if self.sources:
            s = s + "\nsources:\n"
            for src in self.sources:
                s += str(src)
        if self.detectors:
            s = s + "\ndetectors:\n"
            for det in self.detectors:
                s += str(det)
        if self.boundaries:
            s = s + "\nboundaries:\n"
            for bnd in self.boundaries:
                s += str(bnd)
        if self.objects:
            s = s + "\nobjects:\n"
            for obj in self.objects:
                s += str(obj)
        return s

    def save_simulation(self, sim_name=None):
        """
        Creates a folder and initializes environment to store simulation or related details.
        saveSimulation() needs to be run before running any function that stores data (generate_video(), save_data()).

        Parameters:-
            (optional) sim_name (string): Preferred name for simulation
        """
        makedirs("fdtd_output", exist_ok=True)  # Output master folder declaration
        # making full_sim_name with timestamp
        full_sim_name = (
                str(datetime.now().year)
                + "-"
                + str(datetime.now().month)
                + "-"
                + str(datetime.now().day)
                + "-"
                + str(datetime.now().hour)
                + "-"
                + str(datetime.now().minute)
                + "-"
                + str(datetime.now().second)
        )
        # Simulation name (optional)
        if sim_name is not None:
            full_sim_name = full_sim_name + " (" + sim_name + ")"
        folder = "fdtd_output_" + full_sim_name
        # storing folder path for saving simulation
        self.folder = os.path.abspath(path.join("fdtd_output", folder))
        # storing timestamp title for self.generate_video
        self.full_sim_name = full_sim_name
        makedirs(self.folder, exist_ok=True)
        return self.folder

    def generate_video(self, delete_frames=False):
        """Compiles frames into a video

        These framed should be saved through ``fdtd.Grid.visualize(save=True)`` while having ``fdtd.Grid.save_simulation()`` enabled.

        Args:
            delete_frames (optional, bool): delete stored frames after conversion to video.

        Returns:
            the filename of the generated video.

        Note:
            this function requires ``ffmpeg`` to be available in your path.
        """
        print(f"self.folder is {self.folder}")
        frame_folder = path.join(self.folder, "frames")
        if frame_folder is None:
            raise Exception(
                "Save location not initialized. Please read about 'fdtd.Grid.saveSimulation()' or try running 'grid.saveSimulation()'."
            )
        cwd = path.abspath(os.getcwd())
        chdir(frame_folder)
        try:
            check_call(
                [
                    "ffmpeg",
                    "-y",
                    "-framerate",
                    "8",
                    "-i",
                    "file%04d.png",
                    "-r",
                    "30",
                    "-pix_fmt",
                    "yuv420p",
                    "fdtd_sim_video_" + self.full_sim_name + ".mp4",
                ]
            )
        except (FileNotFoundError, CalledProcessError):
            raise CalledProcessError(
                "Error when calling ffmpeg. Is ffmpeg installed and available in your path?"
            )
        if delete_frames:  # delete frames
            for file_name in glob("*.png"):
                remove(file_name)
        video_path = path.abspath(
            path.join(frame_folder, f"fdtd_sim_video_{self.full_sim_name}.mp4")
        )
        chdir(cwd)
        return video_path

    def save_data(self):
        """
        Saves readings from all detectors in the grid into a numpy zip file.
        Each detector is stored in separate arrays. Electric and magnetic field field readings of each detector are also stored separately with suffix " (E)" and " (H)" (Example: ['detector0 (E)', 'detector0 (H)']).
        Therefore, the numpy zip file contains arrays twice the number of detectors.
        REQUIRES 'fdtd.Grid.save_simulation()' to be run before this function.

        Parameters: None
        """
        if self.folder is None:
            raise Exception(
                "Save location not initialized. Please read about 'fdtd.Grid.saveSimulation()' or try running 'grid.saveSimulation()'."
            )
        dic = {}
        for detector in self.detectors:
            dic[detector.name + " (E)"] = [x for x in detector.detector_values()["E"]]
            dic[detector.name + " (H)"] = [x for x in detector.detector_values()["H"]]
        savez(path.join(self.folder, "detector_readings"), **dic)

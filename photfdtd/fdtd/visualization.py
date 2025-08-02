""" visualization methods for the fdtd Grid.

This module supplies visualization methods for the FDTD Grid. They are
imported by the Grid class and hence are available as Grid methods.

"""

## Imports
import os
import numpy as np

# plotting
import matplotlib.pyplot as plt
import matplotlib.patches as ptc
from matplotlib.colors import LogNorm

# 3rd party
from tqdm import tqdm
from numpy import log10, where, sqrt, transpose, round
from scipy.signal import hilbert  # TODO: Write hilbert function to replace using scipy

# relative
from .backend import backend as bd
from . import conversions


# 2D visualization function


def visualize(
        grid,
        x=None,
        y=None,
        z=None,
        cmap="Blues",
        pbcolor="C3",
        pmlcolor=(0, 0, 0, 0.1),
        objcolor=(1, 0, 0, 0.1),
        srccolor="C0",
        detcolor="C2",
        norm="linear",
        show=False,  # default False to allow animate to be true
        animate=False,  # True to see frame by frame states of grid while running simulation
        index=None,  # index for each frame of animation (visualize fn runs in a loop, loop variable is passed as index)
        save=False,  # True to save frames (requires parameters index, folder)
        folder=None,  # folder path to save frames
        geo: list = None,
        background_index: float = 1.0,
        show_structure: bool = False,
        show_energy: bool = False,
):
    """visualize a projection of the grid and the optical energy inside the grid

    Args:
        x: the x-value to make the yz-projection (leave None if using different projection)
        y: the y-value to make the zx-projection (leave None if using different projection)
        z: the z-value to make the xy-projection (leave None if using different projection)
        cmap: the colormap to visualize the energy in the grid
        pbcolor: the color to visualize the periodic boundaries
        pmlcolor: the color to visualize the PML
        objcolor: the color to visualize the objects in the grid
        srccolor: the color to visualize the sources in the grid
        detcolor: the color to visualize the detectors in the grid
        norm: how to normalize the grid_energy color map ('linear' or 'log').
        show: call pyplot.show() at the end of the function
        animate: see frame by frame state of grid during simulation
        index: index for each frame of animation (typically a loop variable is passed)
        save: save frames in a folder
        folder: path to folder to save frames
        @param background_index: 背景折射率
        @param geo:solve.geometry。若为None，程序会自动计算
    """
    if norm not in ("linear", "lin", "log"):
        raise ValueError("Color map normalization should be 'linear' or 'log'.")
    # imports (placed here to circumvent circular imports)
    from .sources import PointSource, LineSource, PlaneSource
    from .boundaries import _PeriodicBoundaryX, _PeriodicBoundaryY, _PeriodicBoundaryZ
    from .boundaries import (
        _PMLXlow,
        _PMLXhigh,
        _PMLYlow,
        _PMLYhigh,
        _PMLZlow,
        _PMLZhigh,
    )

    if animate:  # pause for 0.1s, clear plot
        plt.pause(0.02)
        plt.clf()
        plt.ion()  # ionteration on for animation effect

    # validate x, y and z
    if x is not None:
        if not isinstance(x, int):
            raise ValueError("the `x`-location supplied should be a single integer")
        if y is not None or z is not None:
            raise ValueError(
                "if an `x`-location is supplied, one should not supply a `y` or a `z`-location!"
            )
    elif y is not None:
        if not isinstance(y, int):
            raise ValueError("the `y`-location supplied should be a single integer")
        if z is not None or x is not None:
            raise ValueError(
                "if a `y`-location is supplied, one should not supply a `z` or a `x`-location!"
            )
    elif z is not None:
        if not isinstance(z, int):
            raise ValueError("the `z`-location supplied should be a single integer")
        if x is not None or y is not None:
            raise ValueError(
                "if a `z`-location is supplied, one should not supply a `x` or a `y`-location!"
            )
    else:
        raise ValueError(
            "at least one projection plane (x, y or z) should be supplied to visualize the grid!"
        )

    # just to create the right legend entries:
    legend = False
    if legend:
        plt.plot([], lw=7, color=objcolor, label="Objects")
        plt.plot([], lw=7, color=pmlcolor, label="PML")
        plt.plot([], lw=3, color=pbcolor, label="Periodic Boundaries")
        plt.plot([], lw=3, color=srccolor, label="Sources")
        plt.plot([], lw=3, color=detcolor, label="Detectors")

    # Grid energy
    if not show_energy:
        grid_energy = bd.zeros_like(grid.E[:, :, :, -1])
    else:
        grid_energy = bd.sum(grid.E ** 2 + grid.H ** 2, -1)

    if x is not None:
        # x-平面：横轴 = y 方向，纵轴 = z 方向
        x_coords = grid.y_coordinates  # 长度 Ny+1
        y_coords = grid.z_coordinates  # 长度 Nz+1
        xlabel, ylabel = "y/um", "z/um"
        # 提取能量切片并转置，形状 (Nz, Ny)
        grid_energy = grid_energy[x, :, :].T
    elif y is not None:
        # y-平面：横轴 = x 方向，纵轴 = z 方向
        x_coords = grid.x_coordinates
        y_coords = grid.z_coordinates
        xlabel, ylabel = "x/um", "z/um"
        grid_energy = grid_energy[:, y, :].T
    elif z is not None:
        # z-平面：横轴 = x 方向，纵轴 = y 方向
        x_coords = grid.x_coordinates
        y_coords = grid.y_coordinates
        xlabel, ylabel = "x/um", "y/um"
        grid_energy = grid_energy[:, :, z].T
    else:
        raise ValueError("Visualization only works for 2D grids")
    # 计算物理坐标（米→微米）
    x_um = x_coords * 1e6
    y_um = y_coords * 1e6

    cmap_norm = None
    if norm == "log":
        cmap_norm = LogNorm(vmin=1e-4, vmax=np.nanmax(grid_energy) + 1e-4)
    fig, ax = plt.subplots()  # 或者直接 ax = plt.gca() 如果你不想新开fig

    X, Y = np.meshgrid(x_um, y_um)

    # 用 pcolormesh，它会自动按照非均匀格点绘制每个 cell
    mesh = ax.pcolormesh(
        X, Y,
        bd.numpy(grid_energy),
        shading='auto',  # 自动选择最合适的着色方式
        cmap=cmap,
        norm=cmap_norm
    )
    ax.set_aspect('equal', adjustable='box')
    # 锁定范围
    ax.set_xlim(x_um[0], x_um[-1])
    ax.set_ylim(y_um[0], y_um[-1])
    # µm 刻度：每 1 µm 一个
    xticks = np.arange(0, np.ceil(x_um[-1]) + 1, 1)
    yticks = np.arange(0, np.ceil(y_um[-1]) + 1, 1)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels([str(int(v)) for v in xticks])
    ax.set_yticklabels([str(int(v)) for v in yticks])



    for source in grid.sources:
        if isinstance(source, LineSource):
            if x is not None:
                x_phys = np.array([grid.y_coordinates[source.y[0]], grid.y_coordinates[source.y[-1]]])
                y_phys = np.array([grid.z_coordinates[source.z[0]], grid.z_coordinates[source.z[-1]]])
            elif y is not None:
                x_phys = np.array([grid.x_coordinates[source.x[0]], grid.x_coordinates[source.x[-1]]])
                y_phys = np.array([grid.z_coordinates[source.z[0]], grid.z_coordinates[source.z[-1]]])
            else:  # z is not None
                x_phys = np.array([grid.x_coordinates[source.x[0]], grid.x_coordinates[source.x[-1]]])
                y_phys = np.array([grid.y_coordinates[source.y[0]], grid.y_coordinates[source.y[-1]]])
            plt.plot(x_phys * 1e6, y_phys * 1e6, lw=3, color=srccolor)
        elif isinstance(source, PointSource):
            if x is not None:
                x_phys = grid.y_coordinates[source.y]  # 单个索引转物理
                y_phys = grid.z_coordinates[source.z]
            elif y is not None:
                x_phys = grid.x_coordinates[source.x]
                y_phys = grid.z_coordinates[source.z]
            else:
                x_phys = grid.x_coordinates[source.x]
                y_phys = grid.y_coordinates[source.y]
            # 在该物理位置绘制点标记，坐标乘以1e6转换为μm
            plt.plot(x_phys * 1e6, y_phys * 1e6, lw=3, marker="o", color=srccolor)
            # 将源位置处能量置零（按物理坐标需要换算回索引操作，这里保持原索引操作）
            grid_energy[source.y, source.z] = 0
        elif isinstance(source, PlaneSource):
            # 类似地，计算物理起点和尺寸
            if x is not None:
                x_start = grid.y_coordinates[source.y.start]
                x_end = grid.y_coordinates[source.y.stop]
                y_start = grid.z_coordinates[source.z.start]
                y_end = grid.z_coordinates[source.z.stop]
            elif y is not None:
                x_start = grid.x_coordinates[source.x.start]
                x_end = grid.x_coordinates[source.x.stop]
                y_start = grid.z_coordinates[source.z.start]
                y_end = grid.z_coordinates[source.z.stop]
            else:
                x_start = grid.x_coordinates[source.x.start]
                x_end = grid.x_coordinates[source.x.stop]
                y_start = grid.y_coordinates[source.y.start]
                y_end = grid.y_coordinates[source.y.stop]
            patch = ptc.Rectangle(
                xy=(x_start * 1e6, y_start * 1e6),
                width=(x_end - x_start) * 1e6,
                height=(y_end - y_start) * 1e6,
                linewidth=0, edgecolor=srccolor, facecolor=srccolor, alpha=0.3
            )
            plt.gca().add_patch(patch)

    # Detector
    for detector in grid.detectors:
        if x is not None:
            x_phys = np.array([grid.y_coordinates[detector.y[0]], grid.y_coordinates[detector.y[-1]]])
            y_phys = np.array([grid.z_coordinates[detector.z[0]], grid.z_coordinates[detector.z[-1]]])
        elif y is not None:
            x_phys = np.array([grid.x_coordinates[detector.x[0]], grid.x_coordinates[detector.x[-1]]])
            y_phys = np.array([grid.z_coordinates[detector.z[0]], grid.z_coordinates[detector.z[-1]]])
        else:
            x_phys = np.array([grid.x_coordinates[detector.x[0]], grid.x_coordinates[detector.x[-1]]])
            y_phys = np.array([grid.y_coordinates[detector.y[0]], grid.y_coordinates[detector.y[-1]]])
        x_phys *= 1e6
        y_phys *= 1e6
        if detector.__class__.__name__ == "BlockDetector":
            plt.plot([x_phys[0], x_phys[1], x_phys[1], x_phys[0], x_phys[0]],
                     [y_phys[0], y_phys[0], y_phys[1], y_phys[1], y_phys[0]],
                     lw=3, color=detcolor)
        else:
            plt.plot(x_phys, y_phys, lw=3, color=detcolor)

    # --- 绘制边界（PML/周期） ---
    for boundary in grid.boundaries:
        # 仅示例：以x平面为例绘制下边界 pmly low（物理坐标）
        if isinstance(boundary, _PMLZlow) and x is not None:
            # z方向下边界：从 y=0 到 ymax，z 从 0 到 boundary.thickness
            xy = (x_coords[0] * 1e6, y_coords[0] * 1e6)
            patch = ptc.Rectangle(xy=xy, width=(x_coords[-1] - x_coords[0]) * 1e6,
                                  height=(y_coords[boundary.thickness] - y_coords[0]) * 1e6,
                                  linewidth=0, edgecolor='none', facecolor=pmlcolor)
            plt.gca().add_patch(patch)
        # 其它 PML/周期边界同理，转换坐标后绘制线条或矩形（此处省略）

    # --- 绘制结构轮廓 ---
    if show_structure:
        if geo is None:
            geo = sqrt(1 / grid.inverse_permittivity)
        geo = geo[:, :, :, -1]  # 折射率分布
        if x is not None:
            n_plane = geo[x, :, :]
        elif y is not None:
            n_plane = geo[:, y, :]
        else:
            n_plane = geo[:, :, z]
        contour_data = where(n_plane != background_index, 1, 0)
        # 使用物理坐标生成网格进行等高线绘制
        X, Y = np.meshgrid(x_coords[:-1] * 1e6, y_coords[:-1] * 1e6, indexing='xy')
        plt.contour(X, Y, contour_data.T, levels=[0.5], colors='black', linewidths=1)

    # visualize the energy in the grid
    plt.tight_layout()
    # plt.axis("tight")
    # save frame (require folder path and index)
    if save:
        plt.savefig(os.path.join(folder, f"file{str(index).zfill(4)}.png"))

    # show if not animating
    if show:
        plt.show()




def dB_map_2D(block_det=None, interpolation="spline16", axis="z", field="E", field_axis="z", save=True,
              folder="", name_det="", total_time=0):
    """
    Displays detector readings from an 'fdtd.BlockDetector' in a decibel map spanning a 2D slice region inside the BlockDetector.
    Compatible with continuous sources (not waveform).

    Parameter:-
        block_det (numpy array): 5 axes numpy array (timestep, row, column, height, {x, y, z} parameter) created by 'fdtd.BlockDetector'.
        (optional) interpolation (string): Preferred 'matplotlib.pyplot.imshow' interpolation. Default "spline16".
        @param axis: 选择截面"x" or "y" or "z"
        @param save: 是否保存
        @param folder: 存储文件夹
        @param name_det: 面监视器的名称
        @param total_time: 总模拟时间，可选，仅为了命名
    """
    # if block_det is None:
    #     raise ValueError(
    #         "Function 'dBmap' requires a detector_readings object as parameter."
    #     )
    # if len(block_det.shape) != 5:  # BlockDetector readings object have 5 axes
    #     raise ValueError(
    #         "Function 'dBmap' requires object of readings recorded by 'fdtd.BlockDetector'."
    #     )

    # TODO: convert all 2D slices (y-z, x-z plots) into x-y plot data structure
    plt.ioff()
    plt.close()
    a = []  # array to store wave intensities
    # 首先计算仿真空间上每一点在所有时间上的最大值与最小值之差

    if not axis:
        # Tell which dimension to draw automatically
        shape = block_det.shape
        dims_with_size_one = [i for i, size in enumerate(shape[1:], start=1) if size == 1]
        axis_number = dims_with_size_one[0]
        axis = conversions.number_to_letter(axis_number)

    choose_axis = conversions.letter_to_number(field_axis)

    if axis == "z":
        for i in tqdm(range(len(block_det[0]))):
            a.append([])
            for j in range(len(block_det[0][0])):
                temp = [x[i][j][0][choose_axis] for x in block_det]
                a[i].append(max(temp) - min(temp))
    elif axis == "x":
        for i in tqdm(range(len(block_det[0][0]))):
            a.append([])
            for j in range(len(block_det[0][0][0])):
                temp = [x[0][i][j][choose_axis] for x in block_det]
                a[i].append(max(temp) - min(temp))
    elif axis == "y":
        for i in tqdm(range(len(block_det[0]))):
            a.append([])
            for j in range(len(block_det[0][0][0])):
                temp = [x[i][0][j][choose_axis] for x in block_det]
                a[i].append(max(temp) - min(temp))

    peakVal, minVal = max(map(max, a)), min(map(min, a))
    # print(
    #     "Peak at:",
    #     [
    #         [[i, j] for j, y in enumerate(x) if y == peakVal]
    #         for i, x in enumerate(a)
    #         if peakVal in x
    #     ],
    # )
    # 然后做对数计算
    if minVal == 0:
        raise RuntimeError("minVal == 0, impossible to draw a dB map")
    a = 10 * log10([[y / minVal for y in x] for x in a])

    # plt.title("dB map of Electrical waves in detector region")
    plt.imshow(transpose(a), cmap="inferno", interpolation=interpolation)
    plt.ylim(-1, a.shape[1])
    if axis == "z":
        plt.xlabel('X/grids')
        plt.ylabel('Y/grids')
    elif axis == "x":
        plt.xlabel('Y/grids')
        plt.ylabel('Z/grids')
    elif axis == "y":
        plt.xlabel('X/grids')
        plt.ylabel('Z/grids')
    cbar = plt.colorbar()
    # cbar.ax.set_ylabel("dB scale", rotation=270)
    # plt.show()
    if save:
        fieldaxis = field + field_axis
        plt.savefig(fname='%s//dB_map_%s, detector_name=%s, time=%i.png' % (folder, fieldaxis, name_det, total_time))
    plt.close()


def plot_detection(detector_dict=None, specific_plot=None):
    """
    1. Plots intensity readings on array of 'fdtd.LineDetector' as a function of timestep.
    2. Plots time of arrival of waveform at different LineDetector in array.
    Compatible with waveform sources.

    Args:
        detector_dict (dictionary): Dictionary of detector readings, as created by 'fdtd.Grid.save_data()'.
        (optional) specific_plot (string): Plot for a specific axis data. Choose from {"Ex", "Ey", "Ez", "Hx", "Hy", "Hz"}.
    """
    if detector_dict is None:
        raise Exception(
            "Function plotDetection() requires a dictionary of detector readings as 'detector_dict' parameter."
        )
    detectorElement = 0  # cell to consider in each detectors
    maxArray = {}
    plt.ioff()
    plt.close()

    for detector in detector_dict:
        if len(detector_dict[detector].shape) != 3:
            print("Detector '{}' not LineDetector; dumped.".format(detector))
            continue
        if specific_plot is not None:
            if detector[-2] != specific_plot[0]:
                continue
        if detector[-2] == "E":
            plt.figure(0, figsize=(15, 15))
        elif detector[-2] == "H":
            plt.figure(1, figsize=(15, 15))
        for dimension in range(len(detector_dict[detector][0][0])):
            if specific_plot is not None:
                if ["x", "y", "z"].index(specific_plot[1]) != dimension:
                    continue
            # if specific_plot, subplot on 1x1, else subplot on 2x2
            plt.subplot(
                2 - int(specific_plot is not None),
                2 - int(specific_plot is not None),
                dimension + 1 if specific_plot is None else 1,
            )
            hilbertPlot = abs(
                hilbert([x[0][dimension] for x in detector_dict[detector]])
            )
            plt.plot(hilbertPlot, label=detector)
            plt.title(detector[-2] + "(" + ["x", "y", "z"][dimension] + ")")
            if detector[-2] not in maxArray:
                maxArray[detector[-2]] = {}
            if str(dimension) not in maxArray[detector[-2]]:
                maxArray[detector[-2]][str(dimension)] = []
            maxArray[detector[-2]][str(dimension)].append(
                [detector, where(hilbertPlot == max(hilbertPlot))[0][0]]
            )

    # Loop same as above, only to add axes labels
    for i in range(2):
        if specific_plot is not None:
            if ["E", "H"][i] != specific_plot[0]:
                continue
        plt.figure(i)
        for dimension in range(len(detector_dict[detector][0][0])):
            if specific_plot is not None:
                if ["x", "y", "z"].index(specific_plot[1]) != dimension:
                    continue
            plt.subplot(
                2 - int(specific_plot is not None),
                2 - int(specific_plot is not None),
                dimension + 1 if specific_plot is None else 1,
            )
            plt.xlabel("Time steps")
            plt.ylabel("Magnitude")
        plt.suptitle("Intensity profile")
    plt.legend()
    plt.show()

    for item in maxArray:
        plt.figure(figsize=(15, 15))
        for dimension in maxArray[item]:
            arrival = bd.numpy(maxArray[item][dimension])
            plt.plot(
                [int(x) for x in arrival.T[1]],
                arrival.T[0],
                label=["x", "y", "z"][int(dimension)],
            )
        plt.title(item)
        plt.xlabel("Time of arrival (time steps)")
        plt.legend()
        plt.suptitle("Time-of-arrival plot")
    plt.show()

#
# def dump_to_vtk(pcb, filename, iteration, Ex_dump=False, Ey_dump=False, Ez_dump=False, Emag_dump=True, objects_dump=True, ports_dump=True):
#     '''
#     Extension is automatically chosen, you don't need to supply it
#
#     thanks
#     https://pyscience.wordpress.com/2014/09/06/numpy-to-vtk-converting-your-numpy-arrays-to-vtk-arrays-and-files/
#     https://bitbucket.org/pauloh/pyevtk/src/default/src/hl.py
#
#     Paraview needs a threshold operation to view the objects correctly.
#
#
#     argument scaling_factor=None,True(epsilon and hbar), or Float, to accomodate reduced units
#     or maybe
#     '''

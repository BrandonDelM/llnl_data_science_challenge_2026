import numpy as np
import pyvista as pv
import tifffile

AXIS_MAP = {'x': 0, 'y': 1, 'z': 2}
NORMAL_MAP = {'x': (1, 0, 0), 'y': (0, 1, 0), 'z': (0, 0, 1)}
VIEW_MAP = {'x': 'yz', 'y': 'xz', 'z': 'xy'}  # looking down the sliced axis


def get_fractional_slice(mesh, fraction: float, axis: str = 'y'):
    """Slice a mesh at a given fraction (0.0-1.0) along its own bounding box extent."""
    axis_idx = AXIS_MAP[axis]
    normal = NORMAL_MAP[axis]

    lo = mesh.bounds[axis_idx * 2]
    hi = mesh.bounds[axis_idx * 2 + 1]
    position = lo + fraction * (hi - lo)

    origin = list(mesh.center)
    origin[axis_idx] = position

    return mesh.slice(normal=normal, origin=origin), position


def build_tif_grid(volume: np.ndarray, density_threshold: float = 43400):
    """Convert a raw TIF volume into a thresholded, centered PyVista mesh."""
    volume = volume.copy()
    volume[volume < density_threshold] = 0

    grid = pv.ImageData()
    grid.dimensions = np.array(volume.shape) + 1
    grid.spacing = (1, 1, 1)
    grid.origin = (0, 0, 0)
    grid.cell_data['density'] = volume.flatten(order='F')

    surface = grid.threshold(volume.max() * 0.5)
    surface.translate(-np.array(surface.center), inplace=True)
    return surface


def compare_at_fraction(tif_surface, stl_mesh, fraction: float, axis: str = 'y', thickness: float = None):
    """
    Slice both the TIF surface and STL at the same normalized fraction along the given axis,
    then plot: TIF slice | STL slice | overlay of both — all rendered with PyVista.
    """
    view = VIEW_MAP[axis]

    if thickness:
        # Optional: use a thin slab instead of an infinite-plane slice, for solid-looking cross-sections
        axis_idx = AXIS_MAP[axis]
        lo, hi = tif_surface.bounds[axis_idx*2], tif_surface.bounds[axis_idx*2+1]
        pos = lo + fraction * (hi - lo)
        b = list(tif_surface.bounds)
        b[axis_idx*2], b[axis_idx*2+1] = pos - thickness/2, pos + thickness/2
        tif_slice = tif_surface.clip_box(b, invert=False)

        lo, hi = stl_mesh.bounds[axis_idx*2], stl_mesh.bounds[axis_idx*2+1]
        pos = lo + fraction * (hi - lo)
        b = list(stl_mesh.bounds)
        b[axis_idx*2], b[axis_idx*2+1] = pos - thickness/2, pos + thickness/2
        stl_slice = stl_mesh.clip_box(b, invert=False)
    else:
        tif_slice, tif_pos = get_fractional_slice(tif_surface, fraction, axis=axis)
        stl_slice, stl_pos = get_fractional_slice(stl_mesh, fraction, axis=axis)
        print(f"fraction={fraction:.3f}  TIF position={tif_pos:.2f}  STL position={stl_pos:.2f}")

    pl = pv.Plotter(shape=(1, 3), window_size=(1800, 600))

    pl.subplot(0, 0)
    pl.add_mesh(tif_slice, color='red')
    pl.add_text(f'TIF slice (frac={fraction:.2f})', font_size=10)
    pl.enable_parallel_projection()
    getattr(pl, f'view_{view}')()

    pl.subplot(0, 1)
    pl.add_mesh(stl_slice, color='silver')
    pl.add_text(f'STL slice (frac={fraction:.2f})', font_size=10)
    pl.enable_parallel_projection()
    getattr(pl, f'view_{view}')()

    pl.subplot(0, 2)
    pl.add_mesh(tif_slice, color='red', opacity=0.5, label='TIF')
    pl.add_mesh(stl_slice, color='silver', opacity=0.5, label='STL')
    pl.add_legend()
    pl.add_text('Overlay', font_size=10)
    pl.enable_parallel_projection()
    getattr(pl, f'view_{view}')()

    pl.link_views()
    pl.set_background('white', all_renderers=True)
    pl.show(screenshot=f'slices/comparison_frac_{fraction:.2f}.png')


# ---------- Usage ----------
volume = tifffile.imread('test/json_cropped_aligned_tif.tif')
tif_surface = build_tif_grid(volume, density_threshold=43400)

stl_mesh = pv.read('test/json_cropped_aligned_stl.stl')   # already rotated/aligned
stl_mesh.translate(-np.array(stl_mesh.center), inplace=True)

print("TIF surface bounds:", tif_surface.bounds)
print("STL bounds:", stl_mesh.bounds)

compare_at_fraction(tif_surface, stl_mesh, fraction=0.5, axis='y', thickness=2.0)
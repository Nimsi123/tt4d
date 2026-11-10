import numpy as np
import yaml


# --- helper: build a 4x4 transform from position + desired normal ---
def target_frame_from_normal(p_xyz, n_world, normal_axis_index=0):
    """
    normal_axis_index: which EE axis should align with the normal (0=x,1=y,2=z).
    Returns a 4x4 transform with R's specified axis == normalized(n_world).
    """
    n = np.array(n_world, dtype=float)
    n /= np.linalg.norm(n) + 1e-12

    # make an orthonormal basis with z:=n, then permute columns
    a = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = np.cross(a, n)
    x /= np.linalg.norm(x) + 1e-12
    y = np.cross(n, x)  # already unit

    # base rotation with columns [x, y, z=n]
    R_base = np.column_stack([x, y, n])

    # put n into the requested axis (cyclic perms keep det +1)
    if normal_axis_index == 2:  # z-axis is normal
        R = R_base
    elif normal_axis_index == 0:  # x-axis is normal
        R = np.column_stack([n, x, y])  # equivalent to [z, x, y] w/ n=z
    elif normal_axis_index == 1:  # y-axis is normal
        R = np.column_stack([y, n, x])  # equivalent to [y, z, x]
    else:
        raise ValueError("normal_axis_index must be 0, 1, or 2")

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.array(p_xyz, dtype=float)
    return T


def to_homogeneous(T_3x4):
    """Convert a 3x4 matrix to 4x4 homogeneous transformation matrix."""
    T_hom = np.eye(4)
    T_hom[:3, :] = T_3x4
    return T_hom


def load_pose_from_yaml(filepath):
    with open(filepath) as f:
        data = yaml.safe_load(f)

    # Assumes the YAML contains one top-level key
    key = next(iter(data))
    entry = data[key]
    rows = entry["rows"]
    cols = entry["cols"]
    matrix = np.array(entry["data"]).reshape(rows, cols)

    return matrix


def draw_table_tennis_table(ax):
    # Table dimensions
    length = 2.74  # meters (length along y-axis)
    width = 1.525  # meters (width along x-axis)
    height = 0  # 0.76  # meters
    net_height = 0.1525  # meters

    # Create table surface
    x = np.array([-width / 2, width / 2])
    y = np.array([-length / 2, length / 2])
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    # Draw the top surface of the table
    ax.plot_surface(X, Y, Z + height, color="green", alpha=0.8, zorder=0)

    # Create net surface
    net_x = np.array([-width / 2, width / 2])
    net_z = np.array([height, height + net_height])
    net_X, net_Z = np.meshgrid(net_x, net_z)
    net_Y = np.zeros_like(net_X)

    # Draw the net
    ax.plot_surface(net_X, net_Y, net_Z, color="gray", alpha=0.8, zorder=0.5)

    # Add table edges
    edges_x = [-width / 2, width / 2, width / 2, -width / 2, -width / 2]
    edges_y = [-length / 2, -length / 2, length / 2, length / 2, -length / 2]
    edges_z = [height, height, height, height, height]

    ax.plot(edges_x, edges_y, edges_z, color="black", zorder=0)


def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

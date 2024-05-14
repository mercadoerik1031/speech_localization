import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import spaudiopy as spa
import warnings
from config import config
import random

def get_grid(degree):
    """
    Returns the cartesian coordinates for a t_design.
    This represents a grid of directions, that uniformly sample the unit sphere.
    """
    # degree = int(degree)
    t_design = spa.grids.load_t_design(degree=degree)
    return t_design


def vecs2dirs(vecs, positive_azi=True, include_r=False, use_elevation=False):
    """
    Helper to convert [x, y, z] to [azi, colat].
    """
    azi, colat, r = spa.utils.cart2sph(vecs[:, 0], vecs[:, 1], vecs[:, 2], steady_zen=True)
    
    if positive_azi:
        azi = np.mod(azi + 2 * np.pi, 2 * np.pi)
        
    if use_elevation:
        colat = np.pi / 2 - colat
        
    if include_r:
        output = np.c_[azi, colat, r]
    else:
        output = np.c_[azi, colat]
        
    return output

def sph2unit_vec(azimuth, elevation):
    """ Transforms spherical coordinates into a unit vector.
    Equation 2.1 of:
    [1] M. Kronlachner, “Spatial transformations for the alteration of ambisonic recordings”.
    """
    # Ensure inputs are numpy arrays for vectorized operations
    azimuth = np.array(azimuth)
    elevation = np.array(elevation)

    # Check that azimuth is within the valid range
    assert np.all(azimuth >= 0) and np.all(azimuth <= 2 * np.pi), 'Azimuth should be in radians, between 0 and 2*pi'

    x = np.cos(azimuth) * np.cos(elevation)
    y = np.sin(azimuth) * np.cos(elevation)
    z = np.sin(elevation)

    # Stack results into a single numpy array
    return np.stack([x, y, z], axis=-1)

def draw_random_spherical_cap(spherical_cap_type='soft'):
    """ Draws the parameters for a random spherical cap using:
    spherical_cap_type = 'hard':
        - cap_center = Uniform between [0, 2*pi] for azimuth, and [-pi/2, pi/2]
        - cap_width = Uniform between [pi/4 and pi]
        - g1 = 0  # Assuming the exponential part is commented out
        - g2 = Uniform [-20, -6]

    spherical_cap_type = 'soft':
        - cap_center = Uniform between [0, 2*pi] for azimuth, and [-pi/2, pi/2]
        - cap_width = Uniform between [pi/4 and pi]
        - g1 = 0  # Assuming the exponential part is commented out
        - g2 = Exponential with high = -3, low = -6
    """

    cap_center = np.array([[np.random.uniform(0, 2 * np.pi),
                        np.random.uniform(-np.pi / 2, np.pi / 2)]])  # Now a 2D array with one row

    cap_width = np.random.uniform(np.pi / 4, np.pi)
    g1 = 0  # Since exponential part is commented out

    if spherical_cap_type == 'hard':
        g2 = np.random.uniform(-20, -6)
    elif spherical_cap_type == 'soft':
        # Sample g2 from an exponential distribution
        # The exponential function does not directly support custom bounds, so we simulate this
        scale = 1/3  # Scale for exponential distribution, derived from rate = 1/scale
        g2 = -np.random.exponential(scale) - 6
    else:
        raise ValueError(f'Unsupported spherical cap type: {spherical_cap_type}')

    return cap_center, cap_width, g1, g2

def compute_Y_and_W(grid, rotation_matrix=None, order_input=1, 
                    order_output=1, backend='basic', w_pattern='hypercardioid'):
    """
    Computes the reconstruction matrix Y, and beamforming matrix W using only numpy arrays.

    Args:
        grid (np.ndarray): Grid of points in Cartesian coordinates.
        rotation_matrix (np.ndarray, optional): Rotation matrix to apply to the grid.
        order_input (int): Order of the input spherical harmonics.
        order_output (int): Order of the output spherical harmonics.
        backend (str): Backend to use ('basic' or 'spatial_filterbank').
        w_pattern (str): Type of beamforming pattern ('cardioid', 'hypercardioid', 'maxre').

    Returns:
        tuple: A tuple containing Y and W matrices.
    """
    # Convert grid to spherical directions
    tmp_directions = vecs2dirs(grid)
    
    # Apply rotation if provided
    if rotation_matrix is not None:
        # Apply rotation matrix to grid and convert to directions
        tmp_directions_rotated = vecs2dirs(np.dot(grid, rotation_matrix))
    else:
        tmp_directions_rotated = tmp_directions

    # Compute spherical harmonics matrices based on backend
    if backend == 'basic':
        Y = spa.sph.sh_matrix(order_input, tmp_directions[:, 0], tmp_directions[:, 1], sh_type='real')
        W = spa.sph.sh_matrix(order_output, tmp_directions_rotated[:, 0], tmp_directions_rotated[:, 1], sh_type='real')
    elif backend == 'spatial_filterbank':
        # Ensure the same order for input and output when using spatial filterbank
        assert order_input == order_output, 'Input and output orders must be the same for spatial filterbank.'
        # Check for rotation compatibility
        if rotation_matrix is not None and not np.allclose(rotation_matrix, np.eye(3)):
            raise ValueError('Soundfield rotations not supported when using spatial filterbank')

        # Select beamforming weights based on the specified pattern
        if w_pattern.lower() == 'cardioid':
            c_n = spa.sph.cardioid_modal_weights(order_output)
        elif w_pattern.lower() == 'hypercardioid':
            c_n = spa.sph.hypercardioid_modal_weights(order_output)
        elif w_pattern.lower() == 'maxre':
            c_n = spa.sph.maxre_modal_weights(order_output, True)  # Amplitude compensation
        else:
            raise ValueError(f'Unknown w_pattern type: {w_pattern}. Check spelling?')

        Y, W = spa.sph.design_spat_filterbank(order_output, tmp_directions[:, 0], tmp_directions[:, 1], c_n, 'real', 'perfect')
    else:
        raise ValueError(f'Unknown backend: {backend}. Should be either "basic" or "spatial_filterbank"')

    # Ensure double precision for output matrices
    Y = Y.astype(np.double)
    W = W.astype(np.double)

    return Y, W



sr_params = config["data_augmentation"]["spherical_rotation_params"]


class SphericalRotation:
    def __init__(self,
                 rotation_angles_rad=sr_params["rotation_angles_rad"],
                 mode=sr_params["mode"],
                 num_random_rotations=sr_params["num_random_rotations"],
                 t_design_degree=sr_params["t_design_degree"],
                 order_input=sr_params["order_input"],
                 order_output=sr_params["order_output"],
                 backend=sr_params["backend"],
                 w_pattern=sr_params["w_pattern"],
                 plot_on_init=sr_params["save_plots"]):
        self.rotation_angles_rad = rotation_angles_rad
        self.mode = mode
        self.backend = backend
        self.w_pattern = w_pattern
        self.grid = get_grid(t_design_degree)
        self.n_directions = self.grid.shape[0]
        self.order_input = order_input
        self.order_output = order_output
        
        # Initialize rotation matrices
        if self.mode == "single":
            self.rotations = [self.get_rotation_matrix(*self.rotation_angles_rad)]
            self.rotation_indices = [0]  # Always use the first matrix
        else:
            self.rotations = [self.get_rotation_matrix(np.random.uniform(0, 2 * np.pi),
                                                       np.random.uniform(-np.pi / 2, np.pi / 2),
                                                       np.random.uniform(0, 2 * np.pi))
                              for _ in range(num_random_rotations)]
            self.rotation_indices = random.sample(range(num_random_rotations), num_random_rotations)  # List of indices to use randomly

        # Compute transformation matrices
        self.T_mats = []
        for R in self.rotations:
            Y, W = compute_Y_and_W(self.grid, R, self.order_input, self.order_output, self.backend, self.w_pattern)
            T_mat = np.matmul(Y.T, W)
            if self.backend == 'basic':
                scale = 4 * np.pi / self.n_directions
                T_mat *= scale
            self.T_mats.append(T_mat)

        if plot_on_init:
            self.plot_grids()

    def forward(self, x):
        # Select a transformation matrix
        if self.mode == "random":
            T_mat = self.T_mats[random.choice(self.rotation_indices)]
        else:
            T_mat = self.T_mats[0]

        assert x.shape[-2] == T_mat.shape[0], "ERROR: The order of the input signal does not match the rotation matrix"
        return np.matmul(T_mat, x)

    def get_rotation_matrix(self, rotation_phi, rotation_theta, rotation_psi):
        roll = np.array([[1, 0, 0],
                         [0, np.cos(rotation_phi), -np.sin(rotation_phi)],
                         [0, np.sin(rotation_phi), np.cos(rotation_phi)]])
        pitch = np.array([[np.cos(rotation_theta), 0, np.sin(rotation_theta)],
                          [0, 1, 0],
                          [-np.sin(rotation_theta), 0, np.cos(rotation_theta)]])
        yaw = np.array([[np.cos(rotation_psi), -np.sin(rotation_psi), 0],
                        [np.sin(rotation_psi), np.cos(rotation_psi), 0],
                        [0, 0, 1]])
        return np.matmul(np.matmul(yaw, pitch), roll)

    def plot_grids(self):
        self.plot_sphere(self.grid, "Original Points", "sr_original_points.png")
        for i, R in enumerate(self.rotations):
            rotated_points = np.dot(self.grid, R.T)
            self.plot_sphere(rotated_points, f"Rotated Points Rotation {i+1}", f"sr_rotated_points_{i+1}.png")

    def plot_sphere(self, points, title, filename):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='r', marker='o')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        plt.savefig(filename)
        plt.close(fig)

dl_params = config["data_augmentation"]["directional_loudness_params"]

class DirectionalLoudness:
    def __init__(self, 
                 order_input=dl_params["order_input"],
                 t_design_degree=dl_params["t_design_degree"],
                 order_output=dl_params["order_output"], 
                 g_type=dl_params["g_type"], 
                 g_values=dl_params["g_values"],
                 T_pseudo_floor=dl_params["T_pseudo_floor"], 
                 backend=dl_params["backend"], 
                 w_pattern=dl_params["w_pattern"],
                 use_slepian=dl_params["use_slepian"],
                 save_plots=dl_params["save_plots"]):
        
        assert t_design_degree > 2 * order_output, "ERROR: t_design_degree must be greater than 2 * order_output"
        
        self._Y = None
        self._G = None
        self._W = None
        self.T_mat = None
        
        self.grid = get_grid(t_design_degree)
        self.n_directions = self.grid.shape[0]
        self.order_input = order_input
        self.order_output = order_output
        self.g_type = g_type
        self.backend = backend
        self.w_pattern = w_pattern
        self.use_slepian = use_slepian
        
        self.G_cap_center = None
        self.G_cap_width = None
        self.G_g1 = None
        self.G_g2 = None
        
        Y, W = compute_Y_and_W(self.grid, None, 
                               self.order_input, self.order_output, 
                               self.backend, self.w_pattern)
        
        self.Y, self.W = Y, W
        self.G = self.compute_G(self.g_type, g_values)
        self.T_mat = self.compute_T_mat()
        
        if save_plots:
            self.plot_grids()
        
    def compute_T_mat(self):
        """ 
        Computes the full transformation matrix T_mat, and applies the scaling if selected.
        """
            
        # Matrix multiplications using NumPy
        tmp = np.dot(self.Y.T, self.G)
        T_mat = np.dot(tmp, self.W)

        # Apply scaling if backend is 'basic'
        if self.backend == 'basic':
            scale = 4 * np.pi / self.n_directions
            T_mat = scale * T_mat

        # Optional Slepian function application
        if self.use_slepian:
            if 'spherical_cap' in self.g_type:
                T_mat = self.apply_slepian_function(T_mat, self.G_g1, self.G_g2)
            else:
                print('WARNING: Slepian Functions are only applied when using spherical caps.')
        
        return T_mat
    
    def apply_slepian_function(self, T_mat, g1_db=0, g2_db=-10, alpha=0.5):
        """ Applies a spherical Slepian function to the transformation matrix.
        This is mostly useful when the directional matrix G is not identity, as it helps when using spherical
        harmonics in incomplete spheres.
        """
        # Convert dB to linear scale using utility function assumed to be adapted for NumPy
        g1, g2 = spa.utils.from_db(g1_db), spa.utils.from_db(g2_db)

        # Perform SVD decomposition using NumPy
        U, eig, Vh = np.linalg.svd(T_mat, full_matrices=False)
        largest_eig = np.max(eig)

        # Apply the Slepian function to the eigenvalues
        values = g1 * np.heaviside(eig - alpha * largest_eig, g1) + \
                g2 * np.heaviside(alpha * largest_eig - eig, g2)

        # Create a new diagonal matrix for G
        new_G = np.zeros((values.size, values.size))
        np.fill_diagonal(new_G, values)

        # Reconstruct the new T matrix
        new_T_mat = np.dot(U, np.dot(new_G, Vh))

        return new_T_mat
        
    def compute_G(self, G_type='identity', G_values=None, capsule_center=None, 
              capsule_width=np.pi/2, g1_db=0, g2_db=-10):
        """
        Returns a matrix G with the gains for each direction.
        """
        G = np.eye(self.n_directions)  # Default identity matrix

        if G_type == 'identity':
            pass

        elif G_type == 'random_diag':
            values = np.random.rand(self.n_directions)
            np.fill_diagonal(G, values)

        elif G_type == 'fixed':
            np.fill_diagonal(G, G_values)

        elif G_type == 'random':
            G = np.random.rand(self.n_directions, self.n_directions)

        elif 'spherical_cap' in G_type:
            if capsule_center is None:
                cap_type = 'soft' if 'soft' in G_type else 'hard'
                capsule_center, capsule_width, g1_db, g2_db = draw_random_spherical_cap(spherical_cap_type=cap_type)

            assert capsule_center.shape[-1] == 2, 'Capsule center should be a [1, 2] vector of azimuth and elevation.'
            assert 0 < capsule_width <= np.pi, 'Capsule width should be within (0, π] radians.'

            # Convert capsule center to a unit vector
            tmpA = sph2unit_vec(capsule_center[:, 0], capsule_center[:, 1])
            tmpB = self.grid
            tmpA = np.repeat(tmpA, tmpB.shape[0], axis=0)

            # Dot product between the angles of the capsule and the grid points
            tmp = np.einsum('ij,ij->i', tmpA, tmpB)
            g1 = spa.utils.from_db(g1_db)
            g2 = spa.utils.from_db(g2_db)

            cos_angle = np.cos(capsule_width / 2)
            values = g1 * (tmp >= cos_angle) + g2 * (tmp < cos_angle)
            np.fill_diagonal(G, values)  # Set the diagonal values

            # Store parameters if needed for later use
            self.G_cap_center = capsule_center
            self.G_cap_width = capsule_width
            self.G_g1 = g1_db
            self.G_g2 = g2_db

        return G

    def forward(self, X):
        """
        Processes the input matrix X using the transformation matrix T_mat
        and potentially warns if formats do not match expected dimensions.
        """

        # Ensure X is a NumPy array
        if not isinstance(X, np.ndarray):
            X = np.asarray(X)

        # Check for channel format and warn if it seems not in channels-first format
        if X.shape[-2] > X.shape[-1]:
            warnings.warn('WARNING: It seems that the input array X is NOT in channels-first format')

        # Ensure that the matrix dimensions are aligned for multiplication
        if self.W is not None and self.Y is not None:
            assert X.shape[-2] == self.W.shape[-1], 'Wrong shape for input signal or matrix W.'
        assert self.T_mat.shape[-1] == X.shape[-2], 'Wrong shape for input signal or matrix T.'

        # Perform matrix multiplication
        out = np.dot(self.T_mat, X)

        return out
    
    def plot_grids(self):
        original_points = self.grid
        transformed_points = np.dot(self.grid, self.T_mat[:3, :3].T)  # Apply transformation to the first 3 components
        self.plot_3d(original_points, "Original Grid", "dl_original_grid.png")
        self.plot_3d(transformed_points, "Transformed Grid", "dl_transformed_grid.png")

    def plot_3d(self, data, title="3D Plot", filename="3d_plot.png"):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='r', marker='o')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        ax.view_init(elev=20, azim=120)  # Adjust viewing angle
        plt.savefig(filename)
        plt.close(fig)
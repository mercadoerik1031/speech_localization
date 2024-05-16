import numpy as np
import spaudiopy as spa
import warnings

def get_grid(degree):
    """
    Returns the cartesian coordinates for a t_design.
    This represents a grid of directions, that uniformly sample the unit sphere.
    """
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
    """ Transforms spherical coordinates into a unit vector. """
    azimuth = np.array(azimuth)
    elevation = np.array(elevation)

    x = np.cos(azimuth) * np.cos(elevation)
    y = np.sin(azimuth) * np.cos(elevation)
    z = np.sin(elevation)

    return np.stack([x, y, z], axis=-1)

def draw_random_spherical_cap():
    """ Draws the parameters for a random soft spherical cap. """
    cap_center = np.array([[np.random.uniform(0, 2 * np.pi),
                            np.random.uniform(-np.pi / 2, np.pi / 2)]])
    cap_width = np.random.uniform(np.pi / 4, np.pi)
    g1 = 0  # Assuming the exponential part is commented out
    g2 = -np.random.exponential(1/3) - 6

    return cap_center, cap_width, g1, g2

def compute_Y_and_W(grid, order_input=1, order_output=1):
    """
    Computes the reconstruction matrix Y, and beamforming matrix W using only numpy arrays.
    """
    tmp_directions = vecs2dirs(grid)
    Y = spa.sph.sh_matrix(order_input, tmp_directions[:, 0], tmp_directions[:, 1], sh_type='real')
    W = spa.sph.sh_matrix(order_output, tmp_directions[:, 0], tmp_directions[:, 1], sh_type='real')

    Y = Y.astype(np.double)
    W = W.astype(np.double)

    return Y, W

class DirectionalLoudness:
    def __init__(self, order_input=1, t_design_degree=3, order_output=1):
        assert t_design_degree > 2 * order_output, "ERROR: t_design_degree must be greater than 2 * order_output"
        
        self.grid = get_grid(t_design_degree)
        self.n_directions = self.grid.shape[0]
        self.order_input = order_input
        self.order_output = order_output
        
        self.Y, self.W = compute_Y_and_W(self.grid, self.order_input, self.order_output)
        self.G = self.compute_G()
        self.T_mat = self.compute_T_mat()

    def compute_T_mat(self):
        """ Computes the full transformation matrix T_mat. """
        tmp = np.dot(self.Y.T, self.G)
        T_mat = np.dot(tmp, self.W)
        scale = 4 * np.pi / self.n_directions
        T_mat = scale * T_mat
        
        return T_mat
    
    def compute_G(self):
        """ Returns a matrix G with the gains for each direction using a soft spherical cap. """
        cap_center, cap_width, g1_db, g2_db = draw_random_spherical_cap()
        tmpA = sph2unit_vec(cap_center[:, 0], cap_center[:, 1])
        tmpB = self.grid
        tmpA = np.repeat(tmpA, tmpB.shape[0], axis=0)

        tmp = np.einsum('ij,ij->i', tmpA, tmpB)
        g1 = spa.utils.from_db(g1_db)
        g2 = spa.utils.from_db(g2_db)

        cos_angle = np.cos(cap_width / 2)
        values = g1 * (tmp >= cos_angle) + g2 * (tmp < cos_angle)
        G = np.diag(values)  # Set the diagonal values

        return G

    def forward(self, X):
        """
        Processes the input matrix X using the transformation matrix T_mat.
        """
        if not isinstance(X, np.ndarray):
            X = np.asarray(X)

        if X.shape[-2] > X.shape[-1]:
            warnings.warn('WARNING: It seems that the input array X is NOT in channels-first format')

        assert X.shape[-2] == self.W.shape[-1], 'Wrong shape for input signal or matrix W.'
        assert self.T_mat.shape[-1] == X.shape[-2], 'Wrong shape for input signal or matrix T.'

        out = np.dot(self.T_mat, X)

        return out

    def transform_labels(self, azimuth, elevation):
        """ Transforms azimuth and elevation labels using the transformation matrix T_mat. """
        cartesian_coords = sph2unit_vec(azimuth, elevation)
        
        # Ensure cartesian_coords is 2D
        if cartesian_coords.ndim == 1:
            cartesian_coords = cartesian_coords[np.newaxis, :]
        
        # Convert to homogeneous coordinates
        homogeneous_coords = np.hstack([cartesian_coords, np.ones((cartesian_coords.shape[0], 1))])
        
        # Apply the transformation matrix
        transformed_homogeneous = np.dot(self.T_mat, homogeneous_coords.T).T
        
        # Convert back to Cartesian coordinates by removing the homogeneous coordinate
        transformed_cartesian = transformed_homogeneous[:, :3]
        
        transformed_azi, transformed_colat, _ = spa.utils.cart2sph(transformed_cartesian[:, 0], 
                                                                   transformed_cartesian[:, 1], 
                                                                   transformed_cartesian[:, 2], 
                                                                   steady_zen=True)
        transformed_elev = np.pi / 2 - transformed_colat
        
        return transformed_azi, transformed_elev

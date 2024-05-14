import torch

def to_cartesian(azimuth, elevation):
    """Convert azimuth and elevation angles to Cartesian coordinates."""
    x = torch.cos(elevation) * torch.cos(azimuth)
    y = torch.cos(elevation) * torch.sin(azimuth)
    z = torch.sin(elevation)
    return torch.stack((x, y, z), dim=-1)




def angular_distance(v1, v2):
    """Calculate the angular distance between two vectors."""
    # Ensure the vectors are normalized
    v1_norm = v1 / (torch.linalg.norm(v1, dim=-1, keepdim=True) + 1e-6)
    v2_norm = v2 / (torch.linalg.norm(v2, dim=-1, keepdim=True) + 1e-6)
    # Use dot product to find the cosine of the angle between vectors
    cos_angle = torch.sum(v1_norm * v2_norm, dim=-1)
    cos_angle = torch.clamp(cos_angle, -1, 1)  # Clamp for numerical stability
    # Calculate the angle in radians
    angle_rad = torch.acos(cos_angle)
    return angle_rad




def calc_median_angular_error(t_azimuth, t_elevation, p_azimuth, p_elevation):
    """Calculate the median angular error between true and predicted angles."""
    # Convert angles to Cartesian coordinates
    t_cartesian = to_cartesian(t_azimuth, t_elevation)
    p_cartesian = to_cartesian(p_azimuth, p_elevation)
    # Calculate the angular distance in radians
    ang_dist_rad = angular_distance(t_cartesian, p_cartesian)
    # Convert angular distance from radians to degrees
    ang_dist_deg = torch.rad2deg(ang_dist_rad)
    # Calculate median angular error in degrees
    
    
    return ang_dist_deg
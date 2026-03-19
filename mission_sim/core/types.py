from enum import Enum

class CoordinateFrame(Enum):
    """
    空间坐标系定义字典
    """
    # 惯性系 (Inertial)
    J2000_ECI = "J2000_Earth_Centered_Inertial"  # 地心惯性系
    J2000_SSB = "J2000_Solar_System_Barycenter"  # 太阳系质心惯性系
    
    # 旋转系 (Rotating)
    SUN_EARTH_ROTATING = "Sun_Earth_Rotating_Barycenter" # 日地旋转系 (CRTBP所用)
    EARTH_MOON_ROTATING = "Earth_Moon_Rotating_Barycenter" # 地月旋转系

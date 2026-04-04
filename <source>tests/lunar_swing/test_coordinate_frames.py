"""
CoordinateFrame 扩展的桩测试

验证新增的 EARTH_MOON_ROTATING 坐标系是否正确定义。
"""
import pytest
from mission_sim.core.spacetime.ids import CoordinateFrame

def test_coordinate_frame_enum_extension():
    """测试 CoordinateFrame 枚举是否包含 EARTH_MOON_ROTATING"""
    # 验证现有枚举值
    assert hasattr(CoordinateFrame, 'J2000_ECI')
    assert hasattr(CoordinateFrame, 'SUN_EARTH_ROTATING')
    assert hasattr(CoordinateFrame, 'LVLH')
    assert hasattr(CoordinateFrame, 'VVLH')
    
    # 验证新增的枚举值
    assert hasattr(CoordinateFrame, 'EARTH_MOON_ROTATING')
    
    # 验证枚举值唯一性
    frames = list(CoordinateFrame)
    assert len(frames) == len(set(frames))
    
    # 验证枚举值顺序（新增值应在 SUN_EARTH_ROTATING 之后）
    frame_list = [frame.name for frame in CoordinateFrame]
    sun_earth_idx = frame_list.index('SUN_EARTH_ROTATING')
    earth_moon_idx = frame_list.index('EARTH_MOON_ROTATING')
    
    # EARTH_MOON_ROTATING 应在 SUN_EARTH_ROTATING 之后
    assert earth_moon_idx > sun_earth_idx

def test_coordinate_frame_string_conversion():
    """测试坐标系字符串转换"""
    # 从字符串获取枚举
    frame1 = CoordinateFrame['EARTH_MOON_ROTATING']
    assert frame1 == CoordinateFrame.EARTH_MOON_ROTATING
    
    # 从值获取枚举
    frame2 = CoordinateFrame(frame1.value)
    assert frame2 == CoordinateFrame.EARTH_MOON_ROTATING

def test_coordinate_frame_usage_example():
    """测试坐标系使用示例"""
    # 模拟使用场景
    def get_frame_matrix(frame: CoordinateFrame):
        """根据坐标系返回转换矩阵（桩实现）"""
        if frame == CoordinateFrame.EARTH_MOON_ROTATING:
            # 地月旋转坐标系转换矩阵
            return "earth_moon_rotation_matrix"
        elif frame == CoordinateFrame.SUN_EARTH_ROTATING:
            # 日地旋转坐标系转换矩阵
            return "sun_earth_rotation_matrix"
        elif frame == CoordinateFrame.J2000_ECI:
            # 惯性坐标系
            return "identity_matrix"
        else:
            raise ValueError(f"Unsupported frame: {frame}")
    
    # 测试新坐标系
    matrix = get_frame_matrix(CoordinateFrame.EARTH_MOON_ROTATING)
    assert matrix == "earth_moon_rotation_matrix"
    
    # 测试现有坐标系（向后兼容）
    matrix = get_frame_matrix(CoordinateFrame.SUN_EARTH_ROTATING)
    assert matrix == "sun_earth_rotation_matrix"
    
    matrix = get_frame_matrix(CoordinateFrame.J2000_ECI)
    assert matrix == "identity_matrix"

if __name__ == '__main__':
    pytest.main([__file__, '-v'])

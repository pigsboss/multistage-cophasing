import unittest
import numpy as np
from mission_sim.core.gnc_subsystem import GNC_Subsystem
from mission_sim.core.types import CoordinateFrame, Telecommand

class TestGNCSubsystem(unittest.TestCase):
    def setUp(self):
        self.frame = CoordinateFrame.SUN_EARTH_ROTATING
        self.gnc = GNC_Subsystem("Alpha", self.frame)

    def test_telecommand_rejection(self):
        """测试 GNC 是否能正确拒绝坐标系不匹配的非法指令"""
        
        # 构造一个恶意的/错误的指令包：坐标系设为 J2000_ECI
        bad_packet = Telecommand(
            cmd_type="ORBIT_MAINTENANCE",
            target_state=np.zeros(6),
            frame=CoordinateFrame.J2000_ECI  # 这是一个与 GNC 不匹配的坐标系
        )
        
        # 断言：当传入 frame 错误的指令时，GNC 必须抛出 ValueError
        with self.assertRaises(ValueError):
            self.gnc.process_telecommand(bad_packet)

    def test_telecommand_acceptance(self):
        """测试 GNC 是否能正确接收合法指令 (如有此测试，也一并更新)"""
        good_packet = Telecommand(
            cmd_type="ORBIT_MAINTENANCE",
            target_state=np.array([1.5e11, 0, 0, 0, 0, 0]),
            frame=CoordinateFrame.SUN_EARTH_ROTATING  # 坐标系完美匹配
        )
        
        self.gnc.process_telecommand(good_packet)
        self.assertEqual(self.gnc.control_mode, "ORBIT_MAINTENANCE")
        self.assertTrue(self.gnc.is_active)

    def test_control_output(self):
        """测试简单比例控制输出"""
        self.gnc.target_state = np.zeros(6)
        self.gnc.estimated_state = np.array([100.0, 0, 0, 0, 0, 0]) # 100m 偏差
        
        K = np.zeros((3, 6))
        K[0, 0] = 0.01 # P 增益
        
        force, frame = self.gnc.compute_control_force(K)
        # u = -K * error = -0.01 * 100 = -1.0 N
        self.assertEqual(force[0], -1.0)

if __name__ == '__main__':
    unittest.main()

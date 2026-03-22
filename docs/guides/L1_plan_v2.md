## MCPC 后续计划文件修改需求清单（第一阶段 + 第二阶段）

### 一、核心模块修改（按文件路径）

---

#### 1. `mission_sim/core/gnc/gnc_subsystem.py`
| 操作     | 说明                                                         |
| -------- | ------------------------------------------------------------ |
| **修改** | 重构 `_validate_and_fix_K_matrix` 方法，移除静默备用增益 `np.eye(3,6)*1e-3`，改为在无法安全修复时抛出 `ValueError`，并指向 `_design_control_law`。保留对 `(6,)` 等可修复形状的广播逻辑。 |

#### 2. `mission_sim/core/gnc/propagator.py`
| 操作     | 说明                                                         |
| -------- | ------------------------------------------------------------ |
| **修改** | 在文件末尾新增 `CRTBPPropagator` 类，实现 `propagate` 方法，利用 `CRTBP.dynamics`（需要传入 CRTBP 实例）进行外推。保持与现有外推器相同的接口。 |

#### 3. `mission_sim/core/physics/models/atmospheric_drag.py`
| 操作     | 说明                                                         |
| -------- | ------------------------------------------------------------ |
| **新增** | 实现 `AtmosphericDrag` 类，继承 `IForceModel`。采用指数大气模型，支持配置参考密度、标高、阻力系数。提供 `compute_accel` 方法，计算大气阻力加速度。 |

#### 4. `mission_sim/core/physics/models/__init__.py`
| 操作     | 说明                                                         |
| -------- | ------------------------------------------------------------ |
| **修改** | 添加 `from .atmospheric_drag import AtmosphericDrag` 并更新 `__all__`。 |

#### 5. `mission_sim/core/physics/models/gravity_crtbp.py`
| 操作     | 说明                                                         |
| -------- | ------------------------------------------------------------ |
| **修改** | 抽离核心加速度计算公式为纯函数 `_crtbp_accel(pos, vel, mu, pos_sun, pos_earth, OMEGA)`，用 `@numba.njit` 装饰。原 `compute_accel` 方法调用该纯函数。 |

#### 6. `mission_sim/core/physics/models/j2_gravity.py`
| 操作     | 说明                                                         |
| -------- | ------------------------------------------------------------ |
| **修改** | 抽离核心加速度计算公式为纯函数 `_j2_accel(pos, mu_earth, j2, r_earth)`，用 `@numba.njit` 装饰。原 `compute_accel` 方法调用该纯函数。 |

#### 7. `mission_sim/core/physics/models/srp.py`
| 操作     | 说明                                                         |
| -------- | ------------------------------------------------------------ |
| **修改** | 抽离核心加速度计算公式为纯函数 `_srp_accel(pos, sun_position, area_to_mass, reflectivity, P_solar, AU)`，用 `@numba.njit` 装饰。原 `compute_accel` 方法调用该纯函数。 |

#### 8. `mission_sim/core/dynamics/threebody/base.py`
| 操作     | 说明                                                         |
| -------- | ------------------------------------------------------------ |
| **修改** | 抽离 `dynamics` 中的运动方程核心计算为纯函数 `_crtbp_dynamics_nd(state, mu)`，用 `@numba.njit` 装饰。原 `dynamics` 方法调用该纯函数，并保持类接口不变。 |

#### 9. `mission_sim/simulation/base.py`
| 操作     | 说明                                                         |
| -------- | ------------------------------------------------------------ |
| **修改** | 在 `__init__` 中增加 `self.integrator_type = config.get("integrator", "rk4")`。重构 `_propagate_state` 方法，根据 `integrator_type` 选择积分器：<br> - `"rk4"`：现有 RK4 逻辑。<br> - `"rk45"`：实例化 `scipy.integrate.RK45`，在主循环中调用 `step()` 推进，保持积分器状态。支持事件检测（预留）。 |

#### 10. `mission_sim/simulation/twobody/base.py`
| 操作     | 说明                                                         |
| -------- | ------------------------------------------------------------ |
| **新增** | 创建二体场景基类 `TwoBodyBaseSimulation`，继承 `BaseSimulation`。封装二体环境初始化、J2 引力注册、常用控制律设计（可留空或提供默认 LQR）。供 LEO/GEO 等场景复用。 |

#### 11. `mission_sim/simulation/twobody/leo.py`
| 操作     | 说明                                                         |
| -------- | ------------------------------------------------------------ |
| **新增** | 实现 `LEOL1Simulation`，继承 `TwoBodyBaseSimulation`。实现抽象方法：<br> - `_generate_nominal_orbit`：使用 `KeplerianGenerator` 生成开普勒轨道，或使用 `J2KeplerianGenerator` 生成带 J2 摄动的参考星历。<br> - `_initialize_physical_domain`：注册 `J2Gravity` 和 `AtmosphericDrag`。<br> - `_design_control_law`：设计 LQR 控制器（可复用基类逻辑或自定义）。 |

#### 12. `mission_sim/simulation/twobody/geo.py`
| 操作     | 说明                                                         |
| -------- | ------------------------------------------------------------ |
| **新增** | 实现 `GEOL1Simulation`，继承 `TwoBodyBaseSimulation`。实现方法：<br> - `_generate_nominal_orbit`：使用 `KeplerianGenerator` 生成 GEO 轨道。<br> - `_initialize_physical_domain`：仅注册 `J2Gravity`（无大气）。<br> - `_design_control_law`：同 LEO 类似。 |

#### 13. `run.py`
| 操作     | 说明                                                         |
| -------- | ------------------------------------------------------------ |
| **修改** | 在 `SCENE_MODULE_MAP` 中添加 `"leo": "mission_sim.simulation.twobody.leo"` 和 `"geo": "mission_sim.simulation.twobody.geo"`。 |

---

### 二、工具与脚本修改

#### 14. `mission_sim/utils/math_tools.py`
| 操作     | 说明                                                         |
| -------- | ------------------------------------------------------------ |
| **修改** | 新增函数：<br> - `inertial_to_rotating(state_inertial, t, omega)`：将惯性系状态转换到旋转系。<br> - `rotating_to_inertial(state_rotating, t, omega)`：逆变换。<br>（用于日地旋转系 ↔ 惯性系）<br>并添加必要的注释和类型提示。 |

#### 15. `analysis/control_robustness_analysis.py`
| 操作     | 说明                                                         |
| -------- | ------------------------------------------------------------ |
| **修改** | 重构 `run_monte_carlo`：<br> - 使用 `multiprocessing.Pool` 并行执行 `_run_single_simulation`。<br> - Worker 只计算并返回 `RobustnessMetrics`，不进行文件 I/O。<br> - 主进程汇总结果，定期保存 JSON。 |

#### 16. `analysis/fuel_analysis.py`
| 操作     | 说明                           |
| -------- | ------------------------------ |
| **修改** | 类似地并行化 `run_scan` 方法。 |

---

### 三、测试文件修改与新增

#### 17. `tests/test_gnc.py`
| 操作     | 说明                                                         |
| -------- | ------------------------------------------------------------ |
| **修改** | 新增测试：<br> - `test_gnc_k_matrix_invalid_shape_raises`：传入不可修复形状（如 `(4,6)`），验证抛出 `ValueError`。<br> - `test_gnc_k_matrix_broadcast`：传入 `(6,)` 形状，验证被广播为 `(3,6)` 并成功执行。<br> - `test_crtbp_propagator`：测试 `CRTBPPropagator` 短时外推误差（与线性外推对比）。 |

#### 18. `tests/test_physics.py`
| 操作     | 说明                                                         |
| -------- | ------------------------------------------------------------ |
| **修改** | 新增测试：<br> - `test_atmospheric_drag_acceleration`：验证大气阻力加速度方向与速度方向相反，大小符合指数模型。<br> - 可选测试 JIT 纯函数结果与原始方法一致（在性能测试中）。 |

#### 19. `tests/test_utils.py`
| 操作     | 说明                                                         |
| -------- | ------------------------------------------------------------ |
| **修改** | 新增测试：<br> - `test_inertial_to_rotating_consistency`：验证转换后再逆转换，误差 < 1e-10。 |

#### 20. `tests/test_integration.py`（或新建文件）
| 操作          | 说明                                                         |
| ------------- | ------------------------------------------------------------ |
| **修改/新增** | 新增集成测试：<br> - `test_leo_simulation`：运行 1 天 LEO 仿真，检查输出文件存在，燃料消耗在 0.1~0.5 m/s/天。<br> - `test_geo_simulation`：运行 1 天 GEO 仿真，燃料消耗在 0.01~0.05 m/s/天。<br> - `test_rk45_integrator`：配置 `integrator: "rk45"` 运行 L1 日地 L2 仿真，验证结果与 RK4 一致（允许微小差异）。 |

---

### 四、文档与指南

#### 21. `docs/source/`（Sphinx 项目）
| 操作     | 说明                                                         |
| -------- | ------------------------------------------------------------ |
| **新增** | 创建 Sphinx 文档目录，包含 `conf.py`、`index.rst` 及模块 API 文档。运行 `sphinx-quickstart` 初始化，编写配置，使用 `sphinx.ext.autodoc` 自动生成 API。 |

#### 22. `docs/guides/add_new_scenario.md`
| 操作     | 说明                                                         |
| -------- | ------------------------------------------------------------ |
| **新增** | 编写“添加新场景”的详细指南，包含步骤、代码示例（继承 `BaseSimulation`、实现抽象方法、注册到 `run.py`）。 |

#### 23. `docs/guides/add_new_force_model.md`
| 操作     | 说明                                                         |
| -------- | ------------------------------------------------------------ |
| **新增** | 编写“添加新力模型”指南，包含实现 `IForceModel` 接口、注册到环境引擎的示例。 |

#### 24. `README.md` 和 `README.en.md`
| 操作     | 说明                                                         |
| -------- | ------------------------------------------------------------ |
| **修改** | 添加指向 API 文档和扩展指南的链接；更新“扩展与定制”章节，引用新指南。 |

---

### 五、其他可能涉及的文件

- `mission_sim/simulation/__init__.py`：如需导出新场景，可添加 `__all__`，但非必需。
- `mission_sim/__init__.py`：同上。
- `mission_sim/core/physics/models/__init__.py` 已在 #4 中处理。
- `requirements.txt`：若添加 `numba`、`sphinx` 等新依赖，需更新（建议在第二阶段添加 `numba`，在文档阶段添加 `sphinx` 和 `sphinx-rtd-theme` 等）。

---

### 六、修改统计汇总

| 类型         | 数量 | 文件列表                                                     |
| ------------ | ---- | ------------------------------------------------------------ |
| **新增文件** | 11   | `atmospheric_drag.py`, `twobody/base.py`, `leo.py`, `geo.py`, `docs/source/`（多个）, `guides/*.md`, `test_leo_integration.py`, `test_geo_integration.py`（可选） |
| **修改文件** | 18   | `gnc_subsystem.py`, `propagator.py`, `models/__init__.py`, `gravity_crtbp.py`, `j2_gravity.py`, `srp.py`, `dynamics/threebody/base.py`, `simulation/base.py`, `run.py`, `math_tools.py`, `control_robustness_analysis.py`, `fuel_analysis.py`, `test_gnc.py`, `test_physics.py`, `test_utils.py`, `test_integration.py`, `README.md`, `README.en.md` |

> 注：部分测试文件可能以新增而非修改方式处理，视具体实现而定。
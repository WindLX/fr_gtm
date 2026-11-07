---
id: IM-1
kind: work
group: general
title: 拆分发动机滞后载荷与组装并迁移调用路径
status: done
completion_note: 发动机职责拆分、配置工厂、系统接线、初始化/trim 路径、测试及文档已完成；载荷公式/系数、默认一阶1.5s与二阶矩阵不变。按用户追加授权补齐必要导入/配置衔接，不恢复用户删除的手柄与uncertainty接口，不重生成冻结参考。相关验收通过，整包既有非发动机失败列在进展中。
completion_evidence: src/fr_gtm/components/engine.py；src/fr_gtm/config.py；src/fr_gtm/system.py；tests/test_engine_structure.py 7 passed；tests/test_engine_loads.py 8 passed；test_engine_first_order.py NumPy/CasADi/native 13 passed，补装锁定JAX后2项JAX另行通过，torch未安装；test_engine_lag.py解析/配平/钳位/尺度回归通过；tests/test_backends.py 23 passed；examples/trim_linearize.py残差1.636e-15；examples/simulate.py成功100步；src strict mypy 13文件通过，发动机及相关测试strict mypy5文件通过；根 just check-docs（含全API漂移）通过。
---

## 目标

按用户确认方案将发动机拆为一阶/二阶滞后 Block、载荷 Block 与组装 System，保持公式、系数、默认一阶 τ=1.5s 和 fullscale/subscale 缩放不变；保留现有其他迁移修改。

## 验收

- Block 与各自配置精确关联，载荷只依赖 EngineLoadsConfig，配置工厂只提供一次默认系数。
- 系统调用、初始化、trim、测试中的状态路径改为 engine.lag::lag_state；公开有效油门/force/moment 不变。
- 一阶解析响应、二阶矩阵/稳态初始化、载荷与反馈因果回归通过；同步文档和配置类型检查。

## 当前下一步

本次发动机重构完成，等待用户审查。用户其他迁移继续处理：test_system的旧质量/uncertainty用例、test_config的已删除uncertainty与缺失AircraftConfig形状校验、test_reference的传感器油门扰动0.6差异、手柄/Proto旧接口的残余测试和scipy-stubs环境依赖。本次保留这些状态，不扩展到其他物理模型。

## 进展

2026-11-05：已读取当前修改及相邻调用。发现 config.py 缺少尺度常量/多个配置定义导入，components/__init__ 与 config 的依赖循环。未覆盖这些用户迁移。
2026-11-05：Engine 改为 System，内部 lag/loads 分离；配置类型为 FirstOrderThrottleLagConfig、JetCatThrottleLagConfig、EngineLoadsConfig 与 EngineConfig。工厂 loads 为唯一几何快照，原完整多项式系数只定义一次；默认一阶 1.5s。已更新初始化、trim、示例和相关测试为 Root.engine.lag::lag_state。EngineLoads/反馈8项、新结构/公式7项通过；一阶 NumPy/CasADi/native 解析响应通过，补装独立项目锁定 JAX 后两项JAX响应通过；发动机代码和4个测试文件 strict mypy通过，src全部strict mypy通过。八种尺度/构型冻结值通过；扰动参考在传感器油门通道差0.6，未改原参考或传感器参数。
2026-11-05：最终tests/test_backends.py为23 passed（含NumPy/CasADi/native/JAX、整机冻结轨迹、地面及零速Jacobian）；必要Wind向量转标量衔接采用原有smooth_norm epsilon=1e-3，未改数学约定。完整test_reference为10 passed/1 failed，仅uncertainty_sensor油门通道52.6对52.0；完整test_system为12 passed/1 failed，旧质量测试仍把GTMConfig传给现有AircraftConfig入口；test_config为12 passed/2 failed，旧uncertainty断言及现有AircraftConfig形状检查缺失。项目just check：lint/format通过，类型检查10项错误均在既有测试/手柄/Proto缺失接口及scipy-stubs未安装；src与发动机窄类型检查通过。根just check-docs已通过（构建、格式/类型、链接/正文、API注释及全API漂移）；只补齐迁移类型的文档注释，不更改其他模型参数。
- 2026-11-06：用户审查后继续优化：保留组件配置工厂与frozen方案，EngineLoads采用整向量缩放/加法，JetCat采用A@x+B*u与C@x，一阶不拆装标量。按用户新要求发动机工厂仅接收显式S/cg_pos，无AircraftConfig依赖；共享SCALE_FACTOR放constants，GTMConfig负责注入面积与重心。修复本轮用户修改引入的循环导入、已删除工厂残余调用、全尺寸h_gain漏乘尺度五次方，并恢复原滞后配置校验。engine/wind/sensor/mass共45个input/state/param/output声明已补齐description；fr.export不接收description且表达式导出无法携带独立说明，此项为现有核心API限制。结构/公式/工厂及JetCat四后端13项通过，engine系统相关2项通过，src及结构测试strict mypy、Ruff通过；GTM API已重新生成。整机后端回归正在运行。
- 2026-11-06：补齐本轮新增frozen model_config摘要；check-docstrings通过。发动机4文件回归日志为47 passed/2 skipped（torch未安装），新增矩阵/工厂最终13项及系统2项另外通过。
- 2026-11-06：最终整机tests/test_backends.py 23 passed（NumPy/CasADi/native/JAX），API docstring门禁缺口0，Ruff及相关格式/strict mypy通过。README、实现说明与生成API同步工厂S/cg_pos签名。后续不扩展fr.export接口；当前表达式导出的独立description仍受该公开接口限制。

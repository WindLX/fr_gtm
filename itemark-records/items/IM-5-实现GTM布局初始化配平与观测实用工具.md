---
id: IM-5
kind: work
group: general
title: 实现GTM布局初始化配平与观测实用工具
status: done
completion_note: 已实现布局目录、物理初始化、完整飞行配平和观测工具；删除旧初始化函数及 constants，迁移非 stick/proto 示例并接入 Graph WebUI，同步指南及生成 API。
completion_evidence: tests/test_model_tools.py、test_trim_tools.py、test_system.py 共35通过；相关后端/配置/载荷/参考回归57通过、1个既有基线差异；非集成范围41文件mypy与lint/format通过；just test-examples、JAX编译基准、native短接地示例及根just check-docs/API漂移通过。
---

## 目标

实现已确认的GTMPaths、直接参数init/build_input/噪声工具/read_observation、灵活altitude/tas/gamma/psi/phi配平；删除constants并迁SCALE_FACTOR到config；同步测试示例文档，保留其他进行中修改及数值模型。

## 验收

- 路径支持Root和嵌套前缀，沿用现有通道索引；无固定x/p维数。
- init按NED位置、alpha/beta/TAS、欧拉角、命令、风/相对不确定度生成x/xd/u/p，有风时来流正确，发动机阶数兼容；噪声d标准差pi/n标准差1，维度4/17。
- trim使用TrimSolver和专用条件包装，目标可释放，真实力/力矩及内部状态平衡，结果按路径还原生产模型布局；统计物理残差。
- 删除constants后无循环导入，公共调用/示例/文档与类型检查同步；回归不改冻结基线。

## 当前下一步

本轮实现完成。后续若继续推进完整门禁，需独立处理 uncertainty_sensor 的有效油门冻结基线差异、stick/proto 迁移遗留导入及 scipy-stubs 缺失；本轮未改变这些模型公式或冻结数据。

## 验证

新工具与结构测试35 passed；相关回归57 passed，uncertainty_sensor 仍为有效油门52.6对冻结值52.0的既有差异。lint/format 与41文件mypy通过；完整类型检查另有删除中stick/proto模块和scipy-stubs缺失。just test-examples、JAX编译基准、native touchdown短示例、Graph HTTP与实际浏览器拓扑检查通过；根just check-docs及全API漂移通过。

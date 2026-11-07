---
id: IM-4
kind: work
group: general
title: 去重诊断与物理观测并修正刚体航迹角
status: done
completion_note: 完成七组件out逐项覆盖：根仅y/clean_y各17维、不拼u；effects保留未被x/观测/已有子effects覆盖的组件输出，不导出quat_dot或状态别名。按用户授权修正std四元数/矩阵刚体gamma/chi的NED速度语义，GTM复用eom.gamma并转deg。调用与文档同步。
completion_evidence: src/fr_gtm/system.py及tests/test_system.py严格效果清单21项；GTM系统14 passed；四后端23 passed；std test_rigid_body_flight_path.py 4 passed；SDK改动及测试strict mypy/Ruff通过，GTM源码和相关测试strict mypy15文件通过；根just check-docs通过含API漂移；trim residual1.636e-15、C17x10/D17x4；simulate/disturbances成功。完整reference仍10 passed/1既有sensor油门0.6差异；手柄/Proto已删除接口残余依赖仍未恢复。
---

## 目标

只导出17维物理y/clean_y，移除命令拼接；effects不重复状态、可重算量或已有子组件诊断；删除quat_dot_reference。经用户确认修正std两种刚体的gamma/chi为NED速度角，再让GTM复用。

## 验收

- 根仅两个17维输出，总长34；默认effects最终21项，覆盖组件out未被状态/观测/已有子effects提供的数据，无quat_dot、状态或总风别名。\n- std四元数/旋转矩阵的NumPy/CasADi航迹角回归通过，GTM直接degrees(eom.gamma)。\n- 调用、参考测试边界、文档同步，不修改冻结基线；系统及四后端轨迹有证据。

## 当前下一步

本轮完成，等待用户审查。既有传感器扰动参考差异及手柄/Proto删除接口迁移留待原工作继续，不改写冻结基线。

## 进展

- 2026-11-06：实现完成：GTM仅y/clean_y各17维，n_y34；默认effects严格9项（5个根内部诊断+2个Aero+2个Wind），移除状态、VE/Mach、重力、总风和quat_dot。参考测试仅在边界重构raw四元数导数、重力与总风，比较冻结观测前17项，不改冻结文件。std四元数/矩阵共用NED航迹角计算，GTM直接degrees(eom.gamma)。SDK NumPy/CasADi两表示4项通过，GTM四后端23项通过，trim residual1.636e-15、C17x10/D17x4，simulate成功。
- 2026-11-06：按用户进一步要求核对七个组件out：剔除x/观测已覆盖的状态和物理量、及已有子组件effects后，补充质量/惯量4项、大气VEAS/Mach/压力比3项、刚体DCM/NED速度/chi/平动与角加速度5项。默认effects最终21项（17根诊断+4子组件诊断）；不恢复quat_dot、状态别名、总风或重力计算项。已更新完整out覆盖回归清单，系统14项通过；手柄例子从x取位置/四元数，从现有effects取其余诊断，不复制大气计算。
- 2026-11-06：最终out覆盖候选：GTM系统14项、四后端23项通过；std两种刚体4项通过；SDK改动与新测试strict mypy/Ruff通过，GTM源码及相关测试strict mypy/Ruff通过。原传感器扰动reference仍差0.6，其余10项通过；未修改冻结参考。示例与配平已同步17通道，C17x10/D17x4。

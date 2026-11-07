---
id: IM-2
kind: work
group: general
title: 合并气动配置并统一整机构造工厂
status: done
completion_note: 完成本阶段气动与配置重构：单一扁平AeroConfig与Aero同模块、无顶层配置依赖，GTMConfig仅fullscale/subscale，相关调用和文档迁移。control_arm_ratios依据系数量纲明确为y/b，不作全尺寸二次缩放。保留用户其他迁移及冻结参考。
completion_evidence: src/fr_gtm/components/aero.py；src/fr_gtm/config.py；tests/test_aero.py 4 passed；tests/test_config.py 16 passed；tests/test_backends.py 23 passed；test_cubic 4 passed/1 skipped；test_cubic_linear 4 passed；test_control_table_mixed_ad 1 passed；发动机三文件36 passed/2 skipped；源码和相关测试strict mypy16文件通过；Ruff及格式通过；just example-trim残差1.636e-15；根just check-docs通过。完整reference10 passed/1既有sensor油门0.6差异；完整system12 passed/1既有Mass/uncertainty旧用例失败；整项目mypy9项既有错误。
---

## 目标

合并扁平AeroConfig与Aero，解除气动对整机配置的依赖；GTMConfig只提供fullscale/subscale工厂，保留气动公式、资源方向、八种尺度构型及默认发动机响应。

## 验收

- Aero.Config精确关联AeroConfig，几何/模式来自该配置，删去三个子配置及aero_config模块。
- GTMConfig两个工厂完整组装wind/ground/engine/actuator/aero/sensor，迁移全部有效调用、测试与文档；无default/from_toml兼容层。
- 气动信号description齐全，配置/独立载荷/冻结参考/多后端轨迹、类型与文档检查有证据。

## 当前下一步

本阶段已交付，等待用户审查。用户其他迁移继续处理Mass/uncertainty旧测试、传感器油门扰动参考差异、已删除stick/Proto接口残余调用及scipy-stubs缺失；本阶段不改这些模型或冻结基线。

## 进展

- 2026-11-06：已合并AeroConfig到components/aero.py并扁平化，Aero只消费该配置；统一b/Cbar用于动导和力矩，移除三个气动子配置及aero_config模块。GTMConfig仅保留fullscale/subscale，按用户即时精简保留无aircraft参数的工厂签名；迁移源码/例子/测试及有效benchmark调用，清除已删除TOML测试与rtoml依赖，保留其他迁移。局部配置/气动19项通过，整机四后端23项通过，冻结参考10通过/1既有sensor差0.6失败。用户指出control_arms尺度问题后核查原gtm_ad和量纲：矩阵中的arm必须是y/b无量纲比值，改名control_arm_ratios，不进行二次尺度放大；新增非对称矩阵/量纲回归，气动4项通过。
- 2026-11-06：补充检查：tests/test_cubic.py 4通过/1torch未安装跳过，test_cubic_linear.py 4通过，test_control_table_mixed_ad.py 1通过，完整test_system.py 12通过/1既有Mass/uncertainty旧用例失败；example-trim约束残差1.636e-15，线性化A10x10/B10x4/C21x10/D21x4。源码与新配置/气动测试strict mypy通过，全项目mypy仍9项既有错误（已删除stick/Proto、Mass/uncertainty旧测试、scipy-stubs缺失）。根just check-docs最终通过含API漂移；第一次格式门禁失败已格式化修复。
- 2026-11-06：按用户要求仅增强升降舵段落注释：区分三到六通道嵌入矩阵、单段K(arm)与四段矩阵之和，解释零偏基准扣除、1/4等权分配、FRD叉乘符号、y/b归一化、对称抵消及与最终重心平移的区别。执行语句未改；该文件Ruff与格式检查通过。

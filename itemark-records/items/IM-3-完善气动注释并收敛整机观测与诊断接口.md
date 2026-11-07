---
id: IM-3
kind: work
group: general
title: 完善气动注释并收敛整机观测与诊断接口
status: done
completion_note: 气动全段物理注释补齐且执行表达式不变；GTMSystem中文注释、接口失配和航迹角来源/单位修复；根仅y/clean_y，额外诊断进入effects，17路传感器噪声及相关调用、测试、文档同步。
completion_evidence: src/fr_gtm/components/aero.py与开始快照排除docstring后AST一致；tests/test_hykal.py与test_aero.py共5 passed；tests/test_system.py 14 passed，地面新增接口断言2 passed；tests/test_backends.py 23 passed；example-trim残差1.636e-15，simulate/disturbances/inspect成功；Ruff/格式及src+相关测试strict mypy16文件通过；根just check-docs通过。完整reference仍10 passed/1既有传感器油门0.6差异；全包类型检查剩5项既有stick/Proto缺失和scipy-stubs错误。
---

## 目标

补齐气动物理注释，重点解释HyKal流向轴稳转/剩余角速率分解及动导合成；复核并修正GTMSystem连线，改为中文注释；按用户追加要求仅导出y/clean_y，其他独立诊断量迁入effects，同步调用、测试和文档。保留气动公式。

## 验收

- 角度、速率、坐标系、无量纲归一化、查表通道与矩阵意义明确，气动执行表达式不改变。
- 系统端口正确、航迹角来源/单位正确、17路噪声匹配传感器，根仅两个21维输出；effects诊断可读。
- 结构/接线/因果反馈/HyKal分支和跨后端回归、类型与文档检查有记录。

## 当前下一步

本阶段完成，等待用户审查。现有手柄/Proto删除接口迁移及传感器扰动参考差异仍由用户后续处理，本轮不恢复旧接口或改变传感器公式/冻结参考。

## 进展

- 2026-11-06：实现完成：Aero非docstring执行AST与本轮开始完全一致；补齐HyKal的来流轴分解、p/r候选优先级与符号/幅值条件、q剩余修正、D=diag(b,Cbar,b)/(2V)归一化、动导三/六通道嵌入及非线性表值含义。GTMSystem全中文说明，修正aero端口旧引用、NED航迹角来源和度制转换、Wind缺epsilon，根只导出y/clean_y，独立诊断迁effects，噪声对齐17路。系统14通过，局部气动/HyKal5通过，四后端23通过，trim残差1.636e-15，simulate成功100步。完整reference仍10通过/1既有sensor油门0.6差异；全项目mypy剩5项既有缺失stick/Proto与scipy-stubs错误，src及相关测试strict mypy通过，Ruff通过。
- 2026-11-06：追加验收：NED俯仰爬升用例确认航迹角用NED速度并转deg；根y键集合严格两项且n_y=42、n_u=25，地面开启也不增加y端口。原Mass/uncertainty旧测试已按现有AircraftConfig和标准重力合同迁移，不改Mass公式。示例simulate/disturbances/inspect与trim实际运行成功。SDK真实气动benchmark衔接新的altitude入口；原先已删除的手柄/Proto依赖仍不恢复。
- 2026-11-06：根just check-docs最终通过（含文档构建/格式/链接/正文/注释/API漂移）。文档同步两根输出、42维打包y、17维噪声及effects读取；补充HyKal推导与冻结EngineConfig的替换式修改用法。地面根接口额外断言2通过。原参考扰动用例仍保留既有差异，不重生成基线。

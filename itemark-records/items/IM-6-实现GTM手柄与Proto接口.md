---
id: IM-6
kind: work
group: general
title: 实现GTM手柄与Proto接口
status: done
completion_note: 已实现 stick/proto 两个可选接口，提供精确类型的 command 映射、clean_y 真值协议投影及 effects 遥测绑定；迁移实时手柄示例、依赖 extras、测试和文档。
completion_evidence: tests/test_stick_input.py 与 test_proto_projection.py 共19 passed；含四构型NumPy/CasADi、std坐标对照和遥测展开；lint/format通过，排除既有scipy-stubs缺口后的44文件mypy通过；NumPy/native dry-run及根just check-docs通过。
---

## 目标

实现 interface/stick.py 与 proto.py：完整设备快照映射 command，无噪声飞机真值转换为 AircraftState，按选定 effects 注册/提取遥测；迁移实时示例与测试文档。

## 验收

精确类型和可选依赖隔离；轴方向/死区/不对称边界测试；真值不受 n 与 Sensor 误差影响；协议坐标/单位/实际操纵面正确；遥测字段顺序、单位和多维展开一致；离线实时示例可运行。

## 验证

接口测试19通过；此前连同init/trim工具回归39通过（当时接口18项，随后增加WGS-84与std对照测试并通过）。所有源码/测试/示例lint与format通过。完整mypy当前仅test_cubic.py缺scipy-stubs，排除该文件的44文件检查通过。just test-examples-stick与native无设备/无网络短流程通过。根just check-docs包括生成API漂移通过。

## 边界

真实手柄读取与Proto服务端网络发送未进行联调；接口测试使用真实协议对象以及显式类型的无硬件轴快照，不修改外部包、模型方程或冻结数值基线。

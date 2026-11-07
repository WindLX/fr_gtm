---
id: IM-7
kind: work
group: general
title: 联调GTM与本地Proto服务
status: done
completion_note: 本地独立Proto服务与native无手柄示例联调通过；实际网络状态、真值隔离、effects逐项对照和控制台曲线均验证，服务保持运行。
completion_evidence: build/proto-integration/verification.json：721状态/720遥测，720推进步真值与12诊断标量均与Recorder完全一致；noise-isolation.json：测量TAS143.38至155.28而网络真值100，误差0；console.jpg；会话gtm-proto-integration保存成功。
---

## 目标

启动本机Proto独立服务，以无真实手柄模式联调fly_stick_gtm，核对真值状态与effects遥测的接收和控制台显示。

## 验收

服务端隔离测试数据目录并绑定本机；示例实际握手、状态更新与遥测发送成功；服务端series和实测真值/诊断对照；控制台实际显示状态和遥测曲线；交付可复用的本机访问入口与运行状态。

## 运行与位置

UDP127.0.0.1:18002、控制台http://127.0.0.1:18003，服务PID3077383。主示例native、--no-stick、12秒60Hz，已正常关闭；服务端保持运行。隔离数据根build/proto-integration，包含客户端Parquet、接收样本、核对脚本、JSON证据和截图。

## 结果

真实UDP握手、状态、遥测及HTTP查询通过；721帧状态含初始帧，720帧遥测每帧12通道，逐项网络差0。另20帧固定物理快照、变化传感器高斯噪声及偏置/比例误差探针证实始终发送clean_y。控制台已绘制真空速、气动力/发动机力X、气动系数CZ，工作区保留且会话保存成功。没有连接真实手柄或修改模型/协议源码。

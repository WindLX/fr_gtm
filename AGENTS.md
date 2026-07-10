# fr_gtm 工作指南

## 项目用途

`fr_gtm` 是 NASA Generic Transport Model 的 FlyRuler 实现。顶层连接 actuator、wind subsystem、mass、aero、engine、std rigid-body EOM 与 sensor；数值行为以冻结的 `gtm_ad` 参考轨迹为准。

## 工具链与验证

- 查看命令：`just --list`；安装：`just setup`；格式化：`just fmt`；构建：`just build`。
- 局部测试：`just test-one <path>`；项目验证：`just check`、`just test`。
- 模型验收：`just example-inspect`、`just example-simulate`、`just example-backends`、 `just example-trim`；交付前：`just pre-commit`。

## 本项目约束

- `gtm_ad` 只用于生成冻结 `.npz` reference；常规测试不得依赖外部 checkout。
- 保持 full-scale/scale model、四种 flap/gear 构型、表格方向与单位转换一致。
- 坐标系使用 NED/FRD，quaternion scalar-first；高度为 `-pos_ned[2]`。
- atmosphere、wind 与 rigid-body EOM 优先复用 std；GTM 特有兼容逻辑留在模型侧。
- 气动表、状态、参数和结果均使用完整路径；运行时不加载 MATLAB/Scipy 数据。
- NumPy、CasADi、native、JAX 轨迹与冻结 reference 的变化必须明确审查和更新说明。
- 编写 markdown 文档时，没有新段落时，不要因为行过宽而换行。

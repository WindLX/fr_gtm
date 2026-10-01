# fr_gtm

NASA Generic Transport Model 的 FlyRuler 实现，数值行为以冻结的 `gtm_ad` 参考轨迹为准。

## 事实源与入口

- **约定**：命令以 `just --list` 为准；安装/格式化/构建使用 `just setup` / `just fmt` / `just build`，测试用 `just test-one <path>`，项目门 `just check` / `just test` / `just pre-commit`。
- **约定**：模型验收使用 `just example-inspect`、`just example-simulate`、`just example-backends`、`just example-trim`。
- `tests/` —— 冻结 trajectory reference 与多 backend 回归

## 本目录特有边界

- **必须**：`gtm_ad` 仅用于生成冻结 `.npz` reference；常规测试不依赖外部 checkout。
- **必须**：保持 full-scale/scale model、四种 flap/gear 构型、表格方向与单位转换一致。
- **必须**：使用 NED/FRD、scalar-first quaternion，高度为 `-pos_ned[2]`。
- **必须**：优先复用 std atmosphere、wind 和刚体 EOM；GTM 特有兼容逻辑留在模型侧。
- **禁止**：气动表、状态、参数和结果使用不完整路径；运行时加载 MATLAB/Scipy 数据。
- **必须**：NumPy、CasADi、native、JAX 轨迹变化同步冻结 reference 说明。
- **默认**：大任务拆成阶段；每阶段完成后等待用户审查再继续。

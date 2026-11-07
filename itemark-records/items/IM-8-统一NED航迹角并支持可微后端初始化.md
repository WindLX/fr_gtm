---
id: IM-8
kind: work
group: general
title: 统一NED航迹角并支持可微后端初始化
status: done
completion_note: 航迹角统一到std ned_vel_to_flight_path，删除旧body函数与刚体私有重复实现；GTM初值共享std Quaternion/来流函数，提供可变换的后端原生Initializer，JAX build_input不转宿主；保留已支持嵌套I/O转接所需的io_root。
completion_evidence: GTM新旧初始化/trim/Proto相关33测试与std坐标/两种刚体15测试通过；JAX一阶二阶jit/vmap、正反向导数及有限差分通过；GTM45文件窄mypy与std3文件窄mypy、lint/format和根check-docs通过。std整包类型门禁因环境缺torch阻碍。
---

## 目标

统一坐标库的NED航迹角函数，移除旧body_vel_to_flight_path与刚体私有重复实现；初值复用std Quaternion和来流转换，提供真正支持JAX变换的后端原生构造路径。

## 验收

两种刚体保持NED及epsilon数值行为；NumPy/CasADi初值与配平回归通过；JAX一阶/二阶初始化可jit/vmap/jacfwd/jacrev、导数与有限差分一致，有风及原生PRNG输入不转宿主；文档与调用同步。

## 当前下一步

本轮完成。io_root并非无用字段，已向用户说明其嵌套父子路径职责并保留。若继续执行全项目类型门禁，需在对应环境补齐torch与scipy-stubs。

## 进展

## 历史进展

## 决策

io_root实际用于嵌套GTM将I/O转接到父系统，而状态/参数位于子实例的场景，现有嵌套初始化测试依赖它，保留以避免破坏已支持的用法。init为主机校验入口，新增Initializer在JAX变换外构造一次，内部将std Quaternion.from_euler、aero_to_body_vel与油门平衡映射编译为目标后端；build_input对JAX保留原生值。

## 验证

GTM相关33 passed，std坐标和刚体15 passed；坐标CasADi示例运行通过。GTM类型检查src/tests/examples排除既有test_cubic.py的scipy-stubs缺口后45文件通过；std修改的3文件窄类型检查通过，std整包lint/format通过但类型检查因torch未安装失败。根just check-docs含API漂移通过。所有已有用户更改保留；config中序列单位转换结果显式收回tuple，trim的求解器str路径映射与模型build映射分开注解，数值不变。

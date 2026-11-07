---
id: IM-9
kind: work
group: general
title: 配置独立CI并验证公开SDK最新版本
status: done
completion_note: 独立GitHub Actions配置、SDK浮动升级与本地等价验收完成；用户授权提交推送，远程Actions成功不作为本轮验收前提。
completion_evidence: .github/workflows/ci.yml、scripts/check_install.py、pyproject.toml；actionlint v1.7.12通过；/tmp/fr-gtm-ci-g3QlTR独立Python3.12环境just check通过（47文件），just test为177 passed/2 CUDA skipped，离线及无硬件示例通过；build/ci-validation/install.log包含wheel/sdist两次公开SDK安装与十步仿真通过；Twine元数据及主站check-docs通过，日志/tmp/gtm-ci-*.log。
---

## 目标

为独立fr_gtm配置GitHub Actions，覆盖类型、回归及打包，不使用父仓源码/环境；按用户最新要求CI测试PyPI最新稳定fly-ruler，不固定0.1.7。

## 验收

独立checkout与公开依赖解析；复用just检查、逐文件回归和无硬件示例；wheel/sdist干净安装与气动资源仿真通过；工作流语法及本地等价命令通过，说明远程运行状态。

## 当前下一步

配置交付；后续main/PR自动运行CI，按需workflow_dispatch，不因额度问题重复触发或修改账户配置。

## 进展

用户后续要求CI不锁SDK版本：依赖与extras改为>=0.1.7，CI显式--upgrade-package fly-ruler；本地lock仍保留可复现结果。三个独立作业已实现；actionlint v1.7.12通过；隔离目录Python3.12的just check通过（47文件），wheel与sdist分别干净安装、导入来源及两尺度四构型/十步仿真验收通过。完整CPU后端回归和离线示例正在运行。
用户授权提交推送独立仓CI及主仓引用；明确远程Actions额度可能不足，不将远程成功设为交付前提，不额外workflow_dispatch。完成本地验收后推送。
Python3.12完整CPU后端回归177 passed，只有2项CUDA因无GPU跳过；所有离线示例及stick dry-run通过。wheel/sdist分别安装到/tmp独立环境并用-I验收，SDK当前解析为公开0.1.7而非硬编码；类型、actionlint、元数据与主站文档门禁通过。

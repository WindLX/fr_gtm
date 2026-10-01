# GTM Model - Justfile

set shell := ["bash", "-c"]

# 默认任务
default: help

# 显示所有可用任务
help:
    @just --list

# ============================================================
# 初始化和设置
# ============================================================

# 初始化虚拟环境并安装依赖
setup:
    echo "Synchronizing project dependencies..."
    uv sync
    echo "✓ Setup complete!"

# 安装手柄控制示例的可选依赖
setup-stick:
    echo "Synchronizing optional fly-stick dependencies..."
    uv run --with-editable ../../packages/fly_stick --with-editable ../../packages/fly_ruler_proto/bindings/python python -c "import fly_stick, fly_ruler_proto_python"
    echo "✓ fly-stick example setup complete!"

# ============================================================
# 测试
# ============================================================

# 运行所有测试
test:
    echo "Running all tests..."
    uv run pytest tests/ -v
    echo "✓ All tests passed!"

# 运行特定测试文件
test-one TEST:
    uv run pytest {{TEST}} -v

# ============================================================
# 代码质量
# ============================================================

# 代码格式化
fmt:
    echo "Formatting Python code..."
    uv run ruff format src/ tests/ examples/
    echo "✓ Formatting complete!"

# 代码检查 (Lint + 格式 + 类型)
check: check-lint check-types
    echo "✓ All checks passed!"

# Lint 检查
check-lint:
    echo "Checking Python code..."
    uv run ruff check src/ tests/ examples/ --exclude generate_gtm_backend_reference.py
    uv run ruff format src/ tests/ examples/ --check --exclude generate_gtm_backend_reference.py

check-types:
    echo "Running type check..."
    uv run mypy src/ tests/ examples/ --exclude 'tests/generate_gtm_backend_reference\.py'

build:
    echo "Building wheel and source distribution..."
    uv build

# 修复自动可修复的 linting 问题
fix:
    echo "Fixing Python code..."
    uv run ruff check --fix src/ tests/ examples/
    uv run ruff format src/ tests/ examples/
    echo "✓ Fixes applied!"

# ============================================================
# 示例
# ============================================================

example-inspect:
    uv run python examples/inspect_model.py

example-simulate:
    uv run python examples/simulate.py

example-disturbances:
    uv run python examples/disturbances.py

example-trim:
    uv run python examples/trim_linearize.py

example-backends:
    uv run python examples/backend_compare.py

# Measure generated JAX graph size and local trace/lower/compile latency.
benchmark-jax-compile:
    JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=true uv run --extra jax python examples/benchmark_jax_compile.py

example-ground *ARGS:
    uv run python examples/ground_operations.py {{ARGS}}

# 使用 fly_stick 手柄控制 GTM，并可选发送 fly_ruler_proto 状态
example-fly-stick *ARGS:
    uv run --with-editable ../../packages/fly_stick --with-editable ../../packages/fly_ruler_proto/bindings/python python examples/fly_stick_gtm.py {{ARGS}}

# 使用主动侧杆控制升降舵/副翼，T16000 控制方向舵/油门
example-active-sidestick *ARGS:
    uv run --with-editable ../../packages/fly_stick --with-editable ../../packages/fly_ruler_proto/bindings/python python -m examples.fly_active_sidestick_gtm {{ARGS}}

# 运行全部离线示例
test-examples: example-inspect example-simulate example-disturbances example-trim example-backends
    uv run python examples/ground_operations.py --duration 0.01
    echo "✓ All offline examples passed!"

# 无硬件运行两个实时手柄示例
test-examples-stick:
    uv run python examples/fly_stick_gtm.py --dry-run --backend numpy --duration 0.03
    uv run --with-editable ../../packages/fly_stick --with-editable ../../packages/fly_ruler_proto/bindings/python python -m examples.fly_active_sidestick_gtm --dry-run --duration 0.03

# ============================================================
# 清理
# ============================================================

# 清理构建产物和缓存
clean:
    echo "Cleaning build artifacts..."
    find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name '*.pyc' -delete 2>/dev/null || true
    rm -rf build/ dist/ *.egg-info/
    echo "✓ Cleanup complete!"

# 完整清理（包括虚拟环境）
clean-all: clean
    echo "Removing virtual environment..."
    rm -rf .venv
    echo "✓ Full cleanup complete!"

# ============================================================
# 实用工具
# ============================================================

# 显示项目信息
info:
    @echo "GTM (Generic Transport Model) - Project Information"
    @echo "========================================"
    @uv --version
    @echo ""
    @echo "Python version:"
    @uv run python --version
    @echo ""
    @echo "Installed packages:"
    @uv run pip list | grep -E "(fly-ruler|numpy|rtoml)"

# 进入 Python REPL
repl:
    uv run python

# 开发工作流：格式化 + 测试
dev: fmt test
    echo "✓ Development checks complete!"

# 预提交检查
pre-commit: check test
    echo "✓ Ready to commit!"

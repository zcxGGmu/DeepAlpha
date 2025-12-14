# Makefile for DeepAlpha
# 对应 Go 版本的 Makefile

.PHONY: help test test-unit test-integration test-performance test-all test-quick clean install lint format

# 默认目标
help:
	@echo "DeepAlpha - Python交易系统"
	@echo ""
	@echo "可用命令:"
	@echo "  install     - 安装依赖"
	@echo "  test        - 运行所有测试"
	@echo "  test-unit   - 运行单元测试"
	@echo "  test-integration - 运行集成测试"
	@echo "  test-performance - 运行性能测试"
	@echo "  test-quick  - 运行快速测试（仅单元测试）"
	@echo "  test-bench  - 运行性能基准测试"
	@echo "  test-profile - 运行性能分析"
	@echo "  lint        - 代码检查"
	@echo "  format      - 代码格式化"
	@echo "  clean       - 清理临时文件"
	@echo "  run         - 运行应用"
	@echo "  docker-build - 构建Docker镜像"
	@echo "  docker-run  - 运行Docker容器"

# 安装依赖
install:
	@echo "安装依赖..."
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

# 运行所有测试
test:
	@echo "运行所有测试..."
	python run_tests.py --type all

# 运行单元测试
test-unit:
	@echo "运行单元测试..."
	python run_tests.py --type unit

# 运行集成测试
test-integration:
	@echo "运行集成测试..."
	python run_tests.py --type integration

# 运行性能测试
test-performance:
	@echo "运行性能测试..."
	python run_tests.py --type performance

# 运行快速测试
test-quick:
	@echo "运行快速测试..."
	python run_tests.py --type quick

# 运行性能基准测试
test-bench:
	@echo "运行性能基准测试..."
	python run_tests.py --type bench

# 运行性能分析
test-profile:
	@echo "运行性能分析..."
	python run_tests.py --type profile

# 代码检查
lint:
	@echo "运行代码检查..."
	flake8 deepalpha tests --max-line-length=100 --ignore=E203,W503
	mypy deepalpha --ignore-missing-imports
	black --check deepalpha tests

# 代码格式化
format:
	@echo "格式化代码..."
	black deepalpha tests
	isort deepalpha tests

# 清理
clean:
	@echo "清理临时文件..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .pytest_cache/

# 运行应用
run:
	@echo "启动DeepAlpha..."
	uvicorn deepalpha.main:app --host 0.0.0.0 --port 9991

# 开发模式运行
run-dev:
	@echo "启动开发模式..."
	uvicorn deepalpha.main:app --host 0.0.0.0 --port 9991 --reload

# Docker相关
docker-build:
	@echo "构建Docker镜像..."
	docker build -t deepalpha:latest .

docker-run:
	@echo "运行Docker容器..."
	docker run -p 9991:9991 deepalpha:latest

# 生成依赖文件
requirements:
	@echo "生成依赖文件..."
	pip freeze > requirements.txt
	pip freeze | grep -E "(pytest|flake8|mypy|black|isort)" > requirements-dev.txt

# 数据库迁移（如果使用）
migrate-init:
	@echo "初始化数据库迁移..."
	alembic init alembic

migrate:
	@echo "执行数据库迁移..."
	alembic upgrade head

migrate-new:
	@echo "创建新的迁移..."
	alembic revision --autogenerate -m "$(MSG)"

# 文档生成
docs:
	@echo "生成文档..."
	cd docs && make html

# 测试覆盖率报告
coverage:
	@echo "生成测试覆盖率报告..."
	python run_tests.py --type unit --cov
	@echo "覆盖率报告已生成到 htmlcov/index.html"

# 性能监控
monitor:
	@echo "启动性能监控..."
	python -m memory_profiler deepalpha/main.py

# 安全检查
security:
	@echo "运行安全检查..."
	bandit -r deepalpha
	safety check

# 类型检查
type-check:
	@echo "运行类型检查..."
	mypy deepalpha --strict

# 打包
build:
	@echo "构建包..."
	python -m build

# 发布到PyPI
publish:
	@echo "发布到PyPI..."
	python -m twine upload dist/*

# 版本管理
version-patch:
	@echo "更新补丁版本..."
	bump2version patch

version-minor:
	@echo "更新次版本..."
	bump2version minor

version-major:
	@echo "更新主版本..."
	bump2version major

# Git hooks
install-hooks:
	@echo "安装Git hooks..."
	pre-commit install

# 持续集成模拟
ci:
	@echo "模拟CI流程..."
	$(MAKE) install
	$(MAKE) lint
	$(MAKE) test-quick
	$(MAKE) security
	@echo "CI检查完成"
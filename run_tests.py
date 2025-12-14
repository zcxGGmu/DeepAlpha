#!/usr/bin/env python3
"""
测试运行脚本
对应 Go 版本的 make test 命令
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_unit_tests(coverage=True, verbose=True):
    """运行单元测试"""
    print("\n=== 运行单元测试 ===")

    cmd = ["python", "-m", "pytest"]

    # 添加参数
    if coverage:
        cmd.extend([
            "--cov=deepalpha",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov",
            "--cov-fail-under=80"
        ])

    if verbose:
        cmd.append("-v")

    # 指定测试目录
    cmd.append("tests/test_executor")
    cmd.append("tests/test_decision")
    cmd.append("tests/test_market")
    cmd.append("tests/test_agents")

    # 运行测试
    result = subprocess.run(cmd)
    return result.returncode == 0


def run_integration_tests():
    """运行集成测试"""
    print("\n=== 运行集成测试 ===")

    cmd = [
        "python", "-m", "pytest",
        "tests/test_integration",
        "-v"
    ]

    result = subprocess.run(cmd)
    return result.returncode == 0


def run_performance_tests():
    """运行性能测试"""
    print("\n=== 运行性能测试 ===")

    cmd = [
        "python", "-m", "pytest",
        "tests/test_performance",
        "-v",
        "-s"  # 显示print输出
    ]

    result = subprocess.run(cmd)
    return result.returncode == 0


def run_benchmarks():
    """运行性能基准测试"""
    print("\n=== 运行性能基准测试 ===")

    cmd = ["python", "tests/test_performance/test_benchmarks.py"]
    result = subprocess.run(cmd)
    return result.returncode == 0


def run_profiling():
    """运行性能分析"""
    print("\n=== 运行性能分析 ===")

    cmd = ["python", "tests/test_performance/test_profiling.py"]
    result = subprocess.run(cmd)
    return result.returncode == 0


def run_all_tests(coverage=True):
    """运行所有测试"""
    print("=" * 60)
    print("运行所有测试")
    print("=" * 60)

    results = []

    # 单元测试
    results.append(("单元测试", run_unit_tests(coverage)))

    # 集成测试
    results.append(("集成测试", run_integration_tests()))

    # 性能测试
    results.append(("性能测试", run_performance_tests()))

    # 输出结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)

    all_passed = True
    for test_type, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{test_type}: {status}")
        if not passed:
            all_passed = False

    print("=" * 60)

    if all_passed:
        print("🎉 所有测试通过！")
        return 0
    else:
        print("⚠️ 部分测试失败")
        return 1


def run_quick_tests():
    """运行快速测试（跳过性能测试）"""
    print("=" * 60)
    print("运行快速测试")
    print("=" * 60)

    # 只运行单元测试
    passed = run_unit_tests(coverage=False)

    if passed:
        print("\n✅ 快速测试通过！")
        return 0
    else:
        print("\n❌ 快速测试失败")
        return 1


def test_specific_module(module_name):
    """测试特定模块"""
    print(f"\n=== 测试模块: {module_name} ===")

    cmd = [
        "python", "-m", "pytest",
        f"tests/{module_name}",
        "-v",
        "--cov=deepalpha",
        "--cov-report=term-missing"
    ]

    result = subprocess.run(cmd)
    return result.returncode == 0


def check_dependencies():
    """检查测试依赖"""
    print("检查测试依赖...")

    required_packages = [
        "pytest",
        "pytest-asyncio",
        "pytest-cov",
        "pytest-mock"
    ]

    missing = []
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing.append(package)

    if missing:
        print(f"缺少测试依赖: {', '.join(missing)}")
        print("请运行: pip install " + " ".join(missing))
        return False

    print("✅ 所有依赖已安装")
    return True


def main():
    parser = argparse.ArgumentParser(description="测试运行脚本")
    parser.add_argument(
        "--type",
        choices=["unit", "integration", "performance", "bench", "profile", "all", "quick"],
        default="all",
        help="测试类型"
    )
    parser.add_argument(
        "--module",
        help="测试特定模块 (例如: test_executor)"
    )
    parser.add_argument(
        "--no-coverage",
        action="store_true",
        help="跳过覆盖率统计"
    )

    args = parser.parse_args()

    # 检查依赖
    if not check_dependencies():
        sys.exit(1)

    # 切换到项目根目录
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))

    # 运行测试
    try:
        if args.module:
            passed = test_specific_module(args.module)
            sys.exit(0 if passed else 1)

        coverage = not args.no_coverage

        if args.type == "unit":
            passed = run_unit_tests(coverage)
        elif args.type == "integration":
            passed = run_integration_tests()
        elif args.type == "performance":
            passed = run_performance_tests()
        elif args.type == "bench":
            passed = run_benchmarks()
        elif args.type == "profile":
            passed = run_profiling()
        elif args.type == "quick":
            sys.exit(run_quick_tests())
        else:  # all
            sys.exit(run_all_tests(coverage))

        sys.exit(0 if passed else 1)

    except KeyboardInterrupt:
        print("\n测试被用户中断")
        sys.exit(1)


if __name__ == "__main__":
    main()
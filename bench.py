import subprocess
import time
import sys
import statistics
import os

def run_benchmark():
    # 1. 参数解析
    if len(sys.argv) < 2:
        print("用法: python bench.py <可执行文件路径> [执行次数, 默认10]")
        sys.exit(1)

    executable = sys.argv[1]
    iterations = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    # 检查文件是否存在
    if not os.path.exists(executable):
        print(f"错误: 找不到文件 '{executable}'")
        sys.exit(1)

    print(f"🚀 开始测试: {executable}")
    print(f"📊 设定次数: {iterations}")
    print("-" * 40)

    results = []

    try:
        for i in range(1, iterations + 1):
            # 记录高精度起始时间
            start = time.perf_counter()
            
            # 执行命令并捕获错误
            result = subprocess.run(
                executable, 
                shell=True, 
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.PIPE
            )
            
            # 记录结束时间
            end = time.perf_counter()

            if result.returncode != 0:
                print(f"❌ 第 {i} 次执行出错 (退出码: {result.returncode})")
                continue

            duration_ms = (end - start) * 1000
            results.append(duration_ms)
            print(f"Run {i:02d}: {duration_ms:8.3f} ms")

    except KeyboardInterrupt:
        print("\n⚠️ 测试被用户中断")

    # 2. 统计输出
    if not results:
        print("未获得有效测试数据。")
        return

    print("-" * 40)
    print(f"✅ 测试完成!")
    print(f"平均时间 (Mean): {statistics.mean(results):8.3f} ms")
    print(f"中位数   (Median): {statistics.median(results):8.3f} ms")
    print(f"最大时间 (Max):    {max(results):8.3f} ms")
    print(f"最小时间 (Min):    {min(results):8.3f} ms")
    if len(results) > 1:
        print(f"标准差   (StdDev): {statistics.stdev(results):8.3f} ms")
    print("-" * 40)

if __name__ == "__main__":
    run_benchmark()
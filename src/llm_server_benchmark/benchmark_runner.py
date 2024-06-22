from .benchmark import BenchmarkLLMS
from .benchmark_config import BenchmarkRoot
import yaml


class BenchmarkRunner:
    def __init__(self, config_path: str, benchmark_name: str):
        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)
        self.config = BenchmarkRoot(**config)
        self.benchmark_name = benchmark_name
        self.benchmark_config = [
            b for b in self.config.benchmarks if b.name == self.benchmark_name
        ][0]

    def run(self):
        print("Using the following benchmark config:")
        print(self.benchmark_config.model_dump_json(indent=2))
        print("############")
        for i in range(1, 25 + 1):
            print(f"Running benchmark with PARALLELISM={i}")
            benchmark = BenchmarkLLMS(self.benchmark_config, parallelism=i)
            summary = benchmark.run()
            if summary["avg_tps"] < 12:
                print("TPS is too low, stopping...")
                return
        print("############")

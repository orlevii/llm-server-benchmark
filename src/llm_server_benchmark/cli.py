import click
from .benchmark_runner import BenchmarkRunner


@click.group()
def cli():
    pass


@cli.command(name="run")
@click.option("-n", "--name", required=True)
@click.option("-c", "--config", default="llm_config.yaml", show_default=True)
@click.option("--min_workers", default=1, type=int, show_default=True)
@click.option("--max_workers", default=25, type=int, show_default=True)
@click.option("--min_tps", default=12.0, type=float, show_default=True)
def run(name, config, min_workers, max_workers, min_tps):
    runner = BenchmarkRunner(
        config_path=config,
        benchmark_name=name,
        min_workers=min_workers,
        max_workers=max_workers,
        min_tps=min_tps,
    )
    runner.run()

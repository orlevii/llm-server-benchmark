import click
from .benchmark_runner import BenchmarkRunner


@click.group()
def cli():
    pass


@cli.command(name="run")
@click.option("-n", "--name", required=True)
@click.option("-c", "--config", default="llm_config.yaml", show_default=True)
def run(name, config):
    runner = BenchmarkRunner(config_path=config, benchmark_name=name)
    runner.run()

import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import cached_property
from pathlib import Path
from typing import List

from openai import OpenAI
from .benchmark_config import BenchmarkConfig


@dataclass
class WorkerResult:
    req_times: List[float]
    input_tokens: List[int]
    output_tokens: List[int]

    @cached_property
    def input_tokens_per_min(self):
        return self.total_input_tokens / self.total_time * 60

    @cached_property
    def output_tokens_per_min(self):
        return self.total_output_tokens / self.total_time * 60

    @cached_property
    def total_input_tokens(self):
        return sum(self.input_tokens)

    @cached_property
    def total_output_tokens(self):
        return sum(self.output_tokens)

    @cached_property
    def num_of_requests(self):
        return len(self.req_times)

    @cached_property
    def total_time(self):
        return sum(self.req_times)

    @cached_property
    def tps(self):
        return sum(self.output_tokens) / self.total_time

    @cached_property
    def avg_request_time(self):
        return sum(self.req_times) / len(self.req_times)


class BenchmarkLLMS:
    def __init__(self, config: BenchmarkConfig, parallelism: int):
        self.config = config
        self.parallelism = parallelism
        self.prompt = self._read_prompt()

    def _read_prompt(self) -> dict:
        with open(self.config.prompt_path) as f:
            return json.load(f)

    def _create_openai_client(self) -> OpenAI:
        params = {
            "api_key": self.config.api_key,
        }
        if self.config.base_url:
            params["base_url"] = self.config.base_url
        return OpenAI(**params)

    def worker(self, worker_id: int) -> WorkerResult:
        try:
            client = self._create_openai_client()
            now = datetime.utcnow()
            print("Worker", worker_id, "started", now)
            stop_time = datetime.utcnow() + timedelta(
                seconds=self.config.benchmark_time_sec
            )

            req_times = []
            intput_tokens = []
            output_tokens = []
            debug_print = worker_id == 1
            while now < stop_time:
                req_time, output_tokens_count, input_tokens_count = (
                    self._completion_request(client, debug_print=debug_print)
                )
                req_times.append(req_time)
                output_tokens.append(output_tokens_count)
                intput_tokens.append(input_tokens_count)
                now = datetime.utcnow()
                debug_print = False
        except Exception as e:
            print("Worker ", worker_id, "failed:")
            print(e)
            raise e
        return WorkerResult(
            req_times=req_times, output_tokens=output_tokens, input_tokens=intput_tokens
        )

    def run(self):
        current_path = Path(".")
        output_path = current_path.joinpath(self.config.name)
        if not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=True)

        before_time = datetime.utcnow()
        print(
            f"Started benchmark with {self.parallelism} workers for {self.config.benchmark_time_sec} seconds"
        )
        pool = ProcessPoolExecutor(self.parallelism)
        workers = []
        for i in range(self.parallelism):
            worker = pool.submit(self.worker, i + 1)
            workers.append(worker)
        worker_results: List[WorkerResult] = [worker.result() for worker in workers]
        total_time_sec = (datetime.utcnow() - before_time).total_seconds()
        pool.shutdown(wait=True, cancel_futures=True)

        print(f"Total time: {total_time_sec} seconds")
        dst_file = output_path.joinpath(f"res_{self.parallelism}.json")
        with dst_file.open("w") as f:
            total_request_times = []
            for res in worker_results:
                total_request_times.extend(res.req_times)

            summary = {
                "benchmark_name": self.config.name,
                "benchmark_time": total_time_sec,
                "parallelism": self.parallelism,
                "avg_tps": sum([res.tps for res in worker_results])
                / len(worker_results),
                "total_tps": sum([res.tps for res in worker_results]),
                "total_requests": sum([res.num_of_requests for res in worker_results]),
                "total_input_tokens": sum(
                    [res.total_input_tokens for res in worker_results]
                ),
                "total_output_tokens": sum(
                    [res.total_output_tokens for res in worker_results]
                ),
                "input_tokens_min": sum(
                    [res.input_tokens_per_min for res in worker_results]
                ),
                "output_tokens_min": sum(
                    [res.output_tokens_per_min for res in worker_results]
                ),
            }
            json.dump(summary, f, indent=2)
        print("Summary:")
        print(json.dumps(summary, indent=2))
        return summary

    def _completion_request(self, client: OpenAI, debug_print=False):
        before = datetime.utcnow()
        completion = client.chat.completions.create(
            timeout=self.config.request_timeout,
            model=self.config.model_id,
            messages=self.prompt,
        )
        total_time = (datetime.utcnow() - before).total_seconds()

        if debug_print:
            print(completion.choices[0].message.content)

        return (
            total_time,
            completion.usage.completion_tokens,
            completion.usage.prompt_tokens,
        )

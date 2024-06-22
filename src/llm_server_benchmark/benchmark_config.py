from typing import List, Optional
from pydantic import BaseModel, ConfigDict


class BenchmarkConfig(BaseModel):
    model_config = ConfigDict(
            protected_namespaces=()
        )

    name: str
    base_url: Optional[str] = None
    api_key: str
    model_id: str
    prompt_path: str = "prompt.json"
    benchmark_time_sec: int = 30
    request_timeout: int = 30


class BenchmarkRoot(BaseModel):
    benchmarks: List[BenchmarkConfig] = []

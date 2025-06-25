from collections import deque
from typing import Optional

from vidur.config import SyntheticRequestGeneratorConfig
from vidur.entities import Request
from vidur.request_generator.base_request_generator import BaseRequestGenerator
from vidur.request_generator.request_interval_generator_registry import (
    RequestIntervalGeneratorRegistry,
)
from vidur.request_generator.request_length_generator_registry import (
    RequestLengthGeneratorRegistry,
)


class SyntheticRequestGenerator(BaseRequestGenerator):
    def __init__(self, config: SyntheticRequestGeneratorConfig):
        super().__init__(config)

        self.request_length_generator = RequestLengthGeneratorRegistry.get(
            self._config.length_generator_config.get_type(),
            self._config.length_generator_config,
            self._random_number_generator,
        )
        self.request_interval_generator = RequestIntervalGeneratorRegistry.get(
            self._config.interval_generator_config.get_type(),
            self._config.interval_generator_config,
            self._random_number_generator,
        )
        self.requests = deque()
        self.last_arrived_at = 0
        self.num_requests_generated = 0

    # Attempt to generate a new request and append it to the queue of requests
    def _generate_next_request(self) -> None:
        if self._config.num_requests is not None:
            if self.num_requests_generated >= self._config.num_requests:
                return

        if self._config.duration is not None:
            if self.last_arrived_at >= self._config.duration:
                return

        inter_request_time = (
            self.request_interval_generator.get_next_inter_request_time()
        )
        assert isinstance(inter_request_time, float)
        arrived_at = self.last_arrived_at + inter_request_time
        request_length_output = self.request_length_generator.get_next_num_tokens()

        self.last_arrived_at = arrived_at
        self.num_requests_generated += 1
        self.requests.append(
            Request(
                arrived_at=arrived_at,
                num_prefill_tokens=request_length_output.num_prefill_tokens,
                num_decode_tokens=request_length_output.num_decode_tokens,
                block_hash_ids=request_length_output.block_hash_ids,
                block_size=request_length_output.block_size,
                session_id=request_length_output.session_id,
            )
        )

    def get_next_request_arrival_time(self) -> Optional[float]:
        if len(self.requests) == 0:
            self._generate_next_request()

        return self.requests[0].arrived_at if len(self.requests) > 0 else None

    def get_next_request(self) -> Optional[Request]:
        if len(self.requests) == 0:
            self._generate_next_request()

        return self.requests.popleft() if len(self.requests) > 0 else None

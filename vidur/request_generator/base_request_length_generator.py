from abc import ABC, abstractmethod
from typing import Generator, List, Optional

from vidur.config import BaseRequestLengthGeneratorConfig


class RequestLengthGeneratorOutput:
    num_prefill_tokens: int
    num_decode_tokens: int
    block_hash_ids: Optional[List[int]]
    block_size: Optional[int]
    session_id: Optional[int]

    def __init__(
        self,
        num_prefill_tokens: int,
        num_decode_tokens: int,
        block_hash_ids: Optional[List[int]],
        block_size: Optional[int],
        session_id: Optional[int],
    ):
        self.num_prefill_tokens = int(num_prefill_tokens)
        self.num_decode_tokens = int(num_decode_tokens)
        self.block_hash_ids = block_hash_ids
        self.block_size = block_size
        self.session_id = session_id


class BaseRequestLengthGenerator(ABC):
    def __init__(
        self,
        config: BaseRequestLengthGeneratorConfig,
        random_number_generator: Generator,
    ):
        self._config = config
        self._random_number_generator = random_number_generator

    @abstractmethod
    def get_next_num_tokens(self) -> RequestLengthGeneratorOutput:
        pass

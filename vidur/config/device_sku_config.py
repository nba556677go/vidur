from dataclasses import dataclass, field

from vidur.config.base_fixed_config import BaseFixedConfig
from vidur.logger import init_logger
from vidur.types import DeviceSKUType

logger = init_logger(__name__)


@dataclass
class BaseDeviceSKUConfig(BaseFixedConfig):
    fp16_tflops: int
    total_memory_gb: int


@dataclass
class A40DeviceSKUConfig(BaseDeviceSKUConfig):
    fp16_tflops: int = 150
    total_memory_gb: int = 45

    @staticmethod
    def get_type():
        return DeviceSKUType.A40


@dataclass
class A100DeviceSKUConfig(BaseDeviceSKUConfig):
    fp16_tflops: int = 312
    total_memory_gb: int = 80

    @staticmethod
    def get_type():
        return DeviceSKUType.A100


@dataclass
class H100DeviceSKUConfig(BaseDeviceSKUConfig):
    fp16_tflops: int = 1000
    total_memory_gb: int = 80

    @staticmethod
    def get_type():
        return DeviceSKUType.H100


@dataclass
class L40S_G6E48DeviceSKUConfig(BaseDeviceSKUConfig):
    fp16_tflops: int = 362
    total_memory_gb: int = 48

    @staticmethod
    def get_type():
        return DeviceSKUType.L40S_G6E48


@dataclass
class H100_P5DeviceSKUConfig(BaseDeviceSKUConfig):
    fp16_tflops: int = 1000
    total_memory_gb: int = 80

    @staticmethod
    def get_type():
        return DeviceSKUType.H100_P5


@dataclass
class L4_G6DeviceSKUConfig(BaseDeviceSKUConfig):
    fp16_tflops: int = 121
    total_memory_gb: int = 24

    @staticmethod
    def get_type():
        return DeviceSKUType.L4_G6


@dataclass
class A10G_G5DeviceSKUConfig(BaseDeviceSKUConfig):
    fp16_tflops: int = 125
    total_memory_gb: int = 24

    @staticmethod
    def get_type():
        return DeviceSKUType.A10G_G5

@dataclass
class A100_P4DDeviceSKUConfig(BaseDeviceSKUConfig):
    fp16_tflops: int = 312
    total_memory_gb: int = 40

    @staticmethod
    def get_type():
        return DeviceSKUType.A100_P4D
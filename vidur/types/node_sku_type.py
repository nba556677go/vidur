from vidur.types.base_int_enum import BaseIntEnum


class NodeSKUType(BaseIntEnum):
    A40_PAIRWISE_NVLINK = 1
    A100_PAIRWISE_NVLINK = 2
    H100_PAIRWISE_NVLINK = 3
    A100_DGX = 4
    H100_DGX = 5
    P4D_A100_40G_NVLINK = 6
    L40S_G6E48 = 7
    H100_P5 = 8
    L4_G6 = 9
    A10G_G5 = 10

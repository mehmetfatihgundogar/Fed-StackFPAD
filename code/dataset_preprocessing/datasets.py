# --------------------------------------------------------
# Fed-StackFPAD: Federated Learning for Face Presentation Attack Detection with Stacking to Tackle Data Heterogeneity
#
# Author: Mehmet Fatih Gündoğar
#
# Description:
# This file is part of the Fed-StackFPAD framework, a federated learning pipeline  designed for robust face presentation
# attack detection (FPAD). It combines self-supervised pretrained ViT encoders with local fine-tuning and a final
# stacking-based ensemble classifier.
#
# --------------------------------------------------------

from enum import Enum
from typing import Dict


class Datasets(Enum):
    ReplayAttack = 1
    MsuMfsd = 2
    OuluNpu = 3
    CasiaFasd = 4


class DatasetMap(Dict):
    str_to_str_code: Dict = {'replayattack': 'I', 'msumfsd': 'M', 'oulunpu': 'O', 'casiafasd': 'C'}
    str_to_num_code: Dict = {'replayattack': 1, 'msumfsd': 2, 'oulunpu': 3, 'casiafasd': 4}
    str_code_to_num_code: Dict = {'I': 1, 'M': 2, 'O': 3, 'C': 4}
    num_to_str_code: Dict = {1: 'I', 2: 'M', 3: 'O', 4: 'C'}
    num_to_str: Dict = {1: 'replayattack', 2: 'msumfsd', 3: 'oulunpu', 4: 'casiafasd'}


prediction_index_map = {
    ("O", "M"): 1,
    ("C", "M"): 2,
    ("I", "M"): 3,
    ("IOC", "M"): 4,
    ("OCI", "M"): 5,

    ("O", "C"): 1,
    ("M", "C"): 2,
    ("I", "C"): 3,
    ("IMO", "C"): 4,
    ("OMI", "C"): 5,

    ("O", "I"): 1,
    ("C", "I"): 2,
    ("M", "I"): 3,
    ("MOC", "I"): 4,
    ("OCM", "I"): 5,

    ("M", "O"): 1,
    ("C", "O"): 2,
    ("I", "O"): 3,
    ("IMC", "O"): 4,
    ("ICM", "O"): 5,
}
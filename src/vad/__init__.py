"""
VAD (Video Anomaly Detection) 모델 통합 인터페이스
=================================================

지원 모델:
- MNAD: Memory-guided Normality AutoEncoder
- MemAE: Memory-augmented AutoEncoder
- STAE: Spatio-Temporal AutoEncoder
- STEAD: Spatio-Temporal Efficient Anomaly Detection

모든 모델은 실제 학습된 체크포인트를 사용합니다.
"""

from .base import VADModel
from .mnad import MNADModel
from .memae import MemAEModel
from .stae import STAEModel
from .stead import STEADModel

# Attribute-based VAD는 선택적 import (의존성 문제 가능)
try:
    from .attribute_based import AttributeBasedVAD
    HAS_ATTRIBUTE_VAD = True
except ImportError as e:
    HAS_ATTRIBUTE_VAD = False
    AttributeBasedVAD = None

# Attribute-based VAD (AiVAD 활용)
try:
    from .attribute_based_aivad import AttributeBasedVADAiVAD
    HAS_ATTRIBUTE_VAD_AIVAD = True
except ImportError as e:
    HAS_ATTRIBUTE_VAD_AIVAD = False
    AttributeBasedVADAiVAD = None


def create_model(model_name: str, **kwargs) -> VADModel:
    """
    VAD 모델 생성 팩토리
    
    Args:
        model_name: 모델 이름 (mnad, memae, stae, stead, attribute_based)
        **kwargs: 모델별 추가 인자
    
    Returns:
        VADModel 인스턴스
    """
    models = {
        'mnad': MNADModel,
        'memae': MemAEModel,
        'stae': STAEModel,
        'stead': STEADModel,
    }
    
    # Attribute-based VAD 추가
    if HAS_ATTRIBUTE_VAD:
        models['attribute_based'] = AttributeBasedVAD
    if HAS_ATTRIBUTE_VAD_AIVAD:
        models['attribute_based_aivad'] = AttributeBasedVADAiVAD
    
    model_name = model_name.lower()
    if model_name not in models:
        available = list(models.keys())
        raise ValueError(f"Unknown model: {model_name}. Available: {available}")
    
    return models[model_name](**kwargs)


__all__ = [
    'VADModel',
    'MNADModel',
    'MemAEModel',
    'STAEModel',
    'STEADModel',
    'create_model',
]

if HAS_ATTRIBUTE_VAD:
    __all__.append('AttributeBasedVAD')
if HAS_ATTRIBUTE_VAD_AIVAD:
    __all__.append('AttributeBasedVADAiVAD')




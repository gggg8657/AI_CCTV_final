"""
Agent System
============

LLM 기반 자동 대응 에이전트 시스템

지원 Flow:
- Sequential: VideoAnalysis → Planner → Actor
- Hierarchical: VideoAnalysis → Supervisor → Planner → Actor
- Collaborative: VideoAnalysis → Multiple Planners → Aggregator → Actor

모든 추론은 실제 LLM (Qwen3-8B)으로 수행됩니다.
"""

from .base import (
    LLMManager,
    VideoAnalysisAgent,
    PlannerAgent,
    SupervisorAgent,
    ActorAgent,
    AVAILABLE_ACTIONS,
    VideoAnalysisState,
)
from .flows import (
    SequentialFlow,
    HierarchicalFlow,
    CollaborativeFlow,
)


def create_flow(flow_name: str, config: dict = None, **kwargs):
    """
    Agent Flow 생성 팩토리
    
    Args:
        flow_name: Flow 이름 (sequential, hierarchical, collaborative)
        **kwargs: Flow별 추가 인자
    
    Returns:
        Flow 인스턴스
    """
    flows = {
        'sequential': SequentialFlow,
        'hierarchical': HierarchicalFlow,
        'collaborative': CollaborativeFlow,
    }
    
    flow_name = flow_name.lower()
    if flow_name not in flows:
        raise ValueError(f"Unknown flow: {flow_name}. Available: {list(flows.keys())}")
    
    return flows[flow_name](config=config, **kwargs)


__all__ = [
    'LLMManager',
    'VideoAnalysisAgent',
    'PlannerAgent',
    'SupervisorAgent',
    'ActorAgent',
    'AVAILABLE_ACTIONS',
    'VideoAnalysisState',
    'SequentialFlow',
    'HierarchicalFlow',
    'CollaborativeFlow',
    'create_flow',
]




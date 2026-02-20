"""
DummyAgentFlow — 실제 LLM 없이 Agent Flow 인터페이스를 구현
=============================================================

SequentialFlow/HierarchicalFlow/CollaborativeFlow와 동일한
initialize() / run(video_path) 인터페이스를 제공.
"""

import random
import time
from typing import Dict, Any, List, Optional


SCENARIO_RESPONSES: Dict[str, Dict[str, Any]] = {
    "Fighting": {
        "situation_type": "Fighting",
        "severity_level": "High",
        "actions": [
            {"action": "alert_security", "priority": "high", "description": "Dispatch security team to location"},
            {"action": "dispatch_guard", "priority": "medium", "description": "Send nearest guard to intervene"},
        ],
    },
    "Arson": {
        "situation_type": "Arson",
        "severity_level": "Critical",
        "actions": [
            {"action": "alert_fire_dept", "priority": "critical", "description": "Call fire department immediately"},
            {"action": "evacuate", "priority": "high", "description": "Begin evacuation procedure"},
        ],
    },
    "Falling": {
        "situation_type": "Falling",
        "severity_level": "High",
        "actions": [
            {"action": "alert_medical", "priority": "high", "description": "Call medical emergency"},
            {"action": "check_status", "priority": "high", "description": "Check victim condition"},
        ],
    },
    "default": {
        "situation_type": "Unknown",
        "severity_level": "Low",
        "actions": [
            {"action": "monitor", "priority": "low", "description": "Continue monitoring"},
        ],
    },
}


class DummyAgentFlow:
    """실제 LLM 없이 Agent Flow를 시뮬레이션"""

    def __init__(self, flow_type: str = "sequential", **kwargs: Any):
        self.flow_type = flow_type
        self._initialized = False

    def initialize(self) -> bool:
        self._initialized = True
        return True

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def run(self, video_path: str = "", situation_type: str = "") -> Dict[str, Any]:
        start = time.time()

        if not situation_type:
            situation_type = random.choice(list(SCENARIO_RESPONSES.keys()))

        scenario = SCENARIO_RESPONSES.get(situation_type, SCENARIO_RESPONSES["default"])

        time.sleep(random.uniform(0.02, 0.08))

        return {
            "success": True,
            "flow_type": self.flow_type,
            "situation_type": scenario["situation_type"],
            "severity_level": scenario["severity_level"],
            "agent_plan": {"actions": scenario["actions"]},
            "actions": scenario["actions"],
            "processing_times": {
                "video_analysis": round(random.uniform(10, 30), 1),
                "planning": round(random.uniform(5, 15), 1),
                "execution": round(random.uniform(2, 8), 1),
            },
            "total_time_ms": round((time.time() - start) * 1000, 1),
        }

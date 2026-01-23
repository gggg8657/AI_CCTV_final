"""
Collaborative Flow
==================

협력적 처리 (다중 Planner):
VideoAnalysis → [ConservativePlanner + AggressivePlanner] → Aggregator → Actor

여러 Planner의 계획을 집계하여 최적 결정
"""

import time
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..base import (
    LLMManager,
    VideoAnalysisAgent,
    PlannerAgent,
    ActorAgent,
    VideoAnalysisState,
)
from .function_calling_support import FunctionCallingSupport


class CollaborativeFlow:
    """
    Collaborative Agent Flow
    
    VideoAnalysis → [ConservativePlanner + AggressivePlanner] → Aggregator → Actor
    
    특징:
    - 다중 Planner 병렬 실행
    - 다양한 관점의 계획 수집
    - 집계 전략으로 최종 결정
    """
    
    def __init__(
        self,
        gpu_id: int = 2,
        aggregation_strategy: str = "score",
        e2e_system=None,
        config: Optional[Dict] = None,
    ):
        """
        Args:
            gpu_id: GPU ID
            aggregation_strategy: 집계 전략 (score, conservative, aggressive, majority)
        """
        self.gpu_id = gpu_id
        self.aggregation_strategy = aggregation_strategy
        self.e2e_system = e2e_system
        self.config = config
        self.llm_manager = None
        self._initialized = False
        
        self.video_analysis_agent = None
        self.planners = []
        self.actor_agent = None
        self.function_calling = None
    
    def initialize(self) -> bool:
        """Flow 초기화"""
        if self._initialized:
            return True
        
        print("[CollaborativeFlow] 초기화 중...")
        
        self.llm_manager = LLMManager(self.config)
        
        # 모델 로드
        if not self.llm_manager.load_vision_llm(self.gpu_id):
            print("[CollaborativeFlow] Vision LLM 로드 실패")
            return False
        
        if not self.llm_manager.load_text_llm(self.gpu_id):
            print("[CollaborativeFlow] Text LLM 로드 실패")
            return False
        
        # 에이전트 생성
        self.video_analysis_agent = VideoAnalysisAgent(self.llm_manager)
        
        # 다중 Planner (Conservative, Aggressive)
        self.planners = [
            {"name": "Conservative", "agent": PlannerAgent(self.llm_manager), "weight": 0.6},
            {"name": "Aggressive", "agent": PlannerAgent(self.llm_manager), "weight": 0.4},
        ]
        
        self.actor_agent = ActorAgent(self.llm_manager)

        # Function Calling 초기화
        self.function_calling = FunctionCallingSupport(self.llm_manager, self.e2e_system)
        
        self._initialized = True
        print("[CollaborativeFlow] 초기화 완료")
        return True

    def set_e2e_system(self, e2e_system) -> None:
        """E2ESystem 연결"""
        self.e2e_system = e2e_system
        if self.function_calling is None:
            self.function_calling = FunctionCallingSupport(self.llm_manager, e2e_system)
        else:
            self.function_calling.set_e2e_system(e2e_system)

    def process_query(self, query: str, conversation: Optional[list] = None) -> Dict:
        """자연어 질의 처리 (Function Calling)"""
        if not self._initialized:
            if not self.initialize():
                return {"success": False, "error": "초기화 실패"}
        if self.function_calling is None:
            self.function_calling = FunctionCallingSupport(self.llm_manager, self.e2e_system)
        return self.function_calling.process_query(query, conversation)
    
    def run(self, video_path: str) -> Dict:
        """
        Flow 실행
        
        Args:
            video_path: 분석할 비디오 파일 경로
        
        Returns:
            실행 결과
        """
        if not self._initialized:
            if not self.initialize():
                return {"success": False, "error": "초기화 실패"}
        
        processing_times = {}
        
        # Step 1: Video Analysis
        print("\n[Step 1] Video Analysis")
        start_time = time.time()
        analysis_result = self.video_analysis_agent.analyze_video_and_classify(video_path)
        processing_times["video_analysis"] = time.time() - start_time
        
        if not analysis_result.get("success", False):
            return {
                "success": False,
                "error": analysis_result.get("error", "분석 실패"),
                "processing_times": processing_times
            }
        
        situation_type = analysis_result.get("situation_type", "정상상황")
        severity_level = analysis_result.get("severity_level", "관심")
        situation_description = analysis_result.get("final_situation_description", "")
        
        # Step 2: Parallel Planning
        print("\n[Step 2] Parallel Planning")
        start_time = time.time()
        
        all_plans = []
        for planner_info in self.planners:
            planner = planner_info["agent"]
            planner_name = planner_info["name"]
            
            print(f"  [{planner_name}] 계획 수립 중...")
            planner_result = planner.create_plan(
                situation_type, severity_level, situation_description
            )
            
            all_plans.append({
                "name": planner_name,
                "weight": planner_info["weight"],
                "plan": planner_result.get("plan", {}),
                "report": planner_result.get("report", "")
            })
        
        processing_times["parallel_planning"] = time.time() - start_time
        
        # Step 3: Aggregation
        print("\n[Step 3] Aggregation")
        start_time = time.time()
        aggregated_plan, aggregation_report = self._aggregate_plans(all_plans, self.aggregation_strategy)
        processing_times["aggregator"] = time.time() - start_time
        
        # Step 4: Actor
        print("\n[Step 4] Actor")
        start_time = time.time()
        actor_result = self.actor_agent.execute_plan(aggregated_plan)
        processing_times["actor"] = time.time() - start_time
        
        return {
            "success": True,
            "situation_type": situation_type,
            "severity_level": severity_level,
            "final_situation_description": situation_description,
            "planner_report": aggregation_report,
            "agent_plan": aggregated_plan,
            "all_plans": all_plans,
            "actor_execution_results": actor_result.get("execution_results", []),
            "processing_times": processing_times
        }
    
    def _aggregate_plans(self, plans: List[Dict], strategy: str) -> tuple:
        """
        계획 집계
        
        Args:
            plans: 각 Planner의 계획 리스트
            strategy: 집계 전략
        
        Returns:
            (aggregated_plan, report)
        """
        if not plans:
            return {}, "계획 없음"
        
        if strategy == "conservative":
            # 가장 보수적인 계획 선택 (가장 적은 액션)
            selected = min(plans, key=lambda p: len(p["plan"].get("actions", [])))
            return selected["plan"], f"[집계] conservative 전략으로 {selected['name']} 선택"
        
        elif strategy == "aggressive":
            # 가장 적극적인 계획 선택 (가장 많은 액션)
            selected = max(plans, key=lambda p: len(p["plan"].get("actions", [])))
            return selected["plan"], f"[집계] aggressive 전략으로 {selected['name']} 선택"
        
        elif strategy == "majority":
            # 가장 많이 등장하는 액션들 선택
            action_counts = {}
            for plan_info in plans:
                for action in plan_info["plan"].get("actions", []):
                    action_name = action.get("name", "")
                    if action_name not in action_counts:
                        action_counts[action_name] = {"count": 0, "action": action}
                    action_counts[action_name]["count"] += 1
            
            # 과반수 이상 등장한 액션들
            threshold = len(plans) / 2
            selected_actions = [
                v["action"] for k, v in action_counts.items()
                if v["count"] >= threshold
            ]
            
            merged_plan = {
                "situation_type": plans[0]["plan"].get("situation_type", ""),
                "severity_level": plans[0]["plan"].get("severity_level", ""),
                "actions": selected_actions,
            }
            return merged_plan, f"[집계] majority 전략으로 {len(selected_actions)}개 액션 선택"
        
        else:  # score (weighted)
            # 가중 점수 기반 선택
            best_plan = None
            best_score = -1
            
            for plan_info in plans:
                action_count = len(plan_info["plan"].get("actions", []))
                score = action_count * plan_info["weight"]
                if score > best_score:
                    best_score = score
                    best_plan = plan_info
            
            if best_plan:
                return best_plan["plan"], f"[집계] score 전략으로 {len(best_plan['plan'].get('actions', []))}개 액션 선택"
            else:
                return plans[0]["plan"], "[집계] 기본 선택"




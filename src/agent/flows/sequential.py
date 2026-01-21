"""
Sequential Flow
===============

단순 순차 처리:
VideoAnalysis → Planner → Actor

가장 단순한 Flow, 중간 검토 없음
"""

import time
from typing import Dict, Optional

from ..base import (
    LLMManager,
    VideoAnalysisAgent,
    PlannerAgent,
    ActorAgent,
    VideoAnalysisState,
)


class SequentialFlow:
    """
    Sequential Agent Flow
    
    VideoAnalysis → Planner → Actor
    
    특징:
    - 단순 순차 처리
    - 검토/재시도 없음
    - 가장 빠른 처리
    """
    
    def __init__(self, gpu_id: int = 2):
        self.gpu_id = gpu_id
        self.llm_manager = None
        self._initialized = False
        
        self.video_analysis_agent = None
        self.planner_agent = None
        self.actor_agent = None
    
    def initialize(self) -> bool:
        """Flow 초기화"""
        if self._initialized:
            return True
        
        print("[SequentialFlow] 초기화 중...")
        
        self.llm_manager = LLMManager()
        
        # 모델 로드
        if not self.llm_manager.load_vision_llm(self.gpu_id):
            print("[SequentialFlow] Vision LLM 로드 실패")
            return False
        
        if not self.llm_manager.load_text_llm(self.gpu_id):
            print("[SequentialFlow] Text LLM 로드 실패")
            return False
        
        # 에이전트 생성
        self.video_analysis_agent = VideoAnalysisAgent(self.llm_manager)
        self.planner_agent = PlannerAgent(self.llm_manager)
        self.actor_agent = ActorAgent(self.llm_manager)
        
        self._initialized = True
        print("[SequentialFlow] 초기화 완료")
        return True
    
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
        
        # Step 2: Planner
        print("\n[Step 2] Planner")
        start_time = time.time()
        planner_result = self.planner_agent.create_plan(
            situation_type, severity_level, situation_description
        )
        processing_times["planner"] = time.time() - start_time
        
        plan = planner_result.get("plan", {})
        planner_report = planner_result.get("report", "")
        
        # Step 3: Actor
        print("\n[Step 3] Actor")
        start_time = time.time()
        actor_result = self.actor_agent.execute_plan(plan)
        processing_times["actor"] = time.time() - start_time
        
        return {
            "success": True,
            "situation_type": situation_type,
            "severity_level": severity_level,
            "final_situation_description": situation_description,
            "planner_report": planner_report,
            "agent_plan": plan,
            "actor_execution_results": actor_result.get("execution_results", []),
            "processing_times": processing_times
        }




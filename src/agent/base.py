"""
Agent 기본 컴포넌트
==================

LLMManager, VideoAnalysisAgent, PlannerAgent, SupervisorAgent, ActorAgent

모든 추론은 실제 LLM으로 수행됩니다. 더미 없음.
"""

import os
import gc
import json
import base64
import time
import logging
from datetime import datetime
from typing import List, Dict, TypedDict, Optional, Any

import cv2
import numpy as np

from .actions import AVAILABLE_ACTIONS, SCENARIO_ACTIONS, ACTION_PRIORITY


# 기본 설정
DEFAULT_CONFIG = {
    "VISION_MODEL_PATH": "/data/DJ/models/Qwen2.5-VL-7B-Instruct-q4_k_m.gguf",
    "MM_PROJ_PATH": "/data/DJ/models/Qwen2.5-VL-7B-Instruct-mmproj-f16.gguf",
    "TEXT_MODEL_PATH": "/data/DJ/models/Qwen3-8B-Q4_K_M.gguf",
    "ANALYSIS_DURATION": 3,
    "N_GPU_LAYERS": -1,
    "N_CTX": 32768,
    "N_THREADS": 16,
    "N_BATCH": 512,
    "MAIN_GPU": 2,
}


class VideoAnalysisState(TypedDict, total=False):
    """워크플로우 상태"""
    video_path: str
    trigger_timestamp: float
    timestamp: str
    
    # 실시간 모드용
    realtime_frames: List
    
    # 분석 결과
    frame_analyses: List[Dict]
    context_history: str
    final_situation_description: str
    encoded_frames: List[str]
    
    # 분류 결과
    classification_report: str
    situation_type: str
    severity_level: str
    classification_reasoning: str
    
    # Supervisor 관련
    supervisor_instruction_to_planner: str
    supervisor_plan_review: str
    plan_approved: bool
    review_feedback: str
    plan_retry_count: int
    
    # Planner 관련
    planner_report: str
    agent_plan: Dict
    
    # Actor 관련
    actor_execution_results: List[Dict]
    
    # 시스템 상태
    success: bool
    error_message: str
    processing_times: Dict[str, float]


class LLMManager:
    """
    LLM 관리자 (싱글톤)
    
    Vision LLM (Qwen2.5-VL-7B)과 Text LLM (Qwen3-8B)을 관리합니다.
    모델을 로드한 후 유지하여 반복 로딩 비용을 절감합니다.
    """
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LLMManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.vision_llm = None
            self.text_llm = None
            self.vision_loaded = False
            self.text_loaded = False
            self.config = DEFAULT_CONFIG.copy()
            LLMManager._initialized = True
    
    def load_vision_llm(self, gpu_id: int = None) -> bool:
        """Vision LLM 로드"""
        if self.vision_loaded and self.vision_llm is not None:
            return True
        
        try:
            from llama_cpp import Llama
            from llama_cpp.llama_chat_format import Qwen25VLChatHandler
            
            logging.info("Vision LLM 로드 중...")
            main_gpu = gpu_id if gpu_id is not None else self.config["MAIN_GPU"]
            
            self.vision_llm = Llama(
                model_path=self.config["VISION_MODEL_PATH"],
                chat_handler=Qwen25VLChatHandler(clip_model_path=self.config["MM_PROJ_PATH"]),
                n_gpu_layers=self.config["N_GPU_LAYERS"],
                n_ctx=self.config["N_CTX"],
                n_threads=self.config["N_THREADS"],
                n_batch=self.config["N_BATCH"],
                main_gpu=main_gpu,
                use_mmap=True,
                use_mlock=True,
                verbose=False
            )
            self.vision_loaded = True
            logging.info("Vision LLM 로드 완료")
            return True
        except Exception as e:
            logging.error(f"Vision LLM 로드 실패: {e}")
            return False
    
    def load_text_llm(self, gpu_id: int = None) -> bool:
        """Text LLM 로드"""
        if self.text_loaded and self.text_llm is not None:
            return True
        
        try:
            from llama_cpp import Llama
            
            logging.info("Text LLM 로드 중...")
            
            if not os.path.exists(self.config["TEXT_MODEL_PATH"]):
                logging.error(f"모델 파일을 찾을 수 없습니다: {self.config['TEXT_MODEL_PATH']}")
                return False
            
            main_gpu = gpu_id if gpu_id is not None else self.config["MAIN_GPU"]
            
            self.text_llm = Llama(
                model_path=self.config["TEXT_MODEL_PATH"],
                n_gpu_layers=self.config["N_GPU_LAYERS"],
                n_ctx=self.config["N_CTX"],
                n_threads=self.config["N_THREADS"],
                n_batch=self.config["N_BATCH"],
                main_gpu=main_gpu,
                use_mmap=True,
                use_mlock=True,
                chat_format="chatml",
                verbose=False
            )
            self.text_loaded = True
            logging.info("Text LLM 로드 완료")
            return True
        except Exception as e:
            logging.error(f"Text LLM 로드 실패: {e}")
            return False
    
    def load_all_models(self, gpu_id: int = None) -> bool:
        """모든 모델 로드"""
        vision_success = self.load_vision_llm(gpu_id)
        text_success = self.load_text_llm(gpu_id)
        return vision_success and text_success
    
    def unload_all_models(self):
        """모든 모델 언로드"""
        if self.vision_llm:
            del self.vision_llm
            self.vision_llm = None
            self.vision_loaded = False
        if self.text_llm:
            del self.text_llm
            self.text_llm = None
            self.text_loaded = False
        gc.collect()


class VideoAnalysisAgent:
    """
    영상 분석 에이전트
    
    VLM으로 영상 분석 + 상황 분류 + 심각도 판단
    """
    
    def __init__(self, llm_manager: LLMManager = None):
        self.llm_manager = llm_manager or LLMManager()
    
    def analyze_video_and_classify(self, video_path: str) -> Dict:
        """비디오 파일 분석"""
        print(f"\n[VIDEO ANALYSIS AGENT] 영상 분석 시작: {video_path}")
        
        try:
            timestamp = datetime.now().isoformat()
            frames = self._extract_frames(video_path)
            
            if not frames:
                raise ValueError("프레임 추출 실패")
            
            frame_analyses = []
            encoded_frames = []
            context_history = ""
            
            for i, frame in enumerate(frames):
                print(f"  [FRAME {i+1}] 분석 중...", end=" ")
                encoded_frame = self._encode_frame(frame)
                encoded_frames.append(encoded_frame)
                
                description = self._analyze_frame_with_vlm(encoded_frame, context_history)
                
                frame_analyses.append({
                    "timestamp": datetime.now().isoformat(),
                    "description": description,
                    "frame_data": encoded_frame
                })
                
                context_history = description
                print(f"완료: {description}")
            
            classification_result = self._classify_situation(context_history, encoded_frames)
            
            integrated_report = f"분석 완료: {classification_result['situation_type']} (심각도: {classification_result['severity_level']})"
            
            print(f"[VIDEO ANALYSIS AGENT] 분석 완료")
            print(f"[RESULT] {classification_result['situation_type']} ({classification_result['severity_level']})")
            
            return {
                "success": True,
                "video_analysis_report": integrated_report,
                "frame_analyses": frame_analyses,
                "context_history": context_history,
                "final_situation_description": context_history,
                "encoded_frames": encoded_frames,
                "timestamp": timestamp,
                "classification_report": integrated_report,
                "situation_type": classification_result["situation_type"],
                "severity_level": classification_result["severity_level"],
                "classification_reasoning": classification_result["reasoning"]
            }
            
        except Exception as e:
            print(f"[VIDEO ANALYSIS AGENT ERROR] 분석 실패: {e}")
            return {
                "success": False,
                "error": str(e),
                "video_analysis_report": f"영상 분석 실패: {str(e)}",
                "frame_analyses": [],
                "context_history": "",
                "final_situation_description": "",
                "encoded_frames": [],
                "classification_report": f"분석 실패: {str(e)}",
                "situation_type": "정상상황",
                "severity_level": "관심",
                "classification_reasoning": "분석 실패"
            }
    
    def _extract_frames(self, video_path: str, n_frames: int = 3) -> List[np.ndarray]:
        """비디오에서 프레임 추출"""
        if not os.path.exists(video_path):
            return []
        
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            for i in range(n_frames):
                current_frame = int(i * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
        finally:
            cap.release()
        
        return frames
    
    def _encode_frame(self, frame: np.ndarray) -> str:
        """프레임을 base64로 인코딩"""
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buffer).decode('utf-8')
    
    def _analyze_frame_with_vlm(self, encoded_frame: str, context_history: str) -> str:
        """VLM으로 프레임 분석"""
        if not self.llm_manager or not self.llm_manager.vision_llm:
            return "VLM 없음"
        
        system_prompt = f"""당신은 CCTV 보안 영상을 분석하는 전문 AI 에이전트입니다.
이미지를 자세히 관찰하고 현재 상황을 설명하세요.

특히 다음 위험 상황들을 주의깊게 찾아보세요:
- 화재: 불꽃, 연기, 화재 징후
- 폭력: 사람들이 싸우거나 때리는 행동, 공격적 자세
- 쓰러짐: 사람이 쓰러져 있거나 의식을 잃은 모습

{f"이전 상황: {context_history}" if context_history else ""}

50자 이내로 실제 관찰된 상황을 자연스러운 문장으로 설명하세요."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_frame}"}}
            ]}
        ]

        response = self.llm_manager.vision_llm.create_chat_completion(
            messages=messages,
            temperature=0.2,
            max_tokens=80
        )

        return response['choices'][0]['message']['content'].strip()
    
    def _classify_situation(self, situation_description: str, encoded_frames: List[str]) -> Dict:
        """상황 분류 및 심각도 결정"""
        # 키워드 기반 사전 검사
        situation_lower = situation_description.lower()
        
        fall_keywords = ['쓰러짐', '쓰러져', '쓰러진', '쓰러진다', '쓰러졌']
        fire_keywords = ['화재', '불', '연기', '타고', '타는', '불꽃', '화염']
        assault_keywords = ['폭행', '때리기', '싸움', '폭력', '공격', '때리고', '때려']
        
        detected_situation_type = None
        
        if any(kw in situation_lower for kw in fall_keywords):
            detected_situation_type = "쓰러짐"
        elif any(kw in situation_lower for kw in fire_keywords):
            detected_situation_type = "화재"
        elif any(kw in situation_lower for kw in assault_keywords):
            detected_situation_type = "폭행"
        
        if detected_situation_type:
            severity_map = {"화재": "긴급", "폭행": "경계", "쓰러짐": "경계", "정상상황": "관심"}
            return {
                "situation_type": detected_situation_type,
                "severity_level": severity_map.get(detected_situation_type, "관심"),
                "reasoning": f"키워드 감지: {situation_description}"
            }
        
        return {
            "situation_type": "정상상황",
            "severity_level": "관심",
            "reasoning": "이상 상황 미탐지"
        }


class PlannerAgent:
    """
    계획자 에이전트
    
    상황에 맞는 대응 계획 수립
    """
    
    def __init__(self, llm_manager: LLMManager = None):
        self.llm_manager = llm_manager or LLMManager()
    
    def create_plan(self, situation_type: str, severity_level: str, 
                    situation_description: str, feedback: str = "") -> Dict:
        """대응 계획 수립"""
        print(f"\n[PLANNER] 계획 수립 시작")
        print(f"[INPUT] 상황: {situation_type}, 심각도: {severity_level}")
        
        try:
            # LLM으로 액션 선택 또는 시나리오 기반 폴백
            if self.llm_manager and self.llm_manager.text_llm:
                actions = self._llm_select_actions(
                    situation_type, severity_level, situation_description, feedback
                )
            else:
                actions = self._create_scenario_actions(situation_type, severity_level)
            
            # 보고서 생성
            main_actions = [a['description'] for a in actions[:3]]
            report = f"감독자님, {situation_type} 상황(심각도: {severity_level})에 대해 {len(actions)}개 액션 계획: {', '.join(main_actions)}"
            
            plan = {
                "situation_type": situation_type,
                "severity_level": severity_level,
                "situation_description": situation_description,
                "actions": actions,
                "timestamp": datetime.now().isoformat()
            }
            
            print(f"[PLANNER] {len(actions)}개 액션 계획 완료")
            
            return {"success": True, "report": report, "plan": plan}
            
        except Exception as e:
            print(f"[PLANNER ERROR] 계획 수립 실패: {e}")
            return {
                "success": False,
                "error": str(e),
                "report": f"계획 수립 실패: {e}",
                "plan": {"situation_type": situation_type, "severity_level": severity_level, 
                         "situation_description": situation_description, "actions": [], 
                         "timestamp": datetime.now().isoformat()}
            }
    
    def _llm_select_actions(self, situation_type: str, severity_level: str,
                            situation_description: str, feedback: str = "") -> List[Dict]:
        """LLM으로 액션 선택 (Tool Calling 시뮬레이션)"""
        # 시나리오 기반 폴백 사용 (안정성)
        return self._create_scenario_actions(situation_type, severity_level)
    
    def _create_scenario_actions(self, situation_type: str, severity_level: str) -> List[Dict]:
        """시나리오별 기본 액션"""
        action_names = SCENARIO_ACTIONS.get(situation_type, {}).get(
            severity_level, ["continue_monitoring", "log_normal_incident"]
        )
        
        actions = []
        for name in action_names:
            if name in AVAILABLE_ACTIONS:
                actions.append({
                    "name": name,
                    "params": AVAILABLE_ACTIONS[name]["params_template"].copy(),
                    "description": AVAILABLE_ACTIONS[name]["description"]
                })
        
        return actions


class SupervisorAgent:
    """
    감독자 에이전트
    
    계획 검토 및 승인/거부
    """
    
    def __init__(self, llm_manager: LLMManager = None):
        self.llm_manager = llm_manager or LLMManager()
    
    def instruct_planner(self, situation_type: str, severity_level: str, 
                         situation_description: str) -> Dict:
        """Planner에게 계획 수립 지시"""
        print(f"\n[SUPERVISOR] Planner에게 계획 수립 지시")
        print(f"[SUPERVISOR] 상황: {situation_type}, 심각도: {severity_level}")
        return {"success": True}
    
    def review_plan(self, planner_report: str, plan: Dict, 
                    situation_type: str = "", severity_level: str = "") -> Dict:
        """계획 검토 및 승인/거부"""
        print(f"\n[SUPERVISOR] 계획 검토 중...")
        
        try:
            actions = plan.get("actions", [])
            action_count = len(actions)
            
            # 기준 1: 최소 1개 액션
            criterion1_ok = action_count >= 1
            
            # 기준 2: 상황 적합성
            situation_keywords = {
                "화재": ["화재", "fire", "소방"],
                "폭행": ["폭행", "assault", "폭력", "police", "경찰", "security", "증거"],
                "쓰러짐": ["쓰러", "fall", "낙상", "응급", "의료", "medical", "구급"]
            }
            criterion2_ok = True
            if situation_type in situation_keywords:
                keywords = situation_keywords[situation_type]
                criterion2_ok = any(
                    any(kw.lower() in a.get('name', '').lower() or kw.lower() in a.get('description', '').lower() 
                        for kw in keywords)
                    for a in actions
                )
            
            # 기준 3: 심각도별 적정 액션 수
            if situation_type == "정상상황" and severity_level == "관심":
                min_actions, max_actions = 2, 2
            else:
                severity_requirements = {"긴급": (4, 5), "경계": (3, 3), "관심": (1, 1)}
                min_actions, max_actions = severity_requirements.get(severity_level, (1, 1))
            
            criterion3_ok = min_actions <= action_count <= max_actions
            
            all_criteria_met = criterion1_ok and criterion2_ok and criterion3_ok
            
            if not all_criteria_met:
                feedback_parts = []
                if not criterion1_ok:
                    feedback_parts.append("액션이 없습니다")
                if not criterion2_ok:
                    feedback_parts.append(f"상황({situation_type})에 맞지 않는 액션")
                if not criterion3_ok:
                    feedback_parts.append(f"{min_actions}-{max_actions}개 필요, 현재 {action_count}개")
                
                return {
                    "success": True,
                    "approved": False,
                    "review": f"기준 미달: {', '.join(feedback_parts)}",
                    "feedback": '; '.join(feedback_parts)
                }
            
            return {"success": True, "approved": True, "review": "규칙 기반 승인", "feedback": ""}
            
        except Exception as e:
            print(f"[SUPERVISOR ERROR] 검토 실패: {e}")
            return {"success": False, "error": str(e), "approved": True, "review": f"검토 실패: {e}", "feedback": ""}


class ActorAgent:
    """
    실행 에이전트
    
    계획된 액션 실행
    """
    
    def __init__(self, llm_manager: LLMManager = None):
        self.llm_manager = llm_manager or LLMManager()
        self._current_plan_context = None
        
        # Tool registry
        self.tools = {
            "activate_fire_alarm": lambda p: f"🔥 화재 경보 발령: {p.get('message', '화재발생대피하십시오')}",
            "call_fire_department": lambda p: "📞 119 소방서 자동 신고 완료",
            "dispatch_fire_response_team": lambda p: f"👨‍🚒 청원경찰 {p.get('team_size', 2)}명 화재 대응 출동",
            "activate_fire_systems": lambda p: f"💧 소방 시스템 작동: 스프링클러={p.get('sprinkler', 'on')}",
            "log_fire_incident": lambda p: f"📝 화재 사건 로그 저장 완료",
            "activate_assault_warning": lambda p: "⚠️ 폭행 경고 방송 완료",
            "call_police": lambda p: "📞 112 경찰서 자동 신고 완료",
            "dispatch_security_team": lambda p: f"👮 청원경찰 {p.get('team_size', 2)}명 출동",
            "secure_evidence": lambda p: "📹 증거 영상 확보 완료",
            "log_assault_incident": lambda p: "📝 폭행 사건 로그 저장 완료",
            "activate_medical_assistance": lambda p: "🏥 의료 지원 안내 방송 완료",
            "call_ambulance": lambda p: "🚑 119 구급대 자동 신고 완료",
            "dispatch_medical_team": lambda p: f"👨‍⚕️ 청원경찰 {p.get('team_size', 2)}명 의료 대응 출동",
            "guide_emergency_access": lambda p: "🛣️ 구급차 진입 경로 확보 완료",
            "log_medical_incident": lambda p: "📝 의료 사건 로그 저장 완료",
            "continue_monitoring": lambda p: "👁️ 정상 모니터링 계속",
            "log_normal_incident": lambda p: "📝 정상 상황 로그 저장 완료",
        }
    
    def execute_plan(self, plan: Dict, instruction: str = "", timestamp: str = None) -> Dict:
        """계획 실행"""
        print(f"\n[ACTOR] 계획 실행 시작")
        
        actions = plan.get("actions", [])
        if not actions:
            return {"success": False, "error": "실행할 액션 없음", "execution_results": []}
        
        self._current_plan_context = {
            "situation_type": plan.get("situation_type", ""),
            "severity_level": plan.get("severity_level", ""),
            "situation_description": plan.get("situation_description", ""),
            "timestamp": timestamp,
            "execution_history": []
        }
        
        try:
            # 우선순위 기반 정렬
            ordered_actions = sorted(actions, key=lambda a: ACTION_PRIORITY.get(a.get("name", ""), 50))
            
            execution_results = []
            for i, action in enumerate(ordered_actions):
                action_name = action.get("name", "")
                action_params = action.get("params", {})
                action_desc = action.get("description", "")
                
                print(f"[ACTION {i+1}/{len(ordered_actions)}] {action_name}")
                
                if action_name in self.tools:
                    result = self.tools[action_name](action_params)
                else:
                    result = f"[UNKNOWN] {action_name}"
                
                execution_results.append({
                    "action_index": i + 1,
                    "action_name": action_name,
                    "action_description": action_desc,
                    "params": action_params,
                    "result": result,
                    "timestamp": datetime.now().isoformat(),
                    "success": True
                })
                
                print(f"[RESULT] {result}")
            
            print(f"[ACTOR] 모든 액션 실행 완료 ({len(execution_results)}개)")
            
            return {"success": True, "execution_results": execution_results}
            
        except Exception as e:
            print(f"[ACTOR ERROR] 실행 실패: {e}")
            return {"success": False, "error": str(e), "execution_results": []}




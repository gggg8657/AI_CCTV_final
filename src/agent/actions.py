"""
Agent 액션 정의
==============

보안 상황 대응을 위한 가용 액션들
"""

# 화재 관련 액션
FIRE_ACTIONS = {
    "activate_fire_alarm": {
        "description": "화재 경보 발령 및 대피 유도",
        "params_template": {"alarm_type": "fire_emergency", "message": "화재발생대피하십시오"}
    },
    "call_fire_department": {
        "description": "119 소방서 자동 신고",
        "params_template": {"service": "119", "incident_type": "fire"}
    },
    "dispatch_fire_response_team": {
        "description": "청원경찰 화재 대응 출동",
        "params_template": {"team_size": 2}
    },
    "activate_fire_systems": {
        "description": "스프링클러 및 엘리베이터 정지 시스템 작동",
        "params_template": {"sprinkler": "auto_activate", "elevator": "emergency_stop"}
    },
    "log_fire_incident": {
        "description": "화재 사건 로그 저장",
        "params_template": {"incident_type": "fire"}
    },
}

# 폭행 관련 액션
ASSAULT_ACTIONS = {
    "activate_assault_warning": {
        "description": "폭행 경고 방송 및 다각도 촬영 시작",
        "params_template": {"warning_type": "violence_alert"}
    },
    "call_police": {
        "description": "112 경찰서 자동 신고",
        "params_template": {"service": "112", "incident_type": "assault"}
    },
    "dispatch_security_team": {
        "description": "청원경찰 폭행 대응 출동",
        "params_template": {"team_size": 2}
    },
    "secure_evidence": {
        "description": "증거 영상 확보 및 무결성 보장",
        "params_template": {"video_backup": "high_quality"}
    },
    "log_assault_incident": {
        "description": "폭행 사건 로그 저장",
        "params_template": {"incident_type": "assault"}
    },
}

# 의료 관련 액션
MEDICAL_ACTIONS = {
    "activate_medical_assistance": {
        "description": "상황 안내 및 주변인 도움 요청",
        "params_template": {"assistance_type": "medical_emergency"}
    },
    "call_ambulance": {
        "description": "119 구급대 자동 신고",
        "params_template": {"service": "119", "incident_type": "medical_emergency"}
    },
    "dispatch_medical_team": {
        "description": "청원경찰 의료 대응 출동",
        "params_template": {"team_size": 2}
    },
    "guide_emergency_access": {
        "description": "구급차 진입 경로 확보",
        "params_template": {"access_route": "optimal_ambulance_path"}
    },
    "log_medical_incident": {
        "description": "의료 사건 로그 저장",
        "params_template": {"incident_type": "medical_collapse"}
    },
}

# 일반 액션
GENERAL_ACTIONS = {
    "continue_monitoring": {
        "description": "정상 모니터링을 계속합니다",
        "params_template": {"mode": "normal_surveillance"}
    },
    "log_normal_incident": {
        "description": "정상 상황 로그 저장",
        "params_template": {"incident_type": "normal"}
    },
}

# 모든 액션 통합
AVAILABLE_ACTIONS = {
    **FIRE_ACTIONS,
    **ASSAULT_ACTIONS,
    **MEDICAL_ACTIONS,
    **GENERAL_ACTIONS,
}

# 시나리오별 기본 액션 매핑
SCENARIO_ACTIONS = {
    "화재": {
        "긴급": ["activate_fire_alarm", "call_fire_department", "dispatch_fire_response_team", 
                "activate_fire_systems", "log_fire_incident"],
        "경계": ["activate_fire_alarm", "dispatch_fire_response_team", "log_fire_incident"],
        "관심": ["log_fire_incident"]
    },
    "폭행": {
        "긴급": ["activate_assault_warning", "call_police", "dispatch_security_team", 
                "secure_evidence", "log_assault_incident"],
        "경계": ["activate_assault_warning", "dispatch_security_team", "log_assault_incident"],
        "관심": ["log_assault_incident"]
    },
    "쓰러짐": {
        "긴급": ["activate_medical_assistance", "call_ambulance", "dispatch_medical_team", 
                "guide_emergency_access", "log_medical_incident"],
        "경계": ["activate_medical_assistance", "dispatch_medical_team", "log_medical_incident"],
        "관심": ["log_medical_incident"]
    },
    "정상상황": {
        "긴급": ["continue_monitoring", "log_normal_incident"],
        "경계": ["continue_monitoring", "log_normal_incident"],
        "관심": ["continue_monitoring", "log_normal_incident"]
    }
}

# 우선순위 매핑 (낮을수록 먼저 실행)
ACTION_PRIORITY = {
    "activate_fire_alarm": 1, "activate_assault_warning": 1, "activate_medical_assistance": 1,
    "call_fire_department": 2, "call_police": 2, "call_ambulance": 2,
    "dispatch_fire_response_team": 3, "dispatch_security_team": 3, "dispatch_medical_team": 3,
    "activate_fire_systems": 4, "guide_emergency_access": 5, "secure_evidence": 6,
    "continue_monitoring": 7,
    "log_fire_incident": 8, "log_assault_incident": 8, "log_medical_incident": 8, "log_normal_incident": 8
}




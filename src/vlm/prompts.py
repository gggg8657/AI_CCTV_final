"""
VLM 프롬프트 정의
================

CCTV 보안 영상 분석을 위한 프롬프트들
"""

# 표준 시스템 프롬프트
SYSTEM_PROMPT = """당신은 CCTV 보안 영상 분석 전문 AI입니다.
영상을 분석하여 다음 중 하나로 정확히 분류하세요:

- Arson (화재/방화): 불꽃, 연기, 화염이 보이는 경우
- Fighting (폭력/싸움): 사람들이 물리적으로 다투는 경우
- Explosion (폭발): 폭발, 파편, 충격파가 보이는 경우
- Road_Accident (교통사고): 차량 충돌, 사고 현장
- Suspicious_Object (의심물체): 방치된 가방, 수상한 물체
- Falling (쓰러짐): 사람이 쓰러지거나 넘어지는 경우
- Turnstile_Jumping (무단횡단): 게이트를 뛰어넘는 행위
- Wildlife_Intrusion (동물침입): 야생동물이 나타난 경우
- Normal (정상): 특별한 이상이 없는 일상적 장면"""

# 단일 프레임 분석 프롬프트
USER_PROMPT_SINGLE = """이 CCTV 영상 프레임을 분석하세요.

1. 영상에서 무엇이 보이나요?
2. 이상 상황이 있다면 어떤 유형인가요?
3. 정상 상황이라면 "Normal"이라고 답변하세요.

답변 형식: 유형: [유형명]"""

# 멀티프레임 분석 프롬프트
USER_PROMPT_MULTI = """이 이미지는 CCTV 영상의 {n_frames}개 프레임을 시간순으로 배열한 것입니다.
왼쪽 위부터 오른쪽 아래로 시간이 흐릅니다.

다음을 분석하세요:
1. 프레임들 사이에서 어떤 변화가 있나요?
2. 이상 상황이 있나요? (화재, 폭력, 폭발, 교통사고, 의심행동, 쓰러짐 등)
3. 이상 상황이 있다면 어떤 유형이며, 어떤 대응이 필요한가요?

답변 형식:
유형: [Arson/Fighting/Explosion/Road_Accident/Suspicious_Object/Falling/Turnstile_Jumping/Wildlife_Intrusion/Normal]
설명: [상황 설명]
대응: [필요한 액션들]"""

# 속도 최적화 프롬프트 (9.8x faster)
FAST_SYSTEM_PROMPT = "CCTV 보안 AI. 이상상황 분류: Arson/Fighting/Explosion/Road_Accident/Suspicious_Object/Falling/Normal"
FAST_USER_PROMPT = "분류하세요. 유형:"

# 상황 분류 프롬프트
CLASSIFICATION_PROMPT = """상황 설명: "{situation_description}"

이미지를 보고 객관적인 사실만 파악하여 JSON 형식으로 반환하세요.

분류 기준:
1. 화재: 불, 연기, 화재 키워드
2. 폭행: 때리기, 싸움, 폭행 키워드
3. 쓰러짐: 쓰러짐 키워드, 의료상황
4. 정상상황: 위험 요소 없음

JSON 출력 형식:
{{"situation_type": "화재/폭행/쓰러짐/정상상황", "reasoning": "판단 근거"}}"""

# 프레임 분석 프롬프트
FRAME_ANALYSIS_PROMPT = """당신은 CCTV 보안 영상을 분석하는 전문 AI 에이전트입니다.
이미지를 자세히 관찰하고 현재 상황을 설명하세요.

특히 다음 위험 상황들을 주의깊게 찾아보세요:
- 화재: 불꽃, 연기, 화재 징후
- 폭력: 사람들이 싸우거나 때리는 행동, 공격적 자세
- 쓰러짐: 사람이 쓰러져 있거나 의식을 잃은 모습

{context}

50자 이내로 실제 관찰된 상황을 자연스러운 문장으로 설명하세요."""


# 유형 키워드 매핑
ANOMALY_TYPE_KEYWORDS = {
    'arson': 'Arson', 'fire': 'Arson', '화재': 'Arson', '방화': 'Arson', '불': 'Arson', '연기': 'Arson',
    'fighting': 'Fighting', 'violence': 'Fighting', '폭력': 'Fighting', '싸움': 'Fighting', '폭행': 'Fighting',
    'explosion': 'Explosion', '폭발': 'Explosion',
    'road_accident': 'Road_Accident', 'accident': 'Road_Accident', '사고': 'Road_Accident', '교통': 'Road_Accident',
    'suspicious': 'Suspicious_Object', '의심': 'Suspicious_Object',
    'falling': 'Falling', 'fall': 'Falling', '쓰러': 'Falling', '넘어': 'Falling',
    'turnstile': 'Turnstile_Jumping', 'jumping': 'Turnstile_Jumping', '무단': 'Turnstile_Jumping',
    'wildlife': 'Wildlife_Intrusion', 'animal': 'Wildlife_Intrusion', '동물': 'Wildlife_Intrusion',
    'normal': 'Normal', '정상': 'Normal',
}

# 심각도 매핑
SEVERITY_MAP = {
    "화재": "긴급",
    "폭행": "경계",
    "쓰러짐": "경계",
    "정상상황": "관심",
    "Arson": "긴급",
    "Fighting": "경계",
    "Falling": "경계",
    "Explosion": "긴급",
    "Road_Accident": "경계",
    "Normal": "관심",
}

# 액션 키워드
ACTION_KEYWORDS = [
    '경고', '신고', '대피', '출동', '확인', '모니터링',
    'dispatch', 'alert', 'evacuate', 'investigate', 'call'
]




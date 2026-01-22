# Phase 3: COCO 클래스 매핑 전략

**작성일**: 2026-01-22  
**기반**: Codex 검토 응답  
**Linear Issue**: CHA-55

---

## 문제

COCO 데이터셋에는 "package" 클래스가 명확히 존재하지 않습니다.  
Phase 3에서 패키지를 감지하려면 COCO의 기존 클래스를 매핑하거나 커스텀 데이터셋이 필요합니다.

---

## COCO 클래스 매핑 전략

### 패키지로 간주할 COCO 클래스

| 클래스 ID | 클래스명 | 설명 | 적합도 |
|-----------|----------|------|--------|
| 24 | backpack | 백팩 | ⭐⭐⭐ |
| 26 | handbag | 핸드백 | ⭐⭐⭐ |
| 28 | suitcase | 여행가방 | ⭐⭐⭐⭐ |

**총 3개 클래스를 패키지로 감지**

---

## 구현 계획

### 1. PackageDetector 수정

```python
# COCO 클래스 매핑
COCO_PACKAGE_CLASS_IDS = [24, 26, 28]
COCO_PACKAGE_CLASS_NAMES = ["backpack", "handbag", "suitcase"]

class PackageDetector:
    def __init__(self, model_path: str, device: str = "cuda"):
        self.model = None
        self.model_path = model_path
        self.device = device
        self.confidence_threshold = 0.5
        # 패키지로 간주할 COCO 클래스 ID
        self.target_class_ids = COCO_PACKAGE_CLASS_IDS
        self.target_class_names = COCO_PACKAGE_CLASS_NAMES
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """프레임에서 패키지 감지"""
        results = self.model(frame)
        detections = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                
                # 패키지 클래스만 필터링
                if class_id in self.target_class_ids and confidence >= self.confidence_threshold:
                    detections.append(Detection(
                        bbox=self._box_to_xyxy(box.xyxy[0]),
                        confidence=confidence,
                        class_id=class_id,
                        class_name=COCO_CLASSES[class_id],
                        timestamp=time.time(),
                        timestamp_iso=datetime.now().isoformat()
                    ))
        
        return detections
```

### 2. Config 설정

```yaml
package_detection:
  enabled: true
  model:
    type: yolo
    path: yolo11n.pt  # 또는 yolo12n.pt
    target_class_ids: [24, 26, 28]  # COCO 클래스 ID
    target_class_names: [backpack, handbag, suitcase]
    confidence_threshold: 0.5
```

### 3. 요구사항 문서 수정

**수정 전**:
- "패키지 객체 감지"

**수정 후**:
- "패키지 유사 객체 감지 (backpack, handbag, suitcase)"

---

## 정확도 예상

### COCO 클래스별 mAP (참고)

- **backpack (24)**: COCO mAP@0.5 ≈ 0.6-0.7
- **handbag (26)**: COCO mAP@0.5 ≈ 0.5-0.6
- **suitcase (28)**: COCO mAP@0.5 ≈ 0.6-0.7

**평균 예상**: mAP@0.5 ≈ 0.6 (목표 0.5 초과 달성 가능)

---

## 향후 개선 방안

### 옵션 1: 커스텀 데이터셋 준비
- 실제 패키지 이미지 수집
- YOLO 커스텀 학습
- 정확도 향상 가능

### 옵션 2: 추가 클래스 포함
- "box" 관련 클래스 추가 검토
- COCO에 "box" 클래스가 있는지 확인

---

## 리스크 및 대응

### 리스크
- 실제 "package"와 "suitcase/handbag/backpack"이 다를 수 있음
- 정확도가 실제 패키지보다 낮을 수 있음

### 대응
- 실제 데이터로 정확도 측정
- 필요 시 커스텀 데이터셋 준비
- Confidence threshold 조정

---

**작성 완료일**: 2026-01-22

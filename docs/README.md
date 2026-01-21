# 문서 인덱스

AI CCTV 통합 시스템 문서 모음

---

## 📚 문서 목록

### 1. 프로젝트 계획 및 명세

#### [애자일 프로젝트 계획서](./AGILE_PROJECT_PLAN.md)
- 스프린트 계획 (4개 스프린트, 각 2주)
- 제품 백로그 (Epic 및 User Story)
- 스프린트별 목표 및 작업
- 리스크 관리
- 성공 지표

#### [상세 기능 명세서](./DETAILED_SPECIFICATION.md)
- 멀티 카메라 시스템 상세 설계
- REST API 서버 아키텍처
- 데이터베이스 설계 (ERD, 스키마)
- 알림 시스템 설계
- 인증 및 권한 관리
- 프론트엔드 통합

#### [REST API 명세서](./API_SPECIFICATION.md)
- 모든 API 엔드포인트 상세
- 요청/응답 예시
- 인증 방법
- WebSocket API
- 에러 응답 형식

#### [구현 체크리스트](./IMPLEMENTATION_CHECKLIST.md)
- 스프린트별 작업 체크리스트
- User Story별 구현 항목
- 공통 작업 (문서화, 테스트, 배포)

---

### 2. 시스템 아키텍처

#### [시스템 아키텍처](./SYSTEM_ARCHITECTURE.md)
- 전체 시스템 구조
- 컴포넌트 상세
- 데이터 흐름
- 성능 지표

#### [시스템 명세서](./SYSTEM_SPECIFICATION.md)
- 현재 구현 상태
- 추가 기능 명세 (우선순위별)
- 기술 스택
- 배포 및 운영

---

## 🚀 빠른 시작

### 개발자 가이드

1. **프로젝트 시작**: [애자일 프로젝트 계획서](./AGILE_PROJECT_PLAN.md) 읽기
2. **기능 이해**: [상세 기능 명세서](./DETAILED_SPECIFICATION.md) 참조
3. **API 개발**: [REST API 명세서](./API_SPECIFICATION.md) 참조
4. **작업 추적**: [구현 체크리스트](./IMPLEMENTATION_CHECKLIST.md) 사용

### 문서 읽기 순서

**신규 개발자**:
1. [시스템 아키텍처](./SYSTEM_ARCHITECTURE.md)
2. [시스템 명세서](./SYSTEM_SPECIFICATION.md)
3. [애자일 프로젝트 계획서](./AGILE_PROJECT_PLAN.md)
4. [상세 기능 명세서](./DETAILED_SPECIFICATION.md)

**API 개발자**:
1. [REST API 명세서](./API_SPECIFICATION.md)
2. [상세 기능 명세서](./DETAILED_SPECIFICATION.md) - API 섹션

**프론트엔드 개발자**:
1. [REST API 명세서](./API_SPECIFICATION.md)
2. [상세 기능 명세서](./DETAILED_SPECIFICATION.md) - 프론트엔드 통합 섹션

---

## 📋 문서 업데이트 규칙

- 각 스프린트 종료 시 관련 문서 업데이트
- API 변경 시 [REST API 명세서](./API_SPECIFICATION.md) 즉시 업데이트
- 새로운 기능 추가 시 [상세 기능 명세서](./DETAILED_SPECIFICATION.md) 업데이트
- 구현 완료 시 [구현 체크리스트](./IMPLEMENTATION_CHECKLIST.md) 체크

---

## 🔗 관련 링크

- [프로젝트 README](../README.md)
- [프로젝트 구조](../PROJECT_STRUCTURE.md)
- 원본 저장소: https://github.com/gggg8657/KAERI_CCTV_mt
- SHIBAL UI: https://github.com/gggg8657/SHIBAL

---

**문서 버전**: 1.0  
**최종 업데이트**: 2025-01-21

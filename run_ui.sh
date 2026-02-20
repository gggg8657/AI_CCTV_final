#!/bin/bash
# UI 실행 스크립트

echo '========================================'
echo 'AI_CCTV_final UI 실행'
echo '========================================'

echo ''
echo '실행할 UI를 선택하세요:'
echo '1) FastAPI Backend (http://localhost:8000)'
echo '2) React Frontend (http://localhost:5173) - FastAPI와 함께 실행'
echo '3) CLI UI'
echo ''
read -p '선택 (1-3): ' choice

case $choice in
    1)
        echo 'FastAPI Backend 시작...'
        uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000
        ;;
    2)
        echo 'React Frontend 시작...'
        cd ui && npm run dev
        ;;
    3)
        echo 'CLI UI 시작...'
        python3 app/cli_ui.py
        ;;
    *)
        echo '잘못된 선택입니다.'
        ;;
esac

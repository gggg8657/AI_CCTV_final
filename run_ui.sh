#!/bin/bash
# UI 실행 스크립트 (모델 없이)

echo '========================================'
echo 'AI_CCTV_final UI 실행 (모델 없이)'
echo '========================================'

# Config 확인
if grep -q 'enabled: true' configs/config.yaml | grep agent; then
    echo '⚠️  Agent가 활성화되어 있습니다.'
    echo '   UI 테스트를 위해 Agent를 비활성화하시겠습니까? (y/n)'
    read -r response
    if [ "$response" = "y" ]; then
        python3 -c "
import yaml
with open('configs/config.yaml', 'r') as f:
    config = yaml.safe_load(f)
config['agent']['enabled'] = False
with open('configs/config.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
print('Agent 비활성화 완료')
        "
    fi
fi

echo ''
echo '실행할 UI를 선택하세요:'
echo '1) Streamlit Web UI (http://localhost:8501)'
echo '2) FastAPI Backend (http://localhost:8000)'
echo '3) React Frontend (http://localhost:5173) - FastAPI와 함께 실행'
echo '4) CLI UI'
echo ''
read -p '선택 (1-4): ' choice

case $choice in
    1)
        echo 'Streamlit Web UI 시작...'
        streamlit run app/web_ui.py
        ;;
    2)
        echo 'FastAPI Backend 시작...'
        uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000
        ;;
    3)
        echo 'React Frontend 시작...'
        cd ui && npm run dev
        ;;
    4)
        echo 'CLI UI 시작...'
        python3 app/cli_ui.py
        ;;
    *)
        echo '잘못된 선택입니다.'
        ;;
esac

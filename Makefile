.PHONY: help build up down logs shell enroll ps restart audio-devices clean

help:
	@echo "備品室門禁系統 - 常用指令"
	@echo ""
	@echo "  make build           建構 Docker image"
	@echo "  make up              啟動服務 (背景)"
	@echo "  make down            停止服務"
	@echo "  make logs            查看即時 log"
	@echo "  make ps              顯示容器狀態"
	@echo "  make restart         重啟服務"
	@echo "  make shell           進入容器 shell"
	@echo ""
	@echo "  make enroll-list     列出所有人員"
	@echo "  make enroll-add      新增人員 (互動式)"
	@echo "  make enroll-run      人臉 enrollment (互動式)"
	@echo "  make audio-devices   列出可用音訊設備編號"
	@echo ""
	@echo "  Dashboard: http://localhost:8000"

build:
	docker compose build

up:
	docker compose up -d
	@echo "Dashboard: http://localhost:8000"

down:
	docker compose down

logs:
	docker compose logs -f face-app

ps:
	docker compose ps

restart:
	docker compose restart face-app

shell:
	docker compose exec face-app bash

enroll-list:
	docker compose exec face-app python enrollment/cli.py list

enroll-add:
	@read -p "姓名: " name; \
	read -p "部門: " dept; \
	read -p "員工編號: " eid; \
	docker compose exec face-app python enrollment/cli.py add --name "$$name" --dept "$$dept" --employee-id "$$eid"

enroll-run:
	@read -p "Person ID: " pid; \
	docker compose exec -it face-app python enrollment/cli.py enroll --person-id $$pid

audio-devices:
	docker compose exec face-app python -c "import sounddevice; print(sounddevice.query_devices())"

clean:
	docker compose down --rmi local
	rm -rf data/events/* models/*

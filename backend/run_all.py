import subprocess
import sys
import time
import requests

api = subprocess.Popen([
    sys.executable,
    "-m",
    "uvicorn",
    "backend.main:app",
    "--host",
    "0.0.0.0",
    "--port",
    "8000",
])

for _ in range(30):
    try:
        if requests.get("http://127.0.0.1:8000").status_code == 200:
            break
    except:
        pass
    time.sleep(1)

ui = subprocess.Popen([
    sys.executable,
    "-m",
    "streamlit",
    "run",
    "frontend/industry.py",
    "--server.port",
    "8501",
    "--server.address",
    "0.0.0.0",
])

while True:
    api_status = api.poll()
    ui_status = ui.poll()

    if api_status is not None:
        print(f"FastAPI exited with {api_status}")
        break

    if ui_status is not None:
        print(f"Streamlit exited with {ui_status}")
        break

    time.sleep(1)
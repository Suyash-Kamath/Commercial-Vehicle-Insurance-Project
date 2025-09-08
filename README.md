# Report Comparator (Vite React + FastAPI)

## Setup

### Backend
1. Create virtual env and install deps:
```
cd backend
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
2. Set environment variables:
```
# OpenAI API Key for OCR (only needed if uploading images)
setx OPENAI_API_KEY "YOUR_KEY"
```
3. Run server:
```
python run.py
```

### Frontend
```
cd frontend
npm i
npm run dev
```
Open http://localhost:5173

Set `VITE_API_URL` in a `.env` file if backend is not on default:
```
VITE_API_URL=http://localhost:8000
```

## Usage
- Upload the main report (.xlsx) with columns: Weight, IIB State, FGI State, FGI Zone, Status, OD Out flow, TP Outflow.
- Upload new report (.xlsx/.csv/.png/.jpg/.jpeg). If image, OCR is used.
- Click Compare & Download to receive `updated_report.xlsx` with columns plus Updated OD Outflow and Updated TP Outflow.

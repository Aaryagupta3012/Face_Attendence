# 🎓 Face Attendance System

A computer-vision attendance system: register students with a few face photos, then mark attendance with a single camera shot. Built with **Python, OpenCV (Haar cascade + LBPH), SQLite and Streamlit** — no dlib, no GPU, deploys free on Streamlit Community Cloud.

**Live demo:** _add your Streamlit Cloud URL here_

## Features

- **Register students** — name, roll number, department; capture face samples via webcam or upload
- **Mark attendance** — take one photo, every recognised face is marked present (once per day); unknown faces are flagged
- **Records** — date-range filter, daily headcount chart, full log, CSV export, per-student attendance %
- **Students** — view sample counts, delete students (model retrains automatically)
- **Adjustable strictness** — LBPH distance threshold in the sidebar

## How it works

```
photo ──► Haar cascade (detect faces) ──► grayscale 200×200 crop, histogram-equalised
            │
            ├── Register: crops saved to data/faces/<student_id>/ ──► LBPH model retrained
            │
            └── Mark: LBPH predict ──► (student_id, distance) ──► distance ≤ threshold → INSERT into SQLite
```

| Layer | Tech |
|---|---|
| Detection | OpenCV Haar cascade (`models/haarcascade_frontalface_default.xml`) |
| Recognition | OpenCV LBPH (`cv2.face.LBPHFaceRecognizer`) |
| Storage | SQLite (`students`, `attendance` with `UNIQUE(student_id, date)`) |
| UI | Streamlit (`st.camera_input`, dataframes, charts) |

## Run locally

```bash
git clone https://github.com/<you>/face-attendance.git
cd face-attendance
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy (Streamlit Community Cloud — free)

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app** → pick the repo, branch `main`, file `app.py` → **Deploy**.
3. Paste the URL at the top of this README.

> Note: Streamlit Cloud's filesystem is ephemeral — registered students and attendance reset when the app restarts. Fine for a demo; for persistence swap `db.py` to Postgres (e.g. Supabase/Neon) and store face crops in object storage.

## Project structure

```
app.py            Streamlit UI (4 pages)
face_engine.py    detection, cropping, LBPH train/predict, annotation
db.py             SQLite schema + queries
models/           Haar cascade XML
data/             (gitignored) faces/, attendance.db, lbph_model.yml
```

## Tips for good recognition

- Capture 5–10 samples per student, slight head turns, same-ish lighting as the attendance camera
- Keep the threshold around 55–70; lower it if strangers get matched, raise it if real students show as Unknown

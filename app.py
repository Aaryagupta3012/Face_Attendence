"""Face Attendance System — Streamlit UI."""
from datetime import date, timedelta

import pandas as pd
import streamlit as st

import db
import face_engine as fe

st.set_page_config(page_title="Face Attendance System", page_icon="🎓", layout="wide")
db.init_db()

MATCH_THRESHOLD_DEFAULT = 65.0  # LBPH distance; lower = stricter

# ---------- sidebar ----------
st.sidebar.title("🎓 Face Attendance")
page = st.sidebar.radio("Go to", ["Mark Attendance", "Register Student", "Records", "Students"], label_visibility="collapsed")
st.sidebar.divider()
threshold = st.sidebar.slider("Match strictness (LBPH distance)", 30.0, 110.0, MATCH_THRESHOLD_DEFAULT, 1.0,
                              help="A detected face is accepted only if its distance to a known face is below this. Lower = stricter.")
students_df = db.get_students()
st.sidebar.caption(f"{len(students_df)} students registered · model {'ready' if fe.MODEL_PATH.exists() else 'not trained'}")


def student_label(row) -> str:
    return f"{row['name']} ({row['roll_no']})"


# ======================================================================
if page == "Register Student":
    st.header("Register a student")
    col1, col2 = st.columns([1, 1])

    with col1:
        with st.form("student_form", clear_on_submit=True):
            name = st.text_input("Full name")
            roll_no = st.text_input("Roll number / ID")
            dept = st.text_input("Department", placeholder="e.g. CSE")
            if st.form_submit_button("Add student", type="primary"):
                if not name or not roll_no:
                    st.error("Name and roll number are required.")
                else:
                    try:
                        sid = db.add_student(roll_no, name, dept)
                        st.session_state["register_sid"] = sid
                        st.success(f"Added {name}. Now capture face samples on the right →")
                    except Exception as e:  # UNIQUE violation
                        st.error(f"Could not add student: {e}")

    with col2:
        students_df = db.get_students()
        if students_df.empty:
            st.info("Add a student first, then capture their face samples here.")
        else:
            default_idx = 0
            if "register_sid" in st.session_state:
                hits = students_df.index[students_df["id"] == st.session_state["register_sid"]].tolist()
                default_idx = hits[0] if hits else 0
            choice = st.selectbox("Capture samples for", students_df.index, index=default_idx,
                                  format_func=lambda i: student_label(students_df.loc[i]))
            sid = int(students_df.loc[choice, "id"])
            st.caption(f"Samples saved: **{fe.sample_count(sid)}** — aim for 5–10 from slightly different angles.")

            tab_cam, tab_upload = st.tabs(["📷 Camera", "📁 Upload"])
            frames: list[bytes] = []
            with tab_cam:
                shot = st.camera_input("Take a photo", key=f"cam_{sid}")
                if shot:
                    frames.append(shot.getvalue())
            with tab_upload:
                ups = st.file_uploader("Upload one or more photos", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key=f"up_{sid}")
                frames.extend(u.getvalue() for u in (ups or []))

            if frames and st.button("Save face sample(s)", type="primary"):
                saved, skipped = 0, 0
                for data in frames:
                    bgr = fe.bytes_to_bgr(data)
                    boxes = fe.detect_faces(bgr)
                    if not boxes:
                        skipped += 1
                        continue
                    fe.save_face_sample(sid, fe.crop_face(bgr, boxes[0]))
                    saved += 1
                n = fe.train_model()
                if saved:
                    st.success(f"Saved {saved} sample(s). Model retrained on {n} images.")
                if skipped:
                    st.warning(f"{skipped} image(s) skipped — no face detected. Use good lighting, face the camera.")
                st.rerun()

# ======================================================================
elif page == "Mark Attendance":
    st.header("Mark attendance")
    recognizer = fe.load_model()
    if recognizer is None:
        st.warning("No trained model yet. Register at least one student with face samples first.")
        st.stop()

    students_by_id = {int(r.id): r for r in students_df.itertuples()}
    today_df = db.get_attendance(date.today(), date.today())

    col1, col2 = st.columns([1, 1])
    with col1:
        shot = st.camera_input("Look at the camera and take a photo")
    with col2:
        st.metric("Present today", f"{len(today_df)} / {len(students_df)}")
        if not today_df.empty:
            st.dataframe(today_df[["time", "name", "roll_no"]], hide_index=True, width="stretch")

    if shot:
        bgr = fe.bytes_to_bgr(shot.getvalue())
        boxes = fe.detect_faces(bgr)
        if not boxes:
            st.error("No face detected. Move closer and make sure your face is well lit.")
        else:
            labels, results = [], []
            for box in boxes:
                sid, dist = fe.recognize(recognizer, fe.crop_face(bgr, box))
                if dist <= threshold and sid in students_by_id:
                    s = students_by_id[sid]
                    marked = db.mark_attendance(sid, round(dist, 1))
                    labels.append(f"{s.name} ({dist:.0f})")
                    results.append((s.name, s.roll_no, dist, marked))
                else:
                    labels.append(f"Unknown ({dist:.0f})")
                    results.append((None, None, dist, False))

            st.image(fe.draw_boxes(bgr, boxes, labels), caption="Detections (green = recognised, red = unknown)")
            for name, roll, dist, marked in results:
                if name is None:
                    st.error(f"Unknown face — closest match distance {dist:.1f} (threshold {threshold:.0f}).")
                elif marked:
                    st.success(f"✅ {name} ({roll}) marked present — match distance {dist:.1f}")
                else:
                    st.info(f"ℹ️ {name} ({roll}) was already marked present today.")

# ======================================================================
elif page == "Records":
    st.header("Attendance records")
    c1, c2, c3 = st.columns([1, 1, 2])
    start = c1.date_input("From", date.today() - timedelta(days=30))
    end = c2.date_input("To", date.today())
    df = db.get_attendance(start, end)

    if df.empty:
        st.info("No attendance in this range.")
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric("Records", len(df))
        m2.metric("Days with attendance", df["date"].nunique())
        m3.metric("Unique students", df["roll_no"].nunique())

        daily = df.groupby("date").size().rename("present").reset_index().sort_values("date")
        st.subheader("Daily headcount")
        st.bar_chart(daily.set_index("date")["present"])

        st.subheader("Log")
        st.dataframe(df, hide_index=True, width="stretch")
        st.download_button("⬇️ Download CSV", df.to_csv(index=False).encode(), f"attendance_{start}_{end}.csv", "text/csv")

    st.subheader("Per-student summary (all time)")
    summary = db.attendance_summary()
    if not summary.empty:
        st.dataframe(summary, hide_index=True, width="stretch")

# ======================================================================
elif page == "Students":
    st.header("Registered students")
    if students_df.empty:
        st.info("No students yet.")
    else:
        view = students_df.copy()
        view["samples"] = view["id"].apply(fe.sample_count)
        st.dataframe(view[["roll_no", "name", "department", "samples", "created_at"]], hide_index=True, width="stretch")

        st.divider()
        st.subheader("Remove a student")
        choice = st.selectbox("Student", students_df.index, format_func=lambda i: student_label(students_df.loc[i]))
        if st.button("Delete student and their attendance", type="secondary"):
            sid = int(students_df.loc[choice, "id"])
            db.delete_student(sid)
            folder = fe.FACES_DIR / str(sid)
            if folder.exists():
                for p in folder.glob("*"):
                    p.unlink()
                folder.rmdir()
            fe.train_model()
            st.success("Deleted. Model retrained.")
            st.rerun()

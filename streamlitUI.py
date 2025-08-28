import streamlit as st
import io, csv

# ---------------------------
# 3.1 Page config
# ---------------------------
st.set_page_config(page_title="패스포인트", page_icon="💬", layout="wide", initial_sidebar_state="expanded")

# ---------------------------
# 3.2 Tiny CSS for styling (safe tweaks)
# ---------------------------
st.markdown(
    """
    <style>
      /* Bubbles */
      .bubble-user {background:#e5f2ff;padding:0.7rem 1rem;border-radius:14px;display:inline-block;}
      .bubble-assistant {background:#efecff;padding:0.7rem 1rem;border-radius:14px;display:inline-block;}
      .muted {color:#6b7280;font-size:0.85rem}
      /* Card-like containers */
      [data-testid="stVerticalBlock"]>.st-emotion-cache-ue6h4q {padding:0.6rem 0.8rem;border-radius:12px;background:#f7f7fb;}
      /* Sidebar width hint */
      section[data-testid="stSidebar"] {min-width: 300px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# 3.3 App state (all pages share this)
# ---------------------------
if "profile" not in st.session_state:
    st.session_state.profile = {"name": "Guest", "avatar": None}
if "settings" not in st.session_state:
    st.session_state.settings = {"model": "demo", "temperature": 0.7, "theme": "system"}
if "chat" not in st.session_state:
    st.session_state.chat = {"history": []}  # list of dicts {role, content}
if "docs" not in st.session_state:
    st.session_state.docs = {"files": [], "notes": ""}

# ---------------------------
# 3.4 Navigation (custom router)
# ---------------------------
with st.sidebar:
    st.header("Navigation")
    page = st.radio("Go to", ["Home", "Chat", "Documents", "Settings", "User"], label_visibility="collapsed")
    st.divider()
    st.caption("Tip: Use **/** to focus the chat input. Use the tabs on Documents to upload & preview.")

# ---------------------------
# 3.5 Page functions (all in one file)
# ---------------------------

def hero(title: str, subtitle: str):
    st.markdown(
        f"""
        <div style='padding:1rem;border-radius:16px;background:#f6f6f9'>
          <h1 style='margin:0'>{title}</h1>
          <p style='margin:0.3rem 0 0 0;color:#555'>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# --- Home ---

def page_home():
    hero("💬 Chat App", "A simple chatbot with uploads and settings — no extra packages.")
    st.write("Use the sidebar to switch pages.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Conversations", len([m for m in st.session_state.chat["history"] if m["role"]=="user"]))
    c2.metric("Assistant replies", len([m for m in st.session_state.chat["history"] if m["role"]=="assistant"]))
    c3.metric("Files uploaded", len(st.session_state.docs.get("files", [])))

# --- Chat logic (placeholder brain) ---

def simple_answer(user_text: str) -> str:
    # Very basic: search your notes text for the query and echo back
    notes = (st.session_state.docs.get("notes") or "").lower()
    if user_text and user_text.lower() in notes:
        return "Found that in your notes. Here's a snippet: " + notes[:300]
    return f"You said: {user_text}. (Replace this with your model call.)"

# --- Chat page ---

def page_chat():
    st.title("Chat")

    # Render history
    for msg in st.session_state.chat["history"]:
        with st.chat_message(msg["role"]):
            css = "bubble-user" if msg["role"]=="user" else "bubble-assistant"
            st.markdown(f"<div class='{css}'>{msg['content']}</div>", unsafe_allow_html=True)

    # Input
    user_text = st.chat_input("Type a message…")
    if user_text:
        st.session_state.chat["history"].append({"role": "user", "content": user_text})
        with st.chat_message("user"):
            st.markdown(f"<div class='bubble-user'>{user_text}</div>", unsafe_allow_html=True)

        reply = simple_answer(user_text)
        st.session_state.chat["history"].append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.markdown(f"<div class='bubble-assistant'>{reply}</div>", unsafe_allow_html=True)

    st.divider()
    col1, col2, col3 = st.columns(3)
    if col1.button("Clear chat"):
        st.session_state.chat["history"] = []
        st.toast("Chat cleared.")
    if col2.button("Copy last answer") and any(m["role"]=="assistant" for m in st.session_state.chat["history"]):
        last = [m for m in st.session_state.chat["history"] if m["role"]=="assistant"][-1]["content"]
        st.code(last)
    if col3.download_button("Download history (JSON)", str(st.session_state.chat["history"])):
        pass

# --- Documents page ---

def page_documents():
    st.title("Documents")
    tabs = st.tabs(["Upload", "Preview", "Notes"])  # Notes power the demo chat

    with tabs[0]:
        uploaded = st.file_uploader(
            "Upload files (TXT, CSV, PDF, PNG, JPG)",
            type=["txt","csv","pdf","png","jpg","jpeg"],
            accept_multiple_files=True,
        )
        if uploaded:
            st.session_state.docs["files"] = uploaded
            st.success(f"Saved {len(uploaded)} file(s). Go to Preview tab.")

    with tabs[1]:
        files = st.session_state.docs.get("files", [])
        if not files:
            st.info("No files uploaded yet.")
        else:
            for f in files:
                st.write("—", f.name)
                if f.type == "text/plain":
                    st.text(f.read().decode("utf-8")[:500]); f.seek(0)
                elif f.type in ("text/csv", "application/vnd.ms-excel"):
                    # Use csv module (no pandas)
                    content = io.StringIO(f.read().decode("utf-8")); f.seek(0)
                    reader = csv.reader(content)
                    rows = list(reader)[:10]
                    st.table(rows)
                elif f.type.startswith("image/"):
                    st.image(f, use_container_width=True); f.seek(0)
                else:
                    st.caption("(Preview not implemented — stored for later processing)")

    with tabs[2]:
        st.caption("Add quick notes or paste key text to power the demo chat.")
        st.session_state.docs["notes"] = st.text_area("Reference notes", value=st.session_state.docs.get("notes", ""), height=160)
        st.success("Notes saved.")

# --- Settings page ---

def page_settings():
    st.title("Settings")
    with st.form("settings_form"):
        model = st.selectbox("Model", ["demo", "gpt-like", "local"], index=["demo","gpt-like","local"].index(st.session_state.settings["model"]))
        temperature = st.slider("Creativity (temperature)", 0.0, 1.0, st.session_state.settings["temperature"], 0.05)
        theme = st.radio("Theme", ["system", "light", "dark"], index=["system","light","dark"].index(st.session_state.settings["theme"]))
        submitted = st.form_submit_button("Save settings")
    if submitted:
        st.session_state.settings.update({"model": model, "temperature": temperature, "theme": theme})
        st.success("Settings saved.")

# --- User page ---

def page_user():
    st.title("User")
    c1, c2 = st.columns([1,3])
    with c1:
        if st.session_state.profile["avatar"]:
            st.image(st.session_state.profile["avatar"], caption="Avatar", use_container_width=True)
        else:
            st.markdown("<div class='muted'>(No avatar uploaded)</div>", unsafe_allow_html=True)
    with c2:
        with st.form("profile_form"):
            name = st.text_input("Display name", value=st.session_state.profile.get("name", "Guest"))
            avatar = st.file_uploader("Upload avatar", type=["png","jpg","jpeg"])
            ok = st.form_submit_button("Save profile")
        if ok:
            st.session_state.profile["name"] = name or "Guest"
            if avatar:
                st.session_state.profile["avatar"] = avatar
            st.success("Profile updated.")

# ---------------------------
# 3.6 Router switch
# ---------------------------
if page == "Home":
    page_home()
elif page == "Chat":
    page_chat()
elif page == "Documents":
    page_documents()
elif page == "Settings":
    page_settings()
elif page == "User":
    page_user()
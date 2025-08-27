# app.py
# =========================================================
# AI 자기소개서 코칭 - 모던 UI
# =========================================================
# 설치: pip install streamlit python-docx reportlab langchain langchain-openai python-dotenv
# 실행: streamlit run app.py
# =========================================================

import os, io, datetime, json
from typing import Optional, List, Dict
import streamlit as st

# ===== 문서 생성 라이브러리 (선택) =====
try:
    from docx import Document
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    DOC_LIBS_AVAILABLE = True
except:
    DOC_LIBS_AVAILABLE = False

# ===== LangChain (선택) =====
try:
    from langchain_openai import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate
    from langchain.chains import LLMChain
    LANGCHAIN_AVAILABLE = True
except:
    LANGCHAIN_AVAILABLE = False

# ================= 페이지 설정 =================
st.set_page_config(
    page_title="AI 자기소개서 코칭",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ================= 모던 UI CSS =================
st.markdown("""
<style>
    @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');
    
    /* ===== CSS 변수 정의 ===== */
    :root {
        --primary: #0A84FF;
        --primary-dark: #0A6AD9;
        --primary-light: rgba(10,132,255,.1);
        --surface: #FFFFFF;
        --surface-alt: #F6F8FA;
        --text: #0F172A;
        --text-secondary: #475569;
        --subtext: #6B7280;
        --border: #E5E7EB;
        --success: #10B981;
        --warning: #F59E0B;
        --error: #EF4444;
        --radius-xl: 20px;
        --radius-lg: 16px;
        --radius-md: 12px;
        --radius-sm: 8px;
        --shadow-sm: 0 1px 3px rgba(0,0,0,.08);
        --shadow: 0 8px 24px rgba(0,0,0,.08);
        --shadow-lg: 0 20px 40px rgba(0,0,0,.12);
        --font: 'Pretendard', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    /* ===== 전역 스타일 ===== */
    html, body, .stApp {
        background: var(--surface-alt);
        color: var(--text);
        font-family: var(--font);
    }
    
    .main .block-container {
        max-width: 980px;
        padding: 24px 20px 48px;
        margin: 0 auto;
    }
    
    /* ===== 헤더 ===== */
    .app-header {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        height: 56px;
        background: var(--surface);
        border-bottom: 1px solid var(--border);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 100;
        box-shadow: var(--shadow-sm);
    }
    
    .app-header h1 {
        font-size: 18px;
        font-weight: 600;
        color: var(--text);
        margin: 0;
    }
    
    /* ===== 탭 네비게이션 ===== */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--surface);
        padding: 4px;
        border-radius: var(--radius-md);
        gap: 4px;
        border: 1px solid var(--border);
        margin-top: 70px;
        margin-bottom: 24px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: var(--radius-sm);
        padding: 8px 16px;
        font-weight: 500;
        color: var(--text-secondary);
        background: transparent;
        border: none;
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--primary);
        color: white;
    }
    
    /* ===== 채팅 컨테이너 ===== */
    .chat-container {
        background: var(--surface);
        border-radius: var(--radius-lg);
        padding: 24px;
        min-height: 500px;
        margin-bottom: 20px;
        box-shadow: var(--shadow);
    }
    
    /* ===== 메시지 스타일 ===== */
    .chat {
        display: flex;
        gap: 12px;
        margin: 16px 0;
        align-items: flex-start;
    }
    
    .chat.user {
        justify-content: flex-end;
    }
    
    .avatar {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        flex: 0 0 36px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
        font-size: 14px;
        color: white;
    }
    
    .avatar.ai-avatar {
        background: linear-gradient(135deg, var(--primary), var(--primary-dark));
    }
    
    .avatar.user-avatar {
        background: linear-gradient(135deg, #8B5CF6, #7C3AED);
    }
    
    .bubble {
        max-width: 70%;
        padding: 12px 16px;
        border-radius: var(--radius-lg);
        line-height: 1.5;
        font-size: 14px;
        word-break: break-word;
    }
    
    .chat.ai .bubble {
        background: var(--surface-alt);
        color: var(--text);
        border: 1px solid var(--border);
        border-top-left-radius: 4px;
    }
    
    .chat.user .bubble {
        background: var(--primary);
        color: white;
        border-top-right-radius: 4px;
    }
    
    .msg-time {
        font-size: 11px;
        color: var(--subtext);
        margin-top: 4px;
    }
    
    /* ===== 카드 스타일 ===== */
    .card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        padding: 20px;
        box-shadow: var(--shadow);
        margin-bottom: 16px;
    }
    
    .card-title {
        font-size: 16px;
        font-weight: 600;
        color: var(--text);
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    /* ===== 버튼 스타일 ===== */
    .stButton > button {
        background: var(--primary);
        color: white;
        border: none;
        border-radius: var(--radius-md);
        height: 44px;
        padding: 0 20px;
        font-weight: 500;
        font-size: 14px;
        transition: all 0.2s;
        box-shadow: var(--shadow-sm);
    }
    
    .stButton > button:hover {
        background: var(--primary-dark);
        box-shadow: var(--shadow);
        transform: translateY(-1px);
    }
    
    /* 보조 버튼 */
    .btn-secondary > button {
        background: var(--surface);
        color: var(--text);
        border: 1px solid var(--border);
    }
    
    .btn-secondary > button:hover {
        background: var(--surface-alt);
        border-color: var(--primary);
        color: var(--primary);
    }
    
    /* 위험 버튼 */
    .btn-danger > button {
        background: var(--error);
        color: white;
    }
    
    .btn-danger > button:hover {
        background: #DC2626;
    }
    
    /* ===== 입력 필드 스타일 ===== */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div > select {
        border: 1px solid var(--border);
        border-radius: var(--radius-md);
        padding: 10px 14px;
        font-size: 14px;
        background: var(--surface);
        transition: all 0.2s;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus,
    .stSelectbox > div > div > select:focus {
        outline: none;
        border-color: var(--primary);
        box-shadow: 0 0 0 3px var(--primary-light);
    }
    
    /* ===== 파일 업로드 스타일 ===== */
    .stFileUploader > div {
        background: var(--surface);
        border: 2px dashed var(--border);
        border-radius: var(--radius-md);
        transition: all 0.2s;
    }
    
    .stFileUploader > div:hover {
        border-color: var(--primary);
        background: var(--primary-light);
    }
    
    /* ===== 슬라이더 스타일 ===== */
    .stSlider > div > div > div {
        background: var(--primary);
    }
    
    .stSlider > div > div > div > div {
        background: var(--primary);
        border: 3px solid white;
        box-shadow: var(--shadow-sm);
    }
    
    /* ===== 알림 메시지 스타일 ===== */
    .stSuccess, .stInfo, .stWarning, .stError {
        border-radius: var(--radius-md);
        padding: 12px 16px;
        font-size: 14px;
        border: 1px solid;
    }
    
    .stSuccess {
        background: #D1FAE5;
        color: #065F46;
        border-color: #10B981;
    }
    
    .stInfo {
        background: #DBEAFE;
        color: #1E40AF;
        border-color: #3B82F6;
    }
    
    .stWarning {
        background: #FEF3C7;
        color: #92400E;
        border-color: #F59E0B;
    }
    
    .stError {
        background: #FEE2E2;
        color: #991B1B;
        border-color: #EF4444;
    }
    
    /* ===== 빠른 답변 버튼 ===== */
    .quick-replies {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        margin-bottom: 16px;
    }
    
    .quick-reply-btn {
        padding: 8px 16px;
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius-xl);
        font-size: 13px;
        color: var(--text-secondary);
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .quick-reply-btn:hover {
        background: var(--primary);
        color: white;
        border-color: var(--primary);
        transform: translateY(-2px);
        box-shadow: var(--shadow-sm);
    }
    
    /* ===== 입력 영역 ===== */
    .input-area {
        background: var(--surface);
        border-radius: var(--radius-lg);
        padding: 16px;
        box-shadow: var(--shadow);
        margin-top: 20px;
    }
    
    /* ===== 설정 페이지 ===== */
    .settings-section {
        background: var(--surface);
        border-radius: var(--radius-lg);
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: var(--shadow);
    }
    
    .settings-header {
        font-size: 18px;
        font-weight: 600;
        color: var(--text);
        margin-bottom: 20px;
        padding-bottom: 12px;
        border-bottom: 2px solid var(--border);
    }
    
    /* ===== 파일 아이템 ===== */
    .file-item {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius-md);
        padding: 16px;
        margin-bottom: 12px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        transition: all 0.2s;
    }
    
    .file-item:hover {
        box-shadow: var(--shadow-sm);
        border-color: var(--primary);
    }
    
    .file-info {
        flex: 1;
    }
    
    .file-name {
        font-weight: 500;
        color: var(--text);
        margin-bottom: 4px;
    }
    
    .file-meta {
        font-size: 12px;
        color: var(--subtext);
    }
    
    /* ===== 스크롤바 ===== */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--surface-alt);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--border);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--subtext);
    }
    
    /* ===== 반응형 디자인 ===== */
    @media (max-width: 768px) {
        .main .block-container {
            padding: 16px 12px;
        }
        
        .bubble {
            max-width: 85%;
        }
        
        .chat-container,
        .settings-section {
            padding: 16px;
        }
    }
    
    /* ===== 타이포그래피 ===== */
    h1 {
        font-size: 24px;
        line-height: 32px;
        font-weight: 600;
        color: var(--text);
        margin: 0 0 8px;
    }
    
    h2 {
        font-size: 20px;
        line-height: 28px;
        font-weight: 600;
        color: var(--text);
        margin: 24px 0 8px;
    }
    
    h3 {
        font-size: 16px;
        line-height: 24px;
        font-weight: 600;
        color: var(--text);
        margin: 16px 0 8px;
    }
    
    p, li {
        font-size: 14px;
        line-height: 22px;
        color: var(--text-secondary);
    }
    
    .small {
        font-size: 12px;
        line-height: 18px;
        color: var(--subtext);
    }
    
    /* Streamlit 기본 요소 재정의 */
    .css-1d391kg, .st-ae {
        font-family: var(--font);
    }
    
    /* 탭 컨텐츠 영역 */
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 0;
    }
    
    /* 메트릭 카드 */
    [data-testid="metric-container"] {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius-md);
        padding: 16px;
        box-shadow: var(--shadow-sm);
    }
    
    /* 사이드바 */
    .css-1d391kg {
        background: var(--surface);
    }
    
    /* 프로그레스 바 */
    .stProgress > div > div > div {
        background: var(--primary);
    }
    
    /* 체크박스 & 라디오 */
    .stCheckbox > label,
    .stRadio > label {
        font-size: 14px;
        color: var(--text);
    }
</style>
""", unsafe_allow_html=True)

# ================= 세션 초기화 =================
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "ai",
        "content": "안녕하세요! AI 자기소개서 코치입니다. 무엇을 도와드릴까요?",
        "time": datetime.datetime.now().strftime("%H:%M")
    })

if "api_key" not in st.session_state:
    st.session_state.api_key = os.getenv("OPENAI_API_KEY", "")

if "saved_files" not in st.session_state:
    st.session_state.saved_files = []

if "model_settings" not in st.session_state:
    st.session_state.model_settings = {
        "temperature": 0.7,
        "max_length": 1000,
        "tone": "professional"
    }

# ================= 가이드라인 응답 =================
def get_guideline_response():
    return """📝 **AI 자기소개서 작성 가이드**

**1️⃣ 구체적인 질문하기**
• ✅ "마케팅 직무 신입 자기소개서 도입부 작성해줘"
• ❌ "자소서 써줘"

**2️⃣ 필요한 정보 제공**
• 지원 회사와 직무
• 주요 경험과 프로젝트
• 강조하고 싶은 역량

**3️⃣ 효과적인 활용 예시**
• "고객 서비스 경험을 영업직무와 연결하는 방법"
• "프로젝트 경험을 STAR 기법으로 정리"
• "IT 기업 지원동기 작성 도와줘"

**4️⃣ 첨삭 요청 방법**
• 작성한 내용 복사 → "이 내용 첨삭해줘"
• 파일 업로드 → "구체성 높여줘"

💡 **Pro Tip**: 한 번에 완성하려 하지 말고 단계별로 접근하세요!"""

# ================= AI 응답 생성 =================
def get_ai_response(user_input: str, uploaded_file=None) -> str:
    # 가이드라인 요청 체크
    guideline_keywords = ["가이드", "도움말", "사용법", "어떻게"]
    if any(keyword in user_input for keyword in guideline_keywords):
        return get_guideline_response()
    
    # API 키 없을 때 기본 응답
    if not st.session_state.api_key or not LANGCHAIN_AVAILABLE:
        templates = {
            "default": """자기소개서 작성을 도와드리겠습니다! 

구체적으로 알려주시면 더 정확한 도움을 드릴 수 있어요:
• 어떤 직무에 지원하시나요?
• 어떤 부분이 어려우신가요?
• 강조하고 싶은 경험이 있나요?""",
            
            "첨삭": """자기소개서 첨삭 포인트:

✅ 구체적인 숫자와 성과 포함
✅ 직무와 연관된 경험 강조
✅ 간결하고 명확한 문장
✅ 진정성 있는 지원동기

내용을 보내주시면 자세히 봐드릴게요!"""
        }
        
        if "첨삭" in user_input or "수정" in user_input:
            return templates["첨삭"]
        return templates["default"]
    
    # LangChain AI 응답 생성
    try:
        llm = ChatOpenAI(
            api_key=st.session_state.api_key,
            model="gpt-4o-mini",
            temperature=st.session_state.model_settings["temperature"]
        )
        
        system_prompt = f"""당신은 전문 자기소개서 작성 코치입니다.
        톤: {st.session_state.model_settings["tone"]}
        최대 길이: {st.session_state.model_settings["max_length"]}자
        
        - 구체적이고 실용적인 조언 제공
        - 예시를 들어 명확하게 설명
        - 친근하면서도 전문적인 톤 유지"""
        
        if uploaded_file:
            try:
                content = uploaded_file.read().decode('utf-8')
                user_input = f"다음 자기소개서를 검토해주세요:\n\n{content}\n\n{user_input}"
            except:
                return "파일을 읽을 수 없습니다."
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])
        
        chain = LLMChain(llm=llm, prompt=prompt)
        response = chain.invoke({"input": user_input})
        
        return response.get("text", str(response))
        
    except Exception as e:
        return f"오류가 발생했습니다. 다시 시도해주세요."

# ================= 대화 저장 =================
def save_conversation():
    content = ""
    for msg in st.session_state.messages:
        role = "👤 사용자" if msg["role"] == "user" else "🤖 AI"
        content += f"[{msg.get('time', '')}] {role}\n{msg['content']}\n\n"
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"자소서대화_{timestamp}.txt"
    
    st.session_state.saved_files.append({
        "name": filename,
        "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "size": len(content),
        "data": content
    })
    
    return filename

# ================= UI 렌더링 함수 =================
def render_chat_message(msg):
    """채팅 메시지 렌더링"""
    if msg["role"] == "user":
        st.markdown(f'''
            <div class="chat user">
                <div style="text-align: right; width: 100%;">
                    <div class="bubble">{msg["content"]}</div>
                    <div class="msg-time">{msg.get("time", "")}</div>
                </div>
                <div class="avatar user-avatar">나</div>
            </div>
        ''', unsafe_allow_html=True)
    else:
        content_html = msg["content"].replace('\n', '<br>')
        st.markdown(f'''
            <div class="chat ai">
                <div class="avatar ai-avatar">AI</div>
                <div>
                    <div class="bubble">{content_html}</div>
                    <div class="msg-time">{msg.get("time", "")}</div>
                </div>
            </div>
        ''', unsafe_allow_html=True)

def render_chat_tab():
    """대화 탭 렌더링"""
    # 채팅 컨테이너
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    for msg in st.session_state.messages:
        render_chat_message(msg)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 빠른 답변
    st.markdown('<div class="quick-replies">', unsafe_allow_html=True)
    quick_replies = ["🎯 가이드 보기", "✍️ 자소서 시작", "📝 첨삭 요청", "💡 예시 보기"]
    cols = st.columns(len(quick_replies))
    for i, reply in enumerate(quick_replies):
        with cols[i]:
            if st.button(reply, key=f"quick_{i}"):
                text = reply.split(' ', 1)[1]  # 이모지 제거
                st.session_state.messages.append({
                    "role": "user",
                    "content": text,
                    "time": datetime.datetime.now().strftime("%H:%M")
                })
                response = get_ai_response(text)
                st.session_state.messages.append({
                    "role": "ai",
                    "content": response,
                    "time": datetime.datetime.now().strftime("%H:%M")
                })
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 입력 영역
    st.markdown('<div class="input-area">', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "📎 파일 첨부",
        type=['txt', 'docx'],
        help="자기소개서 파일을 업로드하세요"
    )
    
    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])
        with col1:
            user_input = st.text_input(
                "메시지",
                placeholder="질문을 입력하세요...",
                label_visibility="collapsed"
            )
        with col2:
            send = st.form_submit_button("전송", use_container_width=True)
        
        if send and user_input:
            st.session_state.messages.append({
                "role": "user",
                "content": user_input,
                "time": datetime.datetime.now().strftime("%H:%M")
            })
            
            with st.spinner("AI가 답변을 작성 중..."):
                response = get_ai_response(user_input, uploaded_file)
            
            st.session_state.messages.append({
                "role": "ai",
                "content": response,
                "time": datetime.datetime.now().strftime("%H:%M")
            })
            
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

def render_settings_tab():
    """설정 탭 렌더링"""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # API 설정
        st.markdown('<div class="settings-section">', unsafe_allow_html=True)
        st.markdown('<h3 class="settings-header">🔑 API 설정</h3>', unsafe_allow_html=True)
        
        api_key = st.text_input(
            "OpenAI API Key",
            value=st.session_state.api_key,
            type="password",
            placeholder="sk-...",
            help="더 정확한 AI 응답을 위해 API 키를 입력하세요"
        )
        
        if api_key != st.session_state.api_key:
            st.session_state.api_key = api_key
            st.success("✅ API 키가 저장되었습니다!")
        
        st.info("💡 API 키가 없어도 기본 기능을 사용할 수 있습니다")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # AI 모델 설정
        st.markdown('<div class="settings-section">', unsafe_allow_html=True)
        st.markdown('<h3 class="settings-header">🤖 AI 모델 설정</h3>', unsafe_allow_html=True)
        
        st.session_state.model_settings["temperature"] = st.slider(
            "창의성 레벨",
            0.0, 1.0,
            st.session_state.model_settings["temperature"],
            0.1,
            help="높을수록 창의적인 답변"
        )
        
        st.session_state.model_settings["tone"] = st.selectbox(
            "응답 스타일",
            ["professional", "friendly", "casual"],
            help="AI의 대화 톤 선택"
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # 대화 관리
        st.markdown('<div class="settings-section">', unsafe_allow_html=True)
        st.markdown('<h3 class="settings-header">💬 대화 관리</h3>', unsafe_allow_html=True)
        
        if st.button("🗑️ 대화 초기화", use_container_width=True):
            st.session_state.messages = [{
                "role": "ai",
                "content": "안녕하세요! AI 자기소개서 코치입니다. 무엇을 도와드릴까요?",
                "time": datetime.datetime.now().strftime("%H:%M")
            }]
            st.success("대화가 초기화되었습니다!")
            st.rerun()
        
        if st.button("💾 대화 저장", use_container_width=True):
            filename = save_conversation()
            st.success(f"{filename} 저장완료!")
        
        st.markdown('</div>', unsafe_allow_html=True)

def render_storage_tab():
    """저장소 탭 렌더링"""
    if not st.session_state.saved_files:
        st.info("📂 저장된 파일이 없습니다. 대화를 저장하려면 설정 탭을 이용하세요.")
    else:
        st.success(f"📁 총 {len(st.session_state.saved_files)}개의 파일이 저장되어 있습니다")
        
        for i, file in enumerate(st.session_state.saved_files):
            st.markdown(f'''
                <div class="file-item">
                    <div class="file-info">
                        <div class="file-name">📄 {file["name"]}</div>
                        <div class="file-meta">{file["date"]} · {file["size"]} bytes</div>
                    </div>
                </div>
            ''', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([3, 1, 1])
            with col2:
                st.download_button(
                    "⬇️ 다운로드",
                    data=file["data"],
                    file_name=file["name"],
                    key=f"download_{i}"
                )
        
        if st.session_state.saved_files:
            st.markdown("---")
            if st.button("🗑️ 모든 파일 삭제", type="secondary"):
                st.session_state.saved_files = []
                st.success("모든 파일이 삭제되었습니다!")
                st.rerun()

# ================= 메인 앱 =================
def main():
    # 헤더
    st.markdown('''
        <div class="app-header">
            <h1>✍️ AI 자기소개서 코칭</h1>
        </div>
    ''', unsafe_allow_html=True)
    
    # 탭 생성
    tab1, tab2, tab3 = st.tabs(["💬 대화", "⚙️ 설정", "📁 저장소"])
    
    with tab1:
        render_chat_tab()
    
    with tab2:
        render_settings_tab()
    
    with tab3:
        render_storage_tab()

if __name__ == "__main__":
    main()

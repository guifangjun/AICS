"""
模拟聊天 API + 网页界面
"""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from config.settings import config
from core.memory_manager import get_memory_manager
from customer_service.fast_reply import FastReplyManager
from customer_service.human_handoff import HumanHandoff
from customer_service.tone_optimizer import ToneOptimizer
from utils.logger import get_logger, get_chat_logger

logger = get_logger(__name__)
chat_logger = get_chat_logger()

router = APIRouter()

CHAT_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>华佗医考 · 课程咨询</title>
<style>
* { margin:0; padding:0; box-sizing:border-box; }
body { background:#ededed; font-family:-apple-system,BlinkMacSystemFont,sans-serif; }
.phone { max-width:420px; margin:20px auto; background:#fff; border-radius:12px; overflow:hidden; box-shadow:0 4px 20px rgba(0,0,0,.15); }
.header { background:#07c160; color:#fff; padding:14px 16px; text-align:center; font-size:17px; font-weight:600; }
.chat-box { height:480px; overflow-y:auto; padding:12px; background:#f5f5f5; display:flex; flex-direction:column; gap:10px; }
.msg { max-width:78%; padding:10px 14px; border-radius:8px; font-size:14px; line-height:1.5; word-break:break-word; }
.msg.user { align-self:flex-end; background:#95ec69; }
.msg.bot { align-self:flex-start; background:#fff; border:1px solid #e0e0e0; }
.msg.system { align-self:center; background:#e5e5e5; color:#888; font-size:12px; border-radius:4px; padding:4px 10px; }
.tag { display:inline-block; font-size:10px; padding:2px 6px; border-radius:3px; margin-right:4px; }
.tag.fast { background:#ffd700; color:#333; }
.tag.handoff { background:#ff4444; color:#fff; }
.tag.rag { background:#4a90d9; color:#fff; }
.input-area { display:flex; padding:10px; background:#fff; border-top:1px solid #e0e0e0; gap:8px; }
.input-area input { flex:1; padding:10px 14px; border:1px solid #ddd; border-radius:20px; font-size:14px; outline:none; }
.input-area input:focus { border-color:#07c160; }
.input-area button { background:#07c160; color:#fff; border:none; border-radius:20px; padding:10px 20px; font-size:14px; cursor:pointer; }
.input-area button:hover { background:#06ad56; }
.info { padding:8px 16px; background:#fffbe6; font-size:11px; color:#999; text-align:center; border-top:1px solid #f0f0f0; }
</style>
</head>
<body>
<div class="phone">
  <div class="header">🎓 华佗 · 课程顾问</div>
  <div class="chat-box" id="chatBox">
    <div class="msg bot">你好呀～我是华佗医考的课程顾问"华佗"！🎓<br>想了解什么课程呢？医师、护士、药师、考研都可以问我～</div>
  </div>
  <div class="input-area">
    <input id="msgInput" placeholder="输入消息..." onkeydown="if(event.key==='Enter')send()" autofocus>
    <button onclick="send()">发送</button>
  </div>
  <div class="info">在线客服 | 试试问: 医师课程 · 价格 · 通过率 · 试听 · 人工</div>
</div>
<script>
const chatBox = document.getElementById('chatBox');
const input = document.getElementById('msgInput');
const userId = 'test_user_' + Date.now();

function addMsg(role, text, tags) {
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  let tagHtml = '';
  if (tags && tags.length) {
    tags.forEach(t => { tagHtml += '<span class="tag '+t.cls+'">'+t.label+'</span>'; });
  }
  div.innerHTML = tagHtml + text;
  chatBox.appendChild(div);
  chatBox.scrollTop = chatBox.scrollHeight;
}

async function send() {
  const msg = input.value.trim();
  if (!msg) return;
  addMsg('user', msg);
  input.value = '';

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({user_id: userId, message: msg})
    });
    const data = await res.json();
    const tags = [];
    if (data.fast_reply) tags.push({label:'快捷', cls:'fast'});
    if (data.needs_human) tags.push({label:'转人工', cls:'handoff'});
    if (data.sources && data.sources.length) tags.push({label:'知识库', cls:'rag'});
    addMsg('bot', data.answer, tags);
  } catch(e) {
    addMsg('bot', '网络出错，请重试～');
  }
}
</script>
</body>
</html>"""


@router.get("/", response_class=HTMLResponse)
async def chat_page():
    """返回聊天页面。"""
    return CHAT_HTML


@router.post("/api/chat", tags=["测试"])
async def chat_api(request: Request):
    """模拟聊天 API：用户消息 → 完整客服链路 → 回复。"""
    body = await request.json()
    user_id = body.get("user_id", "test_user")
    message = body.get("message", "").strip()

    if not message:
        return {"answer": "请输入消息内容～", "needs_human": False, "sources": []}

    chat_logger.log(user_id, "in", message)
    engine = None  # 延迟导入

    # 1. 快捷回复
    fast_reply = FastReplyManager()
    fast_text = fast_reply.match(message)
    if fast_text:
        chat_logger.log(user_id, "fast_reply", fast_text)
        return {"answer": fast_text, "needs_human": False, "sources": [], "fast_reply": True}

    # 2. 人工转接
    handoff = HumanHandoff()
    if handoff.should_handoff(message):
        msg = handoff.get_handoff_message()
        chat_logger.log(user_id, "human_handoff", msg)
        return {"answer": msg, "needs_human": True, "sources": []}

    # 3. 对话引擎（RAG + LLM）
    try:
        from core.chat_engine import ChatEngine
        engine = ChatEngine()
        result = engine.chat(user_id, message)
    except ValueError as e:
        # API Key 未配置等
        return {
            "answer": f"系统配置问题：{e}\n请检查 .env 文件中的 API Key 是否正确。",
            "needs_human": False,
            "sources": [],
        }
    except Exception as e:
        return {
            "answer": f"抱歉，系统遇到了一点小问题：{e}",
            "needs_human": False,
            "sources": [],
        }

    answer = result["answer"]

    # 4. 低置信度 → 转人工
    if handoff.should_fallback(answer):
        answer = handoff.get_handoff_message()
        chat_logger.log(user_id, "human_handoff_low_conf", answer)
        return {"answer": answer, "needs_human": True, "sources": result.get("sources", [])}

    # 5. 语气优化
    tone = ToneOptimizer()
    answer = tone.optimize(answer)

    chat_logger.log(user_id, "out", answer)
    return {
        "answer": answer,
        "needs_human": False,
        "sources": result.get("sources", []),
    }

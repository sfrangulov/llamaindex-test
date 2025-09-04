from rag_engine import configure_settings
import re
import json
from typing import Any, Dict, Callable, Optional
from llama_index.core import Settings
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.memory import Memory
from llama_index.core.tools import FunctionTool
from fs_utils import SECTION_TITLES

import structlog
log = structlog.get_logger(__name__)


DEV_STANDARDS = ""
FS_QUESTIONS = ""
FS_TEMPLATE = ""

# Agent state (lazy-inited)
AGENT: FunctionAgent | None = None
MEMORY: Memory | None = None


def split_by_sections_questions(text):
    """Split markdown text into sections by level-2 headings (## ...).

    Returns a dict mapping section title (without the '## ' prefix)
    to the section text (including the heading line) up to the next
    level-2 heading. Leading content before the first heading is ignored.
    """
    # Find all level-2 headings (lines starting with '## ')
    heading_re = re.compile(r"^##\s+.+$", re.MULTILINE)
    matches = list(heading_re.finditer(text))
    sections: dict[str, str] = {}
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk = text[start:end].strip()
        if chunk:
            sections[m.group().strip().removeprefix("## ")] = chunk
    return sections


def start():
    global DEV_STANDARDS, FS_QUESTIONS, FS_TEMPLATE
    configure_settings()
    with open("./fs_analyze/dev_standards.md", "r") as f:
        DEV_STANDARDS = f.read()
    with open("./fs_analyze/fs_questions.md", "r") as f:
        FS_QUESTIONS = split_by_sections_questions(f.read())
    with open("./fs_analyze/fs_template.md", "r") as f:
        FS_TEMPLATE = f.read()
    # Prebuild agent with memory for subsequent calls
    _ensure_agent()


def _mark_missing_section(title: str, results: dict[str, Dict[str, Any]], done: int, total: int, progress_cb: Optional[Callable[[int, int, str], None]]) -> int:
    details = (
        f"### {title}\n\n"
        f"Раздел отсутствует в документе. Проверьте, что он добавлен согласно шаблону ФС."
    )
    results[title] = {"ok": False, "details_markdown": details}
    done += 1
    try:
        if progress_cb:
            progress_cb(done, total, title)
    except Exception:
        pass
    return done


def _ensure_agent() -> FunctionAgent:
    """Initialize and cache a FunctionAgent with Memory.

    Keeps a small set of helper tools to let the model query standards/Qs.
    """
    global AGENT, MEMORY
    if AGENT is not None:
        return AGENT
    # Bounded memory so context doesn't explode across many sections
    try:
        MEMORY = Memory.from_defaults(token_limit=4000)
    except Exception:
        MEMORY = None

    # Provide tiny helper tools the agent can call if it wants more context
    def _tool_get_questions(section: str) -> str:
        try:
            log.debug("Getting FS questions", section=section)
            return (FS_QUESTIONS or {}).get(section, "")
        except Exception:
            return ""

    def _tool_get_standards(_: str = "") -> str:
        log.debug("Getting DEV standards")
        return DEV_STANDARDS or ""

    def _tool_get_template(_: str = "") -> str:
        log.debug("Getting FS template")
        return FS_TEMPLATE or ""

    tools: list[FunctionTool] = []
    try:
        tools = [
            FunctionTool.from_defaults(fn=_tool_get_questions, name="get_fs_questions",
                                       description="Вернуть контрольные вопросы для указанного раздела ФС."),
            FunctionTool.from_defaults(fn=_tool_get_standards, name="get_dev_standards",
                                       description="Вернуть стандарты разработки для справки при анализе."),
            FunctionTool.from_defaults(fn=_tool_get_template, name="get_fs_template",
                                       description="Вернуть шаблон функциональной спецификации для справки при анализе."),
        ]
    except Exception:
        tools = []

    system_prompt = (
        "Ты выступаешь в роли строгого ревьюера функциональных спецификаций (ФС). "
        "Отвечай кратко, по делу и на русском. Когда проситcя вывод JSON — возвращай строго JSON без лишних пояснений."
    )

    try:
        AGENT = FunctionAgent.from_tools(
            tools=tools,
            llm=Settings.llm,
            memory=MEMORY,
            verbose=False,
            system_prompt=system_prompt,
        )
    except Exception:
        # Fallback: build minimal agent without tools/memory
        try:
            AGENT = FunctionAgent.from_tools(tools=[], llm=Settings.llm)
        except Exception:
            AGENT = None
    return AGENT


def _resp_text(res: object) -> str:
    """Extract text from common LlamaIndex agent/LLM responses."""
    if res is None:
        return ""
    if isinstance(res, str):
        return res
    # Prefer common attributes in order
    for attr in ("output", "text", "message", "response"):
        val = getattr(res, attr, None)
        if isinstance(val, str):
            return val
        if getattr(val, "text", None):
            return str(getattr(val, "text"))
    return str(res)


def _agent_complete(prompt: str) -> str:
    """Try to complete via FunctionAgent; fallback to raw LLM on failure."""
    try:
        agent = _ensure_agent()
        if agent is not None:
            if hasattr(agent, "run"):
                return _resp_text(agent.run(prompt))
            if hasattr(agent, "chat"):
                return _resp_text(agent.chat(prompt))
    except Exception as e:
        log.warning("agent_complete_failed", error=e)
    # Hard fallback to raw LLM
    try:
        return _resp_text(Settings.llm.complete(prompt))
    except Exception:
        return ""


def _parse_json_loose(text: str) -> Dict[str, Any]:
    """Best-effort JSON parsing: strip code fences and trailing text."""
    s = (text or "").strip()
    # Remove Markdown code fences if present
    if s.startswith("```"):
        try:
            s = s.strip().strip("`")
            # After stripping fences, try to find first '{' and last '}'
        except Exception:
            pass
    # Find JSON object boundaries
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        s = s[start:end+1]
    try:
        return json.loads(s)
    except Exception:
        return {}


def analyze_fs_sections(
    fs_sections: dict[str, str],
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
) -> dict[str, Dict[str, Any]]:
    """Run LLM-based checklist analysis per FS section using FS_QUESTIONS.

    Returns mapping: section_title -> {
        'ok': bool,
        'details_markdown': str,
        'overall_assessment': str,  # один из: "Полностью соответствует" | "Частично соответствует" | "Не соответствует"
    }
    """

    results: dict[str, Dict[str, Any]] = {}
    total = len(SECTION_TITLES)
    done = 0
    for title in SECTION_TITLES:
        content = (fs_sections or {}).get(title, "").strip()
        questions_md = (FS_QUESTIONS or {}).get(title, "").strip()

        # If no section text found, mark as missing
        if not content:
            done = _mark_missing_section(
                title, results, done, total, progress_cb)
            continue

        if not questions_md:
            questions_md = "* Вопросы для этого раздела не найдены в fs_questions.md. Выполните свободное ревью на соответствие здравому смыслу и стандартам разработки."

        prompt = (
            "Ты выступаешь в роли ревьюера функциональной спецификации (ФС).\n"
            "Дан текст одного раздела ФС, список контрольных вопросов а так же шаблон ФС.\n"
            "Для каждого вопроса ответь кратко на русском: 'Да', 'Нет' или 'Неясно', и добавь короткий комментарий (1-2 предложения).\n"
            "Так же проверь раздел на соответствие шаблону ФС. Не проверяй ошибки в разметке Markdown и не учитывай их в оценке. Не проверяй пустые заголовки в таблицах.\n"
            "Если обнаружены проблемы, перечисли их в виде пунктов.\n"
            "Дай итоговую общую оценку соответствия раздела требованиям одним из вариантов: 'Полностью соответствует', 'Частично соответствует', 'Не соответствует'.\n"
            "Критерии: 'Полностью соответствует' — все критичные вопросы закрыты ('Да') и нет существенных замечаний; 'Частично соответствует' — есть неясности или мелкие замечания; 'Не соответствует' — есть существенные пробелы, ответы 'Нет' по ключевым вопросам или раздел явно слабый.\n"
            "Верни строго JSON без пояснений со следующими полями: details_markdown (string в Markdown), overall_assessment (string, одно из указанных значений).\n\n"
            f"Раздел: {title}\n\n"
            "Текст раздела (Markdown):\n" + content + "\n\n"
            "Контрольные вопросы (Markdown):\n" + questions_md + "\n\n"
            "Шаблон ФС можно получить инструментом get_fs_template.\n\n"
            "Можешь при необходимости вызывать инструменты get_fs_questions и get_dev_standards.\n\n"
            "Требования к краткости: пиши очень кратко и по делу."
        )
        try:
            text = _agent_complete(prompt)
            data = _parse_json_loose(text)
            log.debug("Analyzing FS section", section=title,
                      prompt=prompt, data=data)
        except Exception as e:
            log.warning("Section analysis failed", section=title, error=e)
            data = {}

        details_md = (data.get("details_markdown") or "(нет деталей)") if isinstance(
            data, dict) else "(анализ недоступен)"

        # Normalize/add overall assessment
        overall = (data.get("overall_assessment")
                   if isinstance(data, dict) else None) or ""
        overall_norm = (overall or "").strip().upper()

        results[title] = {
            "details_markdown": details_md,
            "overall_assessment": overall_norm,
        }

        # progress update after finishing this section
        done += 1
        try:
            if progress_cb:
                progress_cb(done, total, title)
        except Exception:
            pass

    return results

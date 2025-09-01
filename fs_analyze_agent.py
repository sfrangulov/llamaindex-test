from rag_engine import configure_settings
import asyncio
import re
import os
import json
from typing import Any, Dict
from llama_index.core import Settings
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.memory import Memory
from llama_index.core.tools import FunctionTool

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


def warmup():
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


def _ensure_agent() -> FunctionAgent:
    """Initialize and cache a FunctionAgent with Memory.

    Keeps a small set of helper tools to let the model query standards/Qs.
    """
    global AGENT, MEMORY
    if AGENT is not None:
        return AGENT
    configure_settings()
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
            FunctionTool.from_defaults(fn=_tool_get_questions, name="get_fs_questions", description="Вернуть контрольные вопросы для указанного раздела ФС."),
            FunctionTool.from_defaults(fn=_tool_get_standards, name="get_dev_standards", description="Вернуть стандарты разработки для справки при анализе."),
            FunctionTool.from_defaults(fn=_tool_get_template, name="get_fs_template", description="Вернуть шаблон функциональной спецификации для справки при анализе."),
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


def _agent_complete(prompt: str) -> str:
    """Try to complete via FunctionAgent; fallback to raw LLM on failure."""
    try:
        agent = _ensure_agent()
        if agent is not None:
            # Try common call surfaces across versions
            if hasattr(agent, "chat"):
                res = agent.chat(prompt)  # may return str or obj
                if isinstance(res, str):
                    return res
                # Try common attributes
                txt = getattr(res, "message", None) or getattr(res, "response", None) or getattr(res, "text", None)
                if hasattr(txt, "text"):
                    txt = getattr(txt, "text", None)
                if txt is not None:
                    return str(txt)
            if hasattr(agent, "run"):
                res = agent.run(prompt)
                txt = getattr(res, "output", None) or getattr(res, "text", None) or (str(res) if res is not None else None)
                if txt is not None:
                    return str(txt)
    except Exception as e:
        log.warning("agent_complete_failed", error=e)
    # Hard fallback to raw LLM
    try:
        resp = Settings.llm.complete(prompt)
        return getattr(resp, "text", "") or str(resp)
    except Exception:
        return ""


SECTION_TITLES = [
    'Лист изменений',
    'Глоссарий',
    'Глоссарий (опционально)',
    'Предмет разработки',
    'Релиз конфигурации',
    'Дизайн объектов Системы',
    'Описание общих алгоритмов',
    'Описание интеграционных интерфейсов',
    'Техническая реализация',
    'Настройки системы, используемые разработкой',
    'Тестовый сценарий',
]


def get_section_titles() -> list[str]:
    return list(SECTION_TITLES)


# Backwards/compat alias for external consumers expecting this name
section_titles = SECTION_TITLES


def split_by_sections_fs(text):
    canonical = {t.lower(): t for t in SECTION_TITLES}
    alts = "|".join(map(re.escape, SECTION_TITLES))
    pattern = rf'.?(?P<title>{alts})[*]*$'
    heading_re = re.compile(pattern, re.IGNORECASE |
                            re.VERBOSE | re.MULTILINE | re.DOTALL)

    matches = list(heading_re.finditer(text))
    result: dict[str, str] = {}
    for i, m in enumerate(matches):
        start = m.start()  # include the heading line in the chunk
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk = text[start:end].strip()
        if not chunk:
            continue
        key = canonical.get(m.group('title').lower(), m.group('title').strip())
        result[key] = '# ' + chunk
    return result


def get_fs(file_path: str) -> dict[str, str]:
    with open(file_path, "r") as f:
        md = f.read()
        return split_by_sections_fs(md)


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


def analyze_fs_sections(fs_sections: dict[str, str]) -> dict[str, Dict[str, Any]]:
    """Run LLM-based checklist analysis per FS section using FS_QUESTIONS.

    Returns mapping: section_title -> {
        'ok': bool,
        'issues_count': int,
        'summary': str,
        'details_markdown': str,
        'overall_assessment': str,  # один из: "полностью соответствует" | "частично соответствует" | "не соответствует"
    }
    """
    # Ensure settings and questions are ready
    if not isinstance(FS_QUESTIONS, dict) or not FS_QUESTIONS:
        try:
            warmup()
        except Exception:
            pass
    configure_settings()

    results: dict[str, Dict[str, Any]] = {}
    for title in SECTION_TITLES:
        content = (fs_sections or {}).get(title, "").strip()
        questions_md = (FS_QUESTIONS or {}).get(title, "").strip()

        # If no section text found, mark as missing
        if not content:
            details = (
                f"### {title}\n\n"
                f"Раздел отсутствует в документе. Проверьте, что он добавлен согласно шаблону ФС."
            )
            results[title] = {
                "ok": False,
                "issues_count": 1,
                "summary": "Раздел отсутствует",
                "details_markdown": details,
            }
            continue

        if not questions_md:
            questions_md = "* Вопросы для этого раздела не найдены в fs_questions.md. Выполните свободное ревью на соответствие здравому смыслу и стандартам разработки."

        prompt = (
            "Ты выступаешь в роли ревьюера функциональной спецификации (ФС).\n"
            "Дан текст одного раздела ФС, список контрольных вопросов а так же шаблон ФС.\n"
            "Для каждого вопроса ответь кратко на русском: 'Да', 'Нет' или 'Неясно', и добавь короткий комментарий (1-2 предложения).\n"
            "Так же проверь раздел на соответствие шаблону ФС.\n",
            "Если обнаружены проблемы, перечисли их в виде пунктов.\n"
            "Дай итоговую общую оценку соответствия раздела требованиям одним из вариантов: 'полностью соответствует', 'частично соответствует', 'не соответствует'.\n"
            "Критерии: 'полностью соответствует' — все критичные вопросы закрыты ('Да') и нет существенных замечаний; 'частично соответствует' — есть неясности или мелкие замечания; 'не соответствует' — есть существенные пробелы, ответы 'Нет' по ключевым вопросам или раздел явно слабый.\n"
            "Верни строго JSON без пояснений со следующими полями: ok (bool), issues_count (int), summary (string), details_markdown (string в Markdown), overall_assessment (string, одно из указанных значений).\n\n"
            f"Раздел: {title}\n\n"
            "Текст раздела (Markdown):\n" + content + "\n\n"
            "Контрольные вопросы (Markdown):\n" + questions_md + "\n\n"
            "Шаблон ФС можно получить инструментом get_fs_template.\n\n"
            "Можешь при необходимости вызывать инструменты get_fs_questions и get_dev_standards.\n\n"
            "Требования к краткости: пиши очень кратко и по делу."
        )
        try:
            # resp = Settings.llm.complete(prompt)
            # data = _parse_json_loose(getattr(resp, "text", ""))
            text = _agent_complete(prompt)
            data = _parse_json_loose(text)
            log.debug("Analyzing FS section", section=title, prompt=prompt, data=data)
        except Exception as e:
            log.warning("Section analysis failed", section=title, error=e)
            data = {}

        ok = bool(data.get("ok")) if isinstance(data, dict) else False
        issues_count = int(data.get("issues_count", 0)
                           or 0) if isinstance(data, dict) else 0
        details_md = (data.get("details_markdown") or "(нет деталей)") if isinstance(
            data, dict) else "(анализ недоступен)"
        if not data:
            # Fallback heuristic: if content length is small, likely issues
            ok = len(content) > 200
            issues_count = 0 if ok else 1
            details_md = f"### {title}\n\n(Не удалось выполнить автоматический анализ. Проверьте раздел вручную.)"

        summary = data.get("summary") if isinstance(data, dict) else None
        if not summary:
            if ok and issues_count == 0:
                summary = "OK"
            elif issues_count:
                summary = f"⚠️ {issues_count} замечания"
            else:
                summary = "Есть замечания"

        # Normalize/add overall assessment
        overall = (data.get("overall_assessment")
                   if isinstance(data, dict) else None) or ""
        overall_norm = (overall or "").strip().lower()
        if not overall_norm:
            # Derive from ok/issues_count if model didn't return the field
            if not content:
                overall_norm = "не соответствует"
            elif ok and issues_count == 0:
                overall_norm = "полностью соответствует"
            elif ok and issues_count > 0:
                overall_norm = "частично соответствует"
            else:
                overall_norm = "частично соответствует" if len(
                    content) > 200 else "не соответствует"

        results[title] = {
            "ok": ok,
            "issues_count": issues_count,
            "summary": summary,
            "details_markdown": details_md,
            "overall_assessment": overall_norm,
        }

    return results


if __name__ == "__main__":
    warmup()
    stats = {}
    files = os.listdir("./data/markdown/")
    for fname in files:
        fs_sections = get_fs(f"./data/markdown/{fname}")
        stats[len(fs_sections)] = stats.get(len(fs_sections), 0) + 1
    log.info("FS sections stats", total=len(files), stats=stats)

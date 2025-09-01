from rag_engine import configure_settings
import asyncio
import re
import os
from llama_index.core.agent.workflow import FunctionAgent

import structlog
log = structlog.get_logger(__name__)


DEV_STANDARDS = ""
FS_QUESTIONS = ""
FS_TEMPLATE = ""


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


def split_by_sections_fs(text):
    section_titles = [
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
    canonical = {t.lower(): t for t in section_titles}
    alts = "|".join(map(re.escape, section_titles))
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
        result[key] = chunk
    return result


def get_fs(file_path: str) -> dict[str, str]:
    with open(file_path, "r") as f:
        md = f.read()
        return split_by_sections_fs(md)


if __name__ == "__main__":
    warmup()
    stats = {}
    for file in os.listdir("./data/markdown/"):
        fs_sections = get_fs(f"./data/markdown/{file}")
        stats[len(fs_sections)] = stats.get(len(fs_sections), 0) + 1
    log.info(f"FS sections stats", len=len(files), stats=stats)
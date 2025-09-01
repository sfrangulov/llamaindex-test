import asyncio
import re
from llama_index.core.agent.workflow import FunctionAgent

from rag_engine import configure_settings

DEV_STANDARDS = ""
FS_QUESTIONS = ""
FS_TEMPLATE = ""

def split_by_sections(text):
    """Split markdown text into sections by level-2 headings (## ...).

    Returns a list of section strings, each including its heading line
    and the content up to (but not including) the next level-2 heading.
    Leading content before the first level-2 heading is ignored.
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
        FS_QUESTIONS = split_by_sections(f.read())
    with open("./fs_analyze/fs_template.md", "r") as f:
        FS_TEMPLATE = f.read()

if __name__ == "__main__":
    warmup()
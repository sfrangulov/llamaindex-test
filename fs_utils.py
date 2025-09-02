import re

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

# SECTION_TITLES = [
#     'Тестовый сценарий',
# ]

def get_section_titles() -> list[str]:
    return list(SECTION_TITLES)

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
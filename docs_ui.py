import os
from typing import List, Tuple

import gradio as gr

from storage import (
    list_storage_files,
    add_docx_to_store,
    list_vector_file_names,
    delete_from_vector_store_by_file_names,
    read_markdown,
    ensure_dirs,
    CFG,
)
from rag_engine import warmup
from gradio_modal import Modal


DOCX_EXT = ".docx"


def _format_size(n: int | None) -> str:
    if not n:
        return "-"
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def _load_table(search: str, sort_by: str, sort_dir: str) -> List[List[str]]:
    rows = list_storage_files(search)
    # sort_by in ["name", "size", "date"]
    key_map = {
        "name": lambda r: r["file_name"].lower(),
        "size": lambda r: r.get("size_bytes") or 0,
        "date": lambda r: r.get("uploaded_at_iso") or "",
    }
    key = key_map.get(sort_by, key_map["name"])  # default by name
    rev = sort_dir == "desc"
    rows.sort(key=key, reverse=rev)
    table: List[List[str]] = []
    for r in rows:
        table.append([
            r["file_name"],
            _format_size(r.get("size_bytes")),
            r.get("uploaded_at_iso") or "",
        ])
    return table


def _vector_list() -> List[str]:
    return list_vector_file_names()


def _on_upload(files: List[gr.File]) -> str:
    if not files:
        return "Файлов не найдено"
    ensure_dirs()
    added = []

    def _extract_path(fobj) -> str | None:
        # Gradio may return TempFile objects or dicts; handle both
        if hasattr(fobj, "name") and isinstance(getattr(fobj, "name"), str):
            return getattr(fobj, "name")
        if hasattr(fobj, "path") and isinstance(getattr(fobj, "path"), str):
            return getattr(fobj, "path")
        if isinstance(fobj, dict):
            return fobj.get("path") or fobj.get("name")
        return None

    for f in files:
        p = _extract_path(f)
        if not p:
            added.append("Не удалось определить путь к файлу")
            continue
        if not p.lower().endswith(DOCX_EXT):
            continue
        try:
            abs_path = os.path.abspath(p)
            info = add_docx_to_store(abs_path)
            added.append(info.get("file_name") or os.path.basename(abs_path))
        except Exception as e:
            added.append(f"Ошибка: {os.path.basename(p)} — {e}")
    if not added:
        return f"Нет {DOCX_EXT} файлов"
    return "\n".join(["Загружено:"] + added)


def _on_delete(selected: List[str] | None) -> Tuple[str, List[str]]:
    names = selected or []
    if not names:
        return "Не выбрано ни одного файла", _vector_list()
    count = delete_from_vector_store_by_file_names(names)
    return f"Удалено: {count}", _vector_list()


def _on_preview(file_name: str) -> str:
    if not file_name:
        return "Выберите файл из списка слева"
    # file_name is expected to be like "X.docx" since we list docx names
    return read_markdown(file_name)


with gr.Blocks(title="Документы") as app:
    gr.Markdown(
        f"""
		# Управление документами

		- Хранилище исходных файлов: `{CFG.data_path}`
		- Markdown сохраняется в: `{CFG.md_dir}`
		- Векторное хранилище: `{CFG.chroma_path}` / коллекция `{CFG.collection}`
		"""
    )

    with gr.Tabs():
        with gr.TabItem("Список файлов"):
            with gr.Row():
                with gr.Column(scale=3):
                    search = gr.Textbox(
                        label="Поиск по имени", placeholder="введите часть названия")
                with gr.Column(scale=2):
                    sort_by = gr.Dropdown(
                        ["name", "size", "date"], value="name", label="Сортировка по")
                with gr.Column(scale=1):
                    sort_dir = gr.Dropdown(
                        ["asc", "desc"], value="asc", label="Порядок")
            table = gr.Dataframe(
                headers=["Название", "Размер", "Дата"],
                column_widths=["70%", "10%", "20%"],
                interactive=False,
            )
            refresh_btn = gr.Button("Обновить")

            # Управление действиями над выбранным файлом
            with gr.Row():
                selected_name = gr.Dropdown(choices=[], label="Выберите файл", allow_custom_value=False)
                btn_preview = gr.Button("Предпросмотр")
                btn_delete = gr.Button("Удалить")

            def refresh_all(s, sb, sd):
                tbl = _load_table(s or "", sb or "name", sd or "asc")
                choices = [row[0] for row in tbl]
                value = choices[0] if choices else None
                return tbl, gr.update(choices=choices, value=value)

            search.input(refresh_all, [search, sort_by, sort_dir], [table, selected_name])
            sort_by.change(refresh_all, [search, sort_by, sort_dir], [table, selected_name])
            sort_dir.change(refresh_all, [search, sort_by, sort_dir], [table, selected_name])
            refresh_btn.click(refresh_all, [search, sort_by, sort_dir], [table, selected_name])

            # Автозагрузка данных в таблицу при старте приложения
            app.load(refresh_all, [search, sort_by, sort_dir], [table, selected_name])

            # Preview modal using gradio_modal
            with Modal(visible=False) as preview_modal:
                preview_md = gr.Markdown("Загрузка…")

            # Confirm deletion modal
            with Modal(visible=False) as confirm_modal:
                confirm_text = gr.Markdown("Подтвердите удаление")
                with gr.Row():
                    confirm_yes = gr.Button("Да")
                    confirm_no = gr.Button("Нет")

            # Helper state and status
            pending_delete = gr.State("")
            status_md = gr.Markdown("")

            def _open_preview():
                return Modal(visible=True)

            def _open_confirm():
                return Modal(visible=True)

            def _close_confirm():
                return Modal(visible=False)

            def _confirm_delete(name: str):
                if not name:
                    return gr.update(value="Имя файла не задано"), Modal(visible=False), gr.update(value="")
                count = delete_from_vector_store_by_file_names([name])
                msg = f"Удалено из векторного хранилища: {name} (записей: {count})"
                return gr.update(value=msg), Modal(visible=False), gr.update(value="")

            # Buttons for actions
            def _on_preview_click(name: str):
                if not name:
                    return gr.update(value="Выберите файл"), gr.skip()
                return gr.update(value=_on_preview(name)), _open_preview()

            def _on_delete_click(name: str):
                if not name:
                    return gr.update(value="Выберите файл"), gr.skip(), gr.update(value="")
                return gr.update(value=f"Удалить из векторного хранилища: {name}?"), _open_confirm(), gr.update(value=name)

            btn_preview.click(_on_preview_click, [selected_name], [preview_md, preview_modal])
            btn_delete.click(_on_delete_click, [selected_name], [confirm_text, confirm_modal, pending_delete])

            # Confirm modal buttons
            confirm_yes.click(
                _confirm_delete,
                [pending_delete],
                [status_md, confirm_modal, pending_delete],
            ).then(refresh_all, [search, sort_by, sort_dir], [table, selected_name])
            confirm_no.click(_close_confirm, None, confirm_modal)

    # (клик по ячейке не используется; действуйте через выпадающий список и кнопки)

        with gr.TabItem("Загрузка и индексация"):
            gr.Markdown(
                "Загрузите один или несколько .docx файлов — они будут конвертированы в Markdown и добавлены в векторное хранилище.")
            uploader = gr.File(file_count="multiple", file_types=[
                               ".docx"], label="Выберите .docx")
            upload_status = gr.Textbox(label="Статус", interactive=False)
            upload_btn = gr.Button("Индексировать")
            upload_btn.click(_on_upload, [uploader], [upload_status])


if __name__ == "__main__":
    warmup()
    app.queue().launch(server_name=os.getenv("HOST", "0.0.0.0"),
                       server_port=int(os.getenv("PORT", "7861")))

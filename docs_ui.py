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

# Lightweight CSS modal overlay (fallback when gr.Modal isn't available)
CSS = """
.modal-overlay { position: fixed; inset: 0; background: rgba(0,0,0,.6); z-index: 9999; display: flex; align-items: center; justify-content: center; padding: 2rem; }
.modal-overlay .modal-content { background: var(--block-background-fill); color: inherit; max-width: min(1200px, 95vw); max-height: 85vh; overflow: auto; border-radius: 12px; padding: 1rem 1.25rem; box-shadow: 0 10px 30px rgba(0,0,0,.3); }
.modal-actions { display:flex; justify-content:flex-end; gap:.5rem; margin-bottom:.5rem; }
"""

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


with gr.Blocks(title="Документы (хранилище и индекс)", css=CSS) as app:
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
					search = gr.Textbox(label="Поиск по имени", placeholder="введите часть названия")
				with gr.Column(scale=2):
					sort_by = gr.Dropdown(["name", "size", "date"], value="name", label="Сортировка по")
				with gr.Column(scale=1):
					sort_dir = gr.Dropdown(["asc", "desc"], value="asc", label="Порядок")
			table = gr.Dataframe(headers=["Название", "Размер", "Дата"], interactive=True)
			refresh_btn = gr.Button("Обновить")

			def refresh_table(s, sb, sd):
				return _load_table(s or "", sb or "name", sd or "asc")

			search.input(refresh_table, [search, sort_by, sort_dir], [table])
			sort_by.change(refresh_table, [search, sort_by, sort_dir], [table])
			sort_dir.change(refresh_table, [search, sort_by, sort_dir], [table])
			refresh_btn.click(refresh_table, [search, sort_by, sort_dir], [table])

			# Overlay preview (CSS-based modal)
			with gr.Group(visible=False, elem_classes=["modal-overlay"]) as preview_overlay:
				with gr.Column(elem_classes=["modal-content"]):
					with gr.Row(elem_classes=["modal-actions"]):
						close_btn = gr.Button("Закрыть")
					preview_md = gr.Markdown("Загрузка…")

			# Open modal when clicking on the first column (Название)
			def on_cell_select(evt: gr.SelectData):
				try:
					# evt.index expected as (row, col) in SelectData
					idx = getattr(evt, "index", None) or (None, None)
					row, col = idx[0], idx[1]
					if row is None or col is None:
						return gr.skip(), gr.skip()
					if int(col) != 0:
						return gr.skip(), gr.skip()
					# For first column, evt.value should be the file name
					file_name = getattr(evt, "value", None)
					if not isinstance(file_name, str) or not file_name:
						return gr.skip(), gr.skip()
					md = _on_preview(file_name)
					# Show overlay with content
					return gr.update(value=md), gr.update(visible=True)
				except Exception as e:
					return gr.update(value=f"Ошибка открытия: {e}"), gr.update(visible=True)

			table.select(on_cell_select, None, [preview_md, preview_overlay])

			# Close overlay
			close_btn.click(lambda: gr.update(visible=False), None, [preview_overlay])

		with gr.TabItem("Загрузка и индексация"):
			gr.Markdown("Загрузите один или несколько .docx файлов — они будут конвертированы в Markdown и добавлены в векторное хранилище.")
			uploader = gr.File(file_count="multiple", file_types=[".docx"], label="Выберите .docx")
			upload_status = gr.Textbox(label="Статус", interactive=False)
			upload_btn = gr.Button("Индексировать")
			upload_btn.click(_on_upload, [uploader], [upload_status])

		with gr.TabItem("Удаление из векторного хранилища"):
			gr.Markdown("Отметьте имена файлов для удаления (по метаданному file_name).")
			vec_list = gr.CheckboxGroup(choices=_vector_list(), label="Файлы в индексе")
			del_btn = gr.Button("Удалить отмеченные")
			del_status = gr.Textbox(label="Статус", interactive=False)
			del_btn.click(_on_delete, [vec_list], [del_status, vec_list])
			refresh_vec = gr.Button("Обновить список")
			refresh_vec.click(lambda: gr.update(choices=_vector_list()), None, [vec_list])

	# (Просмотр перенесён в таблицу — отдельная вкладка больше не нужна)


if __name__ == "__main__":
	warmup()
	app.queue().launch(server_name=os.getenv("HOST", "0.0.0.0"), server_port=int(os.getenv("PORT", "7861")))


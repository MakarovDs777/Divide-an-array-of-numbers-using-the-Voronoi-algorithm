import random
from typing import List, Tuple
import numpy as np
import os
from pathlib import Path
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox


def partition_voronoi(values: List[float],
                      n_seeds: int,
                      lloyd_iters: int = 0,
                      init_method: str = "random") -> Tuple[List[List[float]], np.ndarray]:
    """
    Разбивает список чисел values на n_seeds наборов методом 1D-Voronoi с опциональными итерациями Ллойда.
    Возвращает:
      clusters - список длины n_seeds; каждый элемент — список чисел, принадлежащих соответствему кластеру
      seeds    - массив позиций сайтов (float), длины n_seeds
    """
    if n_seeds <= 0:
        raise ValueError("n_seeds must be >= 1")
    if len(values) == 0:
        return [[] for _ in range(n_seeds)], np.array([])

    arr = np.array(values, dtype=float)

    # Инициализация сайтов
    if init_method == "quantile":
        sorted_vals = np.sort(arr)
        probs = np.linspace(0, 1, n_seeds + 2)[1:-1]
        seeds = np.array([np.quantile(sorted_vals, p) for p in probs], dtype=float)
    else:
        lo, hi = arr.min(), arr.max()
        seeds = np.array([random.uniform(lo, hi) for _ in range(n_seeds)], dtype=float)

    # Итерации Ллойда
    for _ in range(int(lloyd_iters)):
        idx = np.abs(arr[:, None] - seeds[None, :]).argmin(axis=1)
        new_seeds = seeds.copy()
        moved = False
        for s in range(n_seeds):
            members = arr[idx == s]
            if members.size > 0:
                new_pos = members.mean()
                if new_pos != seeds[s]:
                    moved = True
                new_seeds[s] = new_pos
        seeds = new_seeds
        if not moved:
            break

    # Финальное разбиение
    idx = np.abs(arr[:, None] - seeds[None, :]).argmin(axis=1)
    clusters: List[List[float]] = [[] for _ in range(n_seeds)]
    for val, k in zip(values, idx):
        clusters[int(k)].append(val)

    return clusters, seeds


# ----------------- GUI -----------------

class VoronoiApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("1D Voronoi — конвертер")
        self.geometry("700x520")
        self.create_widgets()

    def create_widgets(self):
        pad = 8

        # Ввод чисел
        lbl_values = ttk.Label(self, text="Введите числа (через пробел, запятую или новую строку):")
        lbl_values.pack(anchor="w", pady=(pad, 0), padx=pad)

        self.txt_values = tk.Text(self, height=8)
        self.txt_values.pack(fill="x", padx=pad)
        self.txt_values.insert("1.0", "0 1 2 3 4 5 6 7 8 9")

        frm_params = ttk.Frame(self)
        frm_params.pack(fill="x", padx=pad, pady=(6, 0))

        # Число сайтов
        ttk.Label(frm_params, text="Количество кластеров (n_seeds):").grid(row=0, column=0, sticky="w")
        self.spin_n = tk.Spinbox(frm_params, from_=1, to=100, width=6)
        self.spin_n.delete(0, "end")
        self.spin_n.insert(0, "3")
        self.spin_n.grid(row=0, column=1, sticky="w", padx=(6, 12))

        # Итерации Ллойда
        ttk.Label(frm_params, text="Итераций Ллойда:").grid(row=0, column=2, sticky="w")
        self.spin_lloyd = tk.Spinbox(frm_params, from_=0, to=100, width=6)
        self.spin_lloyd.delete(0, "end")
        self.spin_lloyd.insert(0, "10")
        self.spin_lloyd.grid(row=0, column=3, sticky="w", padx=(6, 12))

        # Метод инициализации
        ttk.Label(frm_params, text="Метод инициализации:").grid(row=0, column=4, sticky="w")
        self.init_var = tk.StringVar(value="quantile")
        cmb = ttk.Combobox(frm_params, textvariable=self.init_var, values=["quantile", "random"], state="readonly", width=10)
        cmb.grid(row=0, column=5, sticky="w")

        # Кнопки
        frm_buttons = ttk.Frame(self)
        frm_buttons.pack(fill="x", padx=pad, pady=(12, 0))

        btn_convert = ttk.Button(frm_buttons, text="Конвертировать и сохранить на рабочий стол", command=self.on_convert)
        btn_convert.pack(side="left", padx=(0, 6))

        btn_preview = ttk.Button(frm_buttons, text="Предпросмотр (не сохранять)", command=self.on_preview)
        btn_preview.pack(side="left")

        # Результат
        lbl_res = ttk.Label(self, text="Результат:")
        lbl_res.pack(anchor="w", pady=(12, 0), padx=pad)

        self.txt_result = tk.Text(self, height=10)
        self.txt_result.pack(fill="both", expand=True, padx=pad, pady=(0, pad))
        self.txt_result.config(state="disabled")

    def parse_values(self) -> List[float]:
        raw = self.txt_values.get("1.0", "end").strip()
        if not raw:
            return []
        # Replace commas and newlines with spaces, then split
        for ch in [",", "\n", "\t", ";"]:
            raw = raw.replace(ch, " ")
        parts = [p for p in raw.split(" ") if p != ""]
        vals = []
        for p in parts:
            try:
                vals.append(float(p))
            except ValueError:
                raise ValueError(f"Невозможно преобразовать '{p}' в число")
        return vals

    def get_desktop_path(self) -> Path:
        p = Path.home() / "Desktop"
        if p.exists():
            return p
        # fallback to home if нет Desktop
        return Path.home()

    def format_output(self, clusters: List[List[float]], seeds: np.ndarray) -> str:
        lines = []
        lines.append(f"Seeds: {np.array2string(seeds, precision=6, separator=', ')}")
        for i, c in enumerate(clusters):
            # сохраняем элементы как в исходном формате (без лишних .0 если целые)
            items = [self._format_num(x) for x in c]
            lines.append(f"Cluster {i} ({len(c)}): {', '.join(items)}")
        return "\n".join(lines)

    def _format_num(self, x: float) -> str:
        if float(x).is_integer():
            return str(int(x))
        else:
            return str(x)

    def save_to_desktop(self, content: str) -> Path:
        desk = self.get_desktop_path()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"voronoi_1d_{ts}.txt"
        path = desk / fname
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    def on_convert(self):
        try:
            vals = self.parse_values()
        except ValueError as e:
            messagebox.showerror("Ошибка ввода", str(e))
            return
        try:
            n = int(self.spin_n.get())
            lloyd = int(self.spin_lloyd.get())
            init = self.init_var.get()
        except Exception:
            messagebox.showerror("Ошибка параметров", "Проверьте параметры n_seeds и lloyd_iters")
            return

        if n > len(vals):
            if not messagebox.askyesno("Мало чисел", "Количество кластеров больше количества чисел. Продолжить?\n(пустые кластеры будут созданы)"):
                return

        clusters, seeds = partition_voronoi(vals, n, lloyd_iters=lloyd, init_method=init)
        out = self.format_output(clusters, seeds)

        # Сохранить на рабочий стол
        try:
            path = self.save_to_desktop(out)
        except Exception as e:
            messagebox.showerror("Ошибка сохранения", f"Не удалось сохранить файл: {e}")
            return

        self._show_result(out)
        messagebox.showinfo("Сохранено", f"Файл сохранён:\n{path}")

    def on_preview(self):
        try:
            vals = self.parse_values()
        except ValueError as e:
            messagebox.showerror("Ошибка ввода", str(e))
            return
        try:
            n = int(self.spin_n.get())
            lloyd = int(self.spin_lloyd.get())
            init = self.init_var.get()
        except Exception:
            messagebox.showerror("Ошибка параметров", "Проверьте параметры n_seeds и lloyd_iters")
            return

        clusters, seeds = partition_voronoi(vals, n, lloyd_iters=lloyd, init_method=init)
        out = self.format_output(clusters, seeds)
        self._show_result(out)

    def _show_result(self, text: str):
        self.txt_result.config(state="normal")
        self.txt_result.delete("1.0", "end")
        self.txt_result.insert("1.0", text)
        self.txt_result.config(state="disabled")


if __name__ == "__main__":
    app = VoronoiApp()
    app.mainloop()

import random
from typing import List, Tuple
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog

# Опционально: matplotlib для визуализации
try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False

def partition_voronoi(values: List[float],
                      n_seeds: int,
                      lloyd_iters: int = 0,
                      init_method: str = "random") -> Tuple[List[List[float]], np.ndarray]:
    """
    1D Voronoi + Lloyd iterations.
    Возвращает (clusters, seeds).
    """
    if n_seeds <= 0:
        raise ValueError("n_seeds must be >= 1")
    if len(values) == 0:
        return [[] for _ in range(n_seeds)], np.array([])

    arr = np.array(values, dtype=float)

    # Инициализация сайтов
    if init_method == "quantile":
        sorted_vals = np.sort(arr)
        probs = np.linspace(0, 1, n_seeds + 2)[1:-1]  # внутренние квантили
        seeds = np.array([np.quantile(sorted_vals, p) for p in probs], dtype=float)
    else:
        lo, hi = arr.min(), arr.max()
        if lo == hi:
            seeds = np.array([lo for _ in range(n_seeds)], dtype=float)
        else:
            seeds = np.array([random.uniform(lo, hi) for _ in range(n_seeds)], dtype=float)

    # Lloyd iterations
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

    # Финальное разбиение (с сохранением порядка исходных значений)
    idx = np.abs(arr[:, None] - seeds[None, :]).argmin(axis=1)
    clusters: List[List[float]] = [[] for _ in range(n_seeds)]
    for val, k in zip(values, idx):
        clusters[int(k)].append(val)

    return clusters, seeds

# ---------- GUI ----------
class VoronoiApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("1D Voronoi / Lloyd — генератор кластеров")
        self.geometry("960x680")
        self._build_ui()
        self.last_result = None  # (clusters, seeds)

    def _build_ui(self):
        frm = ttk.Frame(self)
        frm.pack(fill="both", expand=True, padx=10, pady=10)

        # Левая панель: ввод и параметры
        left = ttk.Frame(frm)
        left.pack(side="left", fill="y", padx=(0,10))

        ttk.Label(left, text="Значения (через запятую или пробел):").pack(anchor="w")
        self.values_text = scrolledtext.ScrolledText(left, width=40, height=10)
        self.values_text.pack(pady=4)
        self.values_text.insert("1.0", "0 1 2 3 4 5 6 7 8 9")  # пример

        params = ttk.LabelFrame(left, text="Параметры")
        params.pack(fill="x", pady=6)

        row = ttk.Frame(params)
        row.pack(fill="x", pady=3)
        ttk.Label(row, text="Число сайтов (n_seeds):").pack(side="left")
        self.n_seeds_var = tk.IntVar(value=3)
        ttk.Entry(row, textvariable=self.n_seeds_var, width=8).pack(side="left", padx=6)

        row2 = ttk.Frame(params)
        row2.pack(fill="x", pady=3)
        ttk.Label(row2, text="Итерации Ллойда:").pack(side="left")
        self.lloyd_var = tk.IntVar(value=10)
        ttk.Entry(row2, textvariable=self.lloyd_var, width=8).pack(side="left", padx=6)

        row3 = ttk.Frame(params)
        row3.pack(fill="x", pady=3)
        ttk.Label(row3, text="Инициализация:").pack(side="left")
        self.init_var = tk.StringVar(value="quantile")
        ttk.Radiobutton(row3, text="quantile", variable=self.init_var, value="quantile").pack(side="left", padx=6)
        ttk.Radiobutton(row3, text="random", variable=self.init_var, value="random").pack(side="left", padx=6)

        # Кнопки действий
        actions = ttk.Frame(left)
        actions.pack(fill="x", pady=6)
        ttk.Button(actions, text="Запустить", command=self.on_run).pack(side="left", padx=4)
        ttk.Button(actions, text="Очистить вывод", command=self.on_clear_output).pack(side="left", padx=4)

        # Кнопки сохранения: TXT и CSV
        save_frame = ttk.Frame(left)
        save_frame.pack(fill="x", pady=6)
        ttk.Button(save_frame, text="Сохранить как TXT", command=self.on_save_txt).pack(side="left", padx=4)
        ttk.Button(save_frame, text="Сохранить CSV", command=self.on_save_csv).pack(side="left", padx=4)

        # Правая панель: вывод и визуализация
        right = ttk.Frame(frm)
        right.pack(side="left", fill="both", expand=True)

        output_box = ttk.LabelFrame(right, text="Результат")
        output_box.pack(fill="both", expand=True, padx=4, pady=4)

        self.output_text = scrolledtext.ScrolledText(output_box, width=80, height=28)
        self.output_text.pack(fill="both", expand=True, padx=6, pady=6)

        # Визуализация (если matplotlib доступен)
        viz_frame = ttk.LabelFrame(right, text="Визуализация (опционально)")
        viz_frame.pack(fill="both", expand=False, padx=4, pady=(0,4))
        if MATPLOTLIB_AVAILABLE:
            self.fig, self.ax = plt.subplots(figsize=(7,2.6))
            self.canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
            self.canvas.get_tk_widget().pack(fill="both", expand=True)
        else:
            ttk.Label(viz_frame, text="Matplotlib не установлен — графика недоступна.").pack(padx=6, pady=10)

    def parse_values(self) -> List[float]:
        raw = self.values_text.get("1.0", "end").strip()
        if not raw:
            return []
        parts = []
        # заменим запятые на пробелы и разобьём
        tokens = raw.replace(",", " ").split()
        for token in tokens:
            # поддержка простых диапазонов вида 1-5
            if "-" in token and token.count("-") == 1:
                a, b = token.split("-")
                try:
                    a_f = float(a); b_f = float(b)
                    if a_f.is_integer() and b_f.is_integer():
                        a_i, b_i = int(a_f), int(b_f)
                        if a_i <= b_i:
                            parts.extend(list(range(a_i, b_i + 1)))
                            continue
                except Exception:
                    pass
            try:
                parts.append(float(token))
            except ValueError:
                raise ValueError(f"Не удалось разобрать токен: '{token}'")
        return parts

    def on_run(self):
        try:
            values = self.parse_values()
        except Exception as e:
            messagebox.showerror("Ошибка ввода", str(e))
            return
        if len(values) == 0:
            messagebox.showinfo("Пусто", "Введите хотя бы одно значение.")
            return
        n_seeds = int(self.n_seeds_var.get())
        lloyd_iters = int(self.lloyd_var.get())
        init_method = self.init_var.get()
        try:
            clusters, seeds = partition_voronoi(values, n_seeds, lloyd_iters=lloyd_iters, init_method=init_method)
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))
            return

        self.last_result = (clusters, seeds)

        # Формируем текстовый вывод
        out_lines = []
        out_lines.append(f"Метод инициализации: {init_method}")
        out_lines.append(f"Итерации Ллойда: {lloyd_iters}")
        out_lines.append(f"Число сайтов: {n_seeds}")
        out_lines.append("-" * 60)
        out_lines.append("Seeds (позиции сайтов):")
        for i, s in enumerate(seeds):
            out_lines.append(f"  {i}: {s}")
        out_lines.append("-" * 60)
        for i, c in enumerate(clusters):
            out_lines.append(f"Cluster {i} ({len(c)}): {c}")

        self.output_text.delete("1.0", "end")
        self.output_text.insert("end", "\n".join(out_lines))

        # Визуализация (если доступна)
        if MATPLOTLIB_AVAILABLE:
            self._plot_result(values, clusters, seeds)

    def _plot_result(self, values, clusters, seeds):
        self.ax.clear()
        colors = plt.cm.get_cmap("tab10")
        for i, cluster in enumerate(clusters):
            if len(cluster) == 0:
                continue
            xs = np.array(cluster, dtype=float)
            ys = np.zeros_like(xs) + 0.05 * i
            self.ax.scatter(xs, ys, label=f"cluster {i}", color=colors(i % 10), s=40)
        self.ax.scatter(seeds, np.full_like(seeds, -0.02), marker="x", color="k", s=80, label="seeds")
        self.ax.set_yticks([])
        self.ax.set_xlabel("Value")
        self.ax.set_title("1D Voronoi partition")
        self.ax.legend(loc="upper right", fontsize="small")
        self.fig.tight_layout()
        self.canvas.draw()

    def on_clear_output(self):
        self.output_text.delete("1.0", "end")

    def on_save_csv(self):
        if not self.last_result:
            messagebox.showinfo("Нет данных", "Сначала выполните разбиение.")
            return
        clusters, seeds = self.last_result
        path = filedialog.asksaveasfilename(defaultextension=".csv",
                                            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("cluster_id,value\n")
                for i, cluster in enumerate(clusters):
                    for v in cluster:
                        f.write(f"{i},{v}\n")
            messagebox.showinfo("Сохранено", f"Кластеры сохранены в {path}")
        except Exception as e:
            messagebox.showerror("Ошибка сохранения", str(e))

    def on_save_txt(self):
        """
        Сохранить ТЕКСТ, который виден в output_text, в .txt файл (UTF-8).
        Это то, что вы просили: сохраняется именно содержимое Text (а не CSV).
        """
        content = self.output_text.get("1.0", "end").rstrip()
        if not content:
            messagebox.showinfo("Нет данных", "В поле вывода нет текста для сохранения.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".txt",
                                            filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            messagebox.showinfo("Сохранено", f"Текст сохранён в {path}")
        except Exception as e:
            messagebox.showerror("Ошибка сохранения", str(e))

if __name__ == "__main__":
    app = VoronoiApp()
    app.mainloop()

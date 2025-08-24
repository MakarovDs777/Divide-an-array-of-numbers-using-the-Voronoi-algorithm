import random
from typing import List, Tuple
import numpy as np

def partition_voronoi(values: List[float],
                      n_seeds: int,
                      lloyd_iters: int = 0,
                      init_method: str = "random") -> Tuple[List[List[float]], np.ndarray]:
    """
    Разбивает список чисел values на n_seeds наборов методом 1D-Voronoi с опциональными итерациями Ллойда.
    Параметры:
      values      - входной список чисел (float или int)
      n_seeds     - количество сайтов (кластеров)
      lloyd_iters - количество итераций Ллойда (>=0)
      init_method - способ инициализации сайтов: "random" или "quantile"
    Возвращает:
      clusters - список длины n_seeds; каждый элемент — список чисел, принадлежащих соответствующему кластеру
      seeds    - массив позиций сайтов (float), длины n_seeds
    """
    if n_seeds <= 0:
        raise ValueError("n_seeds must be >= 1")
    if len(values) == 0:
        return [[] for _ in range(n_seeds)], np.array([])

    arr = np.array(values, dtype=float)

    # Инициализация сайтов
    if init_method == "quantile":
        # равномерные квантили по отсортированному массиву
        sorted_vals = np.sort(arr)
        probs = np.linspace(0, 1, n_seeds + 2)[1:-1]  # внутренние квантили
        seeds = np.array([np.quantile(sorted_vals, p) for p in probs], dtype=float)
    else:
        # случайные позиции в диапазоне данных
        lo, hi = arr.min(), arr.max()
        seeds = np.array([random.uniform(lo, hi) for _ in range(n_seeds)], dtype=float)

    # Lloyd iterations (пересчет сайтов как среднего своих регионов)
    for _ in range(int(lloyd_iters)):
        # назначаем каждый элемент к ближайшему сайту
        # расстояния: |arr[:,None] - seeds[None,:]|
        idx = np.abs(arr[:, None] - seeds[None, :]).argmin(axis=1)
        # пересчитываем сайты
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

    # Финальное разбиение: формируем списки (сохранено исходное значение, не индекс)
    idx = np.abs(arr[:, None] - seeds[None, :]).argmin(axis=1)
    clusters: List[List[float]] = [[] for _ in range(n_seeds)]
    for val, k in zip(values, idx):  # используем оригинальные значения в исходном порядке
        clusters[int(k)].append(val)

    return clusters, seeds

# Пример использования
if __name__ == "__main__":
    values = [0,1,2,3,4,5,6,7,8,9]  # ваш массив
    n_seeds = 3
    clusters, seeds = partition_voronoi(values, n_seeds, lloyd_iters=10, init_method="quantile")

    print("Seeds:", seeds)
    for i, c in enumerate(clusters):
        print(f"Cluster {i} ({len(c)}):", c)

# КТ4 · Нейросетевой советник для КТ3

`damping_advisor.DampingAdvisor` реализует контракт `reset()` и `propose(features)`. Модуль передаёт в решатель положительное демпфирование. Готовые веса MLP и локальной RNN лежат в `results/`. Они обучены для этой реализации на маршрутах 310–329. Для каждой сети испытано 9 сочетаний ширины и шага обучения. Выбранные по валидации веса и эпохи указаны в [таблице подбора](results/selection.csv).

[Отчёт](Отчёт.md) · [Самодостаточный ноутбук с двумя панелями](Советник.ipynb) · [Все 200 прогонов](results/runs.csv)

## Запуск

В Colab выполните все ячейки ноутбука на CPU. Первая панель позволяет обучить обе архитектуры с выбранной шириной, шагом и ограничением памяти. Во второй можно сменить маршрут, политику, Huber и вычислительный бюджет.

Для локального запуска из КТ4 в том же окружении, что и КТ3, выполните команды ниже. Папки КТ3 и КТ4 расположены рядом. Можно также установить библиотеку КТ3 в окружение как обычный пакет.

```bash
python -m pip install -r ../КТ3/requirements.txt
python -m pip install ../КТ3
python -m unittest -v test_advisor
python study_neuro.py
```

## Использование готовых весов

```python
from damping_advisor import DampingAdvisor
from slam_lab import solve
from slam_lab.bench import episode

graph, reference = episode(1810, "dropout")
policy = DampingAdvisor.load("results/rnn.npz")
estimate = solve(graph, policy=policy, initial_lambda=0.0003, budget=24)
print(estimate.solves, estimate.status, estimate.history[-1])
```

Порядок шести входов, средние и масштабы нормировки, ограничение памяти и параметры сети записаны в NPZ. Загрузчик проверяет размерности и имена признаков. В `results/protocol.json` сохранены разбиение, сетка учителя и параметры обучения. Локальные проверки проводились на CPU в Python 3.12.

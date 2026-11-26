# КТ4 · Нейросетевой советник для КТ3

[Pull request КТ3–КТ4](https://github.com/palkakopalka52-ui/fusion-slam-coursework/pull/1) · [История решений](CHANGELOG.md)

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
python stress_test.py
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

Дополнительный опыт можно выполнить сразу с готовыми весами. [План](experiment_plan.md), [72 запуска](results/stress_runs.csv) и [медианы и диапазоны](results/stress_summary.csv) позволяют проверить прогноз. В таблице `advisor_calls` считает обращения к сети, `solves` — попытки линейного решения, `loss_evaluations` — все фактические вычисления скалярной потери при инициализации, линеаризации и проверке кандидата.

[Короткая панель для демонстрации](Панель.ipynb) запускается из папки КТ4 с готовыми весами. Откройте обе ячейки в Jupyter и выполните их либо запустите в этой папке `voila Панель.ipynb`, если установлен пакет `voila`. На снимке панели выбран маршрут 1817 и MLP. Виджет пересчитал ATE 0,1034 м за девять решений.

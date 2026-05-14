# PyTorch Lab - основы PyTorch

> Лабораторная работа по дисциплине «Нейронные сети».

Репозиторий содержит пять последовательных ноутбуков по основам PyTorch: от тензоров и автоматического дифференцирования до регрессии, ранней остановки и классификации. Каждый ноутбук можно открыть напрямую в Google Colab.

## Состав работы *

| № | Тема | Файл | Colab |
|---|------|------|-------|
| 01 | Основы PyTorch: тензоры, операции, autograd, GPU | [`01_pytorch_basics.ipynb`](./01_pytorch_basics.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/megusto0/pytorch-lab/blob/main/01_pytorch_basics.ipynb) |
| 02 | Введение в нейронные сети: один нейрон и функции активации | [`02_neural_net_intro.ipynb`](./02_neural_net_intro.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/megusto0/pytorch-lab/blob/main/02_neural_net_intro.ipynb) |
| 03 | Регрессия на PyTorch: датасет Auto MPG | [`03_pytorch_regression.ipynb`](./03_pytorch_regression.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/megusto0/pytorch-lab/blob/main/03_pytorch_regression.ipynb) |
| 04 | Ранняя остановка и сохранение лучшей модели | [`04_early_stopping.ipynb`](./04_early_stopping.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/megusto0/pytorch-lab/blob/main/04_early_stopping.ipynb) |
| 05 | Классификация: `nn.Module` и `nn.Sequential` | [`05_pytorch_classification.ipynb`](./05_pytorch_classification.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/megusto0/pytorch-lab/blob/main/05_pytorch_classification.ipynb) |

## Описание ноутбуков

**01 - Основы PyTorch.** Ноутбук знакомит с тензорами, базовыми операциями, изменением формы, индексацией, связью PyTorch с NumPy и автоматическим дифференцированием. В конце проверяется доступность CUDA и сравнивается время матричного умножения на CPU и GPU, если GPU доступен.

**02 - Введение в нейронные сети.** На синтетическом двумерном датасете строится один нейрон `nn.Linear(2, 1)` с sigmoid-активацией. Показаны начальная и обученная границы решения, ручной цикл обучения и график основных функций активации.

**03 - Регрессия на PyTorch.** Выполняется полный цикл обучения регрессионной модели на датасете Auto MPG: загрузка данных из UCI, очистка, кодирование категориального признака, стандартизация, `DataLoader`, обучение MLP и оценка RMSE. В конце сравниваются несколько топологий MLP на одном разбиении данных.

**04 - Ранняя остановка.** Продолжается задача регрессии Auto MPG, но обучение дополняется классом `EarlyStopping`. Лучшие веса сохраняются в `best_model.pt`, затем сравниваются варианты `patience = 3`, `10` и `25`.

**05 - Классификация.** На датасете Iris сравниваются два способа объявления одной и той же архитектуры: через собственный класс `nn.Module` и через `nn.Sequential`. Для оценки используется accuracy и матрица ошибок.

## Запуск

В Google Colab откройте нужный ноутбук кнопкой **Open in Colab** из таблицы выше. Все ноутбуки самостоятельны: данные загружаются из интернета или создаются внутри ноутбука.

Для локального запуска:

```bash
pip install -r requirements.txt
jupyter lab
```

## Литература

- [Jeff Heaton, T81-558: Applications of Deep Neural Networks](https://github.com/jeffheaton/app_deep_learning)
- [Jeff Heaton, PyTorch numerical notebook](https://github.com/jeffheaton/app_deep_learning/blob/main/t81_558_class_02_1_pytorch_numerical.ipynb)
- [Jeff Heaton, beyond CPU notebook](https://github.com/jeffheaton/app_deep_learning/blob/main/t81_558_class_02_5_beyond_cpu.ipynb)
- [Jeff Heaton, neural network intro notebook](https://github.com/jeffheaton/app_deep_learning/blob/main/t81_558_class_01_3_neural_net.ipynb)
- [Jeff Heaton, PyTorch regression notebook](https://github.com/jeffheaton/app_deep_learning/blob/main/t81_558_class_02_2_pytorch_neural.ipynb)
- [Jeff Heaton, early stopping notebook](https://github.com/jeffheaton/app_deep_learning/blob/main/t81_558_class_04_2_early_stop.ipynb)
- [Jeff Heaton, PyTorch classification sequence notebook](https://github.com/jeffheaton/app_deep_learning/blob/main/t81_558_class_02_4_pytorch_class_sequence.ipynb)
- [Maxim Lapan, Deep Reinforcement Learning Hands-On, Chapter 3](https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On/tree/master/Chapter03)
- [Официальная документация PyTorch](https://pytorch.org/docs/stable/index.html)

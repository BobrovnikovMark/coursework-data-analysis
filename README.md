# Курсовой проект: Первичный анализ наборов данных

Поиск и первичный анализ наборов данных разных типов по различным областям применения.

## Структура репозитория

```
├── chapter1_tabular/        # Глава 1 — Табличные данные
│   ├── plots_code.py        # Графики и визуализация
│   ├── create_dataset.py    # Генерация датасета
│   └── glava1_original.py   # Оригинальный код (Colab)
│
├── chapter2_timeseries/     # Глава 2 — Временные ряды
│   └── full_analysis.py     # Полный анализ MSFT
│
├── chapter3_images/         # Глава 3 — Изображения
│   └── README.md            # Описание (паспорт датасета)
│
└── chapter4_text/           # Глава 4 — Текстовые данные
    ├── nlp_analysis.py      # NLP-анализ
    └── text_processing_original.py  # Оригинальный код (Colab)
```

## Датасеты

| Глава | Датасет | Источник |
|-------|---------|----------|
| 1 | Bachelor's Degree Majors by Age, Sex, and State | [Kaggle](https://www.kaggle.com/datasets/tjkyner/bachelor-degree-majors-by-age-sex-and-state/data) |
| 2 | Microsoft Stock (MSFT) | [Yahoo Finance](https://finance.yahoo.com/quote/MSFT/history/) |
| 3 | Horses or Humans Dataset | [Kaggle](https://www.kaggle.com/datasets/sanikamal/horses-or-humans-dataset) |
| 4 | ru_sentiment_dataset | [HuggingFace](https://huggingface.co/datasets/MonoHime/ru_sentiment_dataset) |

## Технологии

- Python 3
- pandas, numpy
- matplotlib, seaborn
- scikit-learn
- NLTK, pymorphy3

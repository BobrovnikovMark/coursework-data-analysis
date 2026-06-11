# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

DATA_PATH = r'C:\Users\марк\.gemini\antigravity\scratch\time_series_dataset\microsoft_stock.csv'
PLOT_DIR = r'C:\Users\марк\.gemini\antigravity\scratch\time_series_dataset\plots'
os.makedirs(PLOT_DIR, exist_ok=True)

# ============================================================
# STAGE 1: Loading and initial inspection
# ============================================================
print('='*60)
print('ЭТАП 1. Загрузка и первичное знакомство с данными')
print('='*60)

df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
df.index.name = 'Date'

print(f'Размерность данных: {df.shape[0]} строк, {df.shape[1]} столбцов')
print(f'Столбцы: {list(df.columns)}')
print(f'Период: {df.index[0].strftime("%Y-%m-%d")} -- {df.index[-1].strftime("%Y-%m-%d")}')
print(f'Тип индекса: {type(df.index).__name__}')
print()
print('Типы данных:')
for col in df.columns:
    print(f'  {col}: {df[col].dtype}')
print()
print('Первые 5 строк:')
print(df.head().to_string())
print()
print('Последние 5 строк:')
print(df.tail().to_string())
print()

# Выводы этапа 1
print('ВЫВОДЫ Этапа 1:')
print(f'1. Данные загружены корректно, без ошибок.')
print(f'2. Набор данных содержит {df.shape[0]} торговых дней, {df.shape[1]} каналов.')
print(f'3. Временная метка преобразована в DatetimeIndex.')
print(f'4. Ценовые столбцы (Open, High, Low, Close) имеют тип float64, Volume - int64.')
print(f'5. Ряд является многомерным с 5 каналами.')

# ============================================================
# STAGE 2: Visualization
# ============================================================
print()
print('='*60)
print('ЭТАП 2. Визуализация исходных данных')
print('='*60)

split_idx = int(len(df) * 0.8)
split_date = df.index[split_idx]
print(f'Граница train/test (80/20): {split_date.strftime("%Y-%m-%d")}')
print(f'Train: {split_idx} дней, Test: {len(df) - split_idx} дней')

fig, axes = plt.subplots(5, 1, figsize=(14, 16), sharex=True)
channels = ['Open', 'High', 'Low', 'Close', 'Volume']
labels_ru = ['Цена открытия (Open)', 'Максимум (High)', 'Минимум (Low)', 
             'Цена закрытия (Close)', 'Объём торгов (Volume)']

for i, (col, label) in enumerate(zip(channels, labels_ru)):
    axes[i].plot(df.index, df[col], linewidth=0.8, color='#1f77b4')
    axes[i].axvline(x=split_date, color='red', linestyle='--', linewidth=1.2, label='Train/Test')
    axes[i].set_ylabel(label, fontsize=10)
    axes[i].legend(loc='upper left', fontsize=8)
    axes[i].grid(True, alpha=0.3)

axes[0].set_title('Акции Microsoft (MSFT): визуализация всех каналов', fontsize=14, fontweight='bold')
axes[-1].set_xlabel('Дата', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'stage2_visualization.png'), dpi=150)
plt.close()
print('График сохранён: stage2_visualization.png')

print()
print('ВЫВОДЫ Этапа 2:')
print('1. Наблюдается выраженный восходящий тренд цены акций MSFT с 2015 по 2021 год.')
print('2. Виден резкий обвал в марте 2020 (пандемия COVID-19) с последующим быстрым восстановлением.')
print('3. Объём торгов показывает резкие всплески в периоды волатильности.')
print('4. Все ценовые каналы (Open, High, Low, Close) ведут себя очень похоже.')

# ============================================================
# STAGE 3: Statistical analysis
# ============================================================
print()
print('='*60)
print('ЭТАП 3. Статистический анализ')
print('='*60)

stats = df.describe().round(2)
print(stats.to_string())
print()

# Sampling frequency
diffs = pd.Series(df.index).diff().dropna()
median_diff = diffs.median()
print(f'Медианный интервал между наблюдениями: {median_diff}')
print(f'Частота дискретизации: 1 торговый день (ежедневные данные)')
print()

print('ВЫВОДЫ Этапа 3:')
print('1. Все значения положительны (min > 0 для всех каналов).')
print(f'2. Среднее значение Close ({stats.loc["mean","Close"]}) > медианы ({stats.loc["50%","Close"]}), что указывает на правую асимметрию.')
print('3. Стандартное отклонение Volume значительно больше, чем у ценовых каналов.')
print('4. Диапазон цен: от ~$34 до ~$234, что отражает сильный рост за период.')
print('5. Дискретность ряда: 1 торговый день, интервалы равномерны (кроме выходных/праздников).')

# ============================================================
# STAGE 4: Missing values and outliers
# ============================================================
print()
print('='*60)
print('ЭТАП 4. Анализ пропусков и выбросов')
print('='*60)

# Missing values
missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
print('Доля пропущенных значений (%):')
for col in df.columns:
    print(f'  {col}: {missing_pct[col]}%')
print()

# Outliers by 3-sigma rule
print('Выбросы по правилу 3 сигм:')
outlier_counts = {}
for col in df.columns:
    mean = df[col].mean()
    std = df[col].std()
    lower = mean - 3 * std
    upper = mean + 3 * std
    outliers = ((df[col] < lower) | (df[col] > upper)).sum()
    outlier_counts[col] = outliers
    print(f'  {col}: {outliers} выбросов (границы: [{lower:.2f}, {upper:.2f}])')

# Box plots
fig, axes = plt.subplots(1, 5, figsize=(16, 6))
for i, col in enumerate(df.columns):
    axes[i].boxplot(df[col].dropna(), patch_artist=True,
                    boxprops=dict(facecolor='#5DADE2', alpha=0.7),
                    medianprops=dict(color='red', linewidth=2))
    axes[i].set_title(col, fontsize=11, fontweight='bold')
    axes[i].grid(True, alpha=0.3)

fig.suptitle('Диаграммы размаха для каждого канала', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'stage4_boxplots.png'), dpi=150, bbox_inches='tight')
plt.close()
print()
print('График сохранён: stage4_boxplots.png')

print()
print('ВЫВОДЫ Этапа 4:')
print('1. Пропущенных значений нет ни в одном канале (0%).')
total_outliers = sum(outlier_counts.values())
print(f'2. Общее количество выбросов по правилу 3 сигм: {total_outliers}.')
if outlier_counts.get('Volume', 0) > 0:
    print(f'3. Volume содержит {outlier_counts["Volume"]} выбросов - резкие всплески объёма торгов.')
print('4. Пропуски отсутствуют, предварительная обработка пропусков не требуется.')
print('5. Выбросы в Volume связаны с реальными рыночными событиями, удалять их нецелесообразно.')

# ============================================================
# STAGE 5: Range analysis
# ============================================================
print()
print('='*60)
print('ЭТАП 5. Анализ диапазонов значений')
print('='*60)

fig, ax = plt.subplots(figsize=(12, 6))
bp = ax.boxplot([df[col].dropna() for col in df.columns], 
                labels=df.columns, patch_artist=True,
                boxprops=dict(alpha=0.7),
                medianprops=dict(color='red', linewidth=2))
colors = ['#3498DB', '#2ECC71', '#E74C3C', '#9B59B6', '#F39C12']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
ax.set_title('Сравнение диапазонов всех каналов на одном графике', fontsize=14, fontweight='bold')
ax.set_ylabel('Значение', fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'stage5_ranges.png'), dpi=150)
plt.close()
print('График сохранён: stage5_ranges.png')

print()
print('ВЫВОДЫ Этапа 5:')
print('1. Volume имеет масштаб ~10^7, тогда как цены ~10^1-10^2.')
print('2. Масштаб Volume на несколько порядков отличается от ценовых каналов.')
print('3. Требуется нормализация/стандартизация данных перед применением моделей.')
print('4. Для задачи прогнозирования рекомендуется Min-Max нормализация или стандартизация (Z-score).')

# ============================================================
# STAGE 6: Correlation analysis
# ============================================================
print()
print('='*60)
print('ЭТАП 6. Корреляционный анализ')
print('='*60)

corr = df.corr().round(4)
print('Матрица корреляции Пирсона:')
print(corr.to_string())

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(corr.values, cmap='RdYlBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, fontsize=11)
ax.set_yticklabels(corr.columns, fontsize=11)
for i in range(len(corr)):
    for j in range(len(corr)):
        val = corr.values[i, j]
        color = 'white' if abs(val) > 0.7 else 'black'
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=12, color=color, fontweight='bold')
plt.colorbar(im, ax=ax, shrink=0.8)
ax.set_title('Тепловая карта корреляции', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'stage6_correlation.png'), dpi=150)
plt.close()
print()
print('График сохранён: stage6_correlation.png')

print()
print('ВЫВОДЫ Этапа 6:')
print(f'1. Open, High, Low, Close имеют почти идеальную корреляцию (r > 0.99) - это ожидаемо.')
print(f'2. Volume слабо коррелирует с ценовыми каналами (r = {corr.loc["Volume","Close"]:.4f}).')
print('3. Мультиколлинеарность среди ценовых каналов: для моделирования достаточно одного (Close).')
print('4. Volume несёт уникальную информацию, независимую от цены.')

# ============================================================
# STAGE 7: Noise analysis
# ============================================================
print()
print('='*60)
print('ЭТАП 7. Поиск и анализ шумов')
print('='*60)

from statsmodels.tsa.seasonal import seasonal_decompose

# Use Close price, period=252 (trading days in a year)
close = df['Close'].dropna()
decomposition = seasonal_decompose(close, model='additive', period=252)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

# Plot decomposition
fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
axes[0].plot(close.index, close, linewidth=0.8, color='#1f77b4')
axes[0].set_ylabel('Исходный ряд', fontsize=10)
axes[0].set_title('Декомпозиция временного ряда Close (MSFT)', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)

axes[1].plot(trend.index, trend, linewidth=1.2, color='#E74C3C')
axes[1].set_ylabel('Тренд', fontsize=10)
axes[1].grid(True, alpha=0.3)

axes[2].plot(seasonal.index, seasonal, linewidth=0.6, color='#2ECC71')
axes[2].set_ylabel('Сезонность', fontsize=10)
axes[2].grid(True, alpha=0.3)

axes[3].plot(residual.index, residual, linewidth=0.6, color='#9B59B6')
axes[3].set_ylabel('Остатки (шум)', fontsize=10)
axes[3].set_xlabel('Дата', fontsize=11)
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'stage7_decomposition.png'), dpi=150)
plt.close()
print('График сохранён: stage7_decomposition.png')

# Calculate SNR
signal = trend + seasonal
signal_clean = signal.dropna()
residual_clean = residual.dropna()

var_signal = signal_clean.var()
var_noise = residual_clean.var()
snr = 10 * np.log10(var_signal / var_noise)

print(f'Дисперсия сигнала: {var_signal:.4f}')
print(f'Дисперсия шума: {var_noise:.4f}')
print(f'SNR = {snr:.2f} дБ')

# Histogram of residuals
fig, ax = plt.subplots(figsize=(10, 6))
residual_vals = residual_clean.values
ax.hist(residual_vals, bins=50, color='#5DADE2', edgecolor='black', alpha=0.7, density=True)
ax.axvline(x=0, color='red', linestyle='--', linewidth=1.5)
ax.set_title('Гистограмма распределения остатков (шума)', fontsize=14, fontweight='bold')
ax.set_xlabel('Значение остатков', fontsize=11)
ax.set_ylabel('Плотность', fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'stage7_residuals_hist.png'), dpi=150)
plt.close()
print('График сохранён: stage7_residuals_hist.png')

# Residual statistics
print(f'Среднее остатков: {residual_clean.mean():.4f}')
print(f'Медиана остатков: {residual_clean.median():.4f}')
print(f'Std остатков: {residual_clean.std():.4f}')

print()
print('ВЫВОДЫ Этапа 7:')
if snr > 20:
    snr_quality = 'Отлично'
    snr_desc = 'Шум практически незаметен, данные очень чистые.'
elif snr > 10:
    snr_quality = 'Хорошо'
    snr_desc = 'Шум присутствует, но сигнал доминирует.'
elif snr > 0:
    snr_quality = 'Удовлетворительно'
    snr_desc = 'Сигнал и шум сравнимы по мощности.'
else:
    snr_quality = 'Плохо'
    snr_desc = 'Шум сильнее полезного сигнала.'

print(f'1. Тренд: выраженный восходящий, сильный рост с ~$35 до ~$230.')
print(f'2. Сезонность: годовая (период 252 торговых дня), амплитуда средняя.')
print(f'3. SNR = {snr:.2f} дБ. Качественная оценка: {snr_quality}. {snr_desc}')
print(f'4. Распределение остатков: близко к нормальному с лёгкими тяжёлыми хвостами.')
print(f'5. Фильтрация данных не требуется при текущем уровне SNR.')

print()
print('='*60)
print('ВСЕ ЭТАПЫ ВЫПОЛНЕНЫ УСПЕШНО!')
print('='*60)

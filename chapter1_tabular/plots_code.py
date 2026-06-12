# -*- coding: utf-8 -*-
"""
Код для построения графиков к Главе 1 (НА РУССКОМ)
Датасет: Bachelor's Degree Majors by Age, Sex, and State
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

DATA_PATH = r'C:\Users\марк\.gemini\antigravity\scratch\bachelor_degree\bachelor_degrees.csv'
PLOT_DIR = r'C:\Users\марк\.gemini\antigravity\scratch\bachelor_degree\plots'
os.makedirs(PLOT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

# Русские названия столбцов для графиков
col_ru = {
    'Count': 'Кол-во обладателей степени',
    'Total_Population_25plus': 'Население 25+',
    'Percent_With_Bachelors': 'Доля с бакалавром (%)',
    'Median_Income': 'Медианный доход ($)',
    'Employment_Rate': 'Уровень занятости (%)',
    'Graduate_Degree_Holders': 'Имеют магистратуру/PhD',
}

# Русские названия направлений
major_ru = {
    'Science and Engineering': 'Наука и инженерия',
    'Science and Engineering Related': 'Смежные с наукой',
    'Business': 'Бизнес',
    'Education': 'Образование',
    'Arts, Humanities and Others': 'Искусство и гуманитарные',
    'Computers, Mathematics, Statistics': 'Информатика и математика',
    'Biological, Agricultural, Environmental': 'Биология и экология',
    'Physical and Related Sciences': 'Физические науки',
    'Psychology': 'Психология',
    'Social Sciences': 'Социальные науки',
    'Engineering': 'Инженерия',
    'Literature and Languages': 'Литература и языки',
}

df['Major_RU'] = df['Major_Field'].map(major_ru)

num_cols = ['Count', 'Total_Population_25plus', 'Percent_With_Bachelors',
            'Median_Income', 'Employment_Rate', 'Graduate_Degree_Holders']

# ============================================================
# 1. Гистограммы распределения
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()
for i, col in enumerate(num_cols):
    axes[i].hist(df[col], bins=30, color='#3498DB', edgecolor='black', alpha=0.7)
    axes[i].set_title(col_ru[col], fontsize=12, fontweight='bold')
    axes[i].set_xlabel('Значение')
    axes[i].set_ylabel('Частота')
    axes[i].grid(True, alpha=0.3)
fig.suptitle('Гистограммы распределения числовых признаков', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'fig1_histograms.png'), dpi=150)
plt.close()
print('Saved: fig1_histograms.png')

# ============================================================
# 2a. Сравнение мужчин и женщин по направлениям подготовки
# ============================================================
sex_ru = {'Male': 'Мужчины', 'Female': 'Женщины'}
df['Пол'] = df['Sex'].map(sex_ru)

age_ru = {'25 to 34': '25–34', '35 to 44': '35–44', '45 to 64': '45–64', '65 and over': '65+'}
df['Возраст'] = df['Age_Group'].map(age_ru)

sex_major = df.groupby(['Major_RU', 'Пол'])['Count'].sum().unstack(fill_value=0)
sex_major = sex_major.sort_values('Мужчины', ascending=True)

fig, ax = plt.subplots(figsize=(12, 7))
y_pos = np.arange(len(sex_major))
h = 0.35
ax.barh(y_pos - h/2, sex_major['Мужчины'], h, label='Мужчины', color='#3498DB', alpha=0.85)
ax.barh(y_pos + h/2, sex_major['Женщины'], h, label='Женщины', color='#E74C3C', alpha=0.85)
ax.set_yticks(y_pos)
ax.set_yticklabels(sex_major.index, fontsize=10)
ax.set_xlabel('Общее количество обладателей степени', fontsize=11)
ax.set_title('Сравнение мужчин и женщин по направлениям подготовки', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'fig2a_sex_by_major.png'), dpi=150)
plt.close()
print('Saved: fig2a_sex_by_major.png')

# ============================================================
# 2b. Распределение возрастных групп по направлениям (стек)
# ============================================================
age_major = df.groupby(['Major_RU', 'Возраст'])['Count'].sum().unstack(fill_value=0)
age_order = ['25–34', '35–44', '45–64', '65+']
age_major = age_major[age_order]
age_major = age_major.loc[age_major.sum(axis=1).sort_values(ascending=False).index]

fig, ax = plt.subplots(figsize=(14, 7))
colors_age = ['#2ECC71', '#F39C12', '#9B59B6', '#1ABC9C']
age_major.plot(kind='bar', stacked=True, ax=ax, color=colors_age, edgecolor='white', linewidth=0.5)
ax.set_title('Распределение обладателей степени по возрасту и направлению', fontsize=13, fontweight='bold')
ax.set_xlabel('Направление подготовки', fontsize=11)
ax.set_ylabel('Общее количество', fontsize=11)
ax.legend(title='Возраст', fontsize=10, title_fontsize=11)
plt.xticks(rotation=40, ha='right', fontsize=9)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'fig2b_age_by_major.png'), dpi=150)
plt.close()
print('Saved: fig2b_age_by_major.png')

# ============================================================
# 2c. Boxplot по направлениям
# ============================================================
fig, ax = plt.subplots(figsize=(14, 6))
order = df.groupby('Major_RU')['Count'].median().sort_values(ascending=False).index
sns.boxplot(data=df, x='Major_RU', y='Count', order=order, palette='Set2', ax=ax)
ax.set_title('Распределение кол-ва обладателей степени по направлениям', fontsize=13, fontweight='bold')
ax.set_xlabel('Направление подготовки')
ax.set_ylabel('Кол-во обладателей степени')
plt.xticks(rotation=45, ha='right', fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'fig2c_boxplot_major.png'), dpi=150)
plt.close()
print('Saved: fig2c_boxplot_major.png')

# ============================================================
# 3a. Топ-10 штатов
# ============================================================
state_totals = df.groupby('State')['Count'].sum().sort_values(ascending=False).head(10)
fig, ax = plt.subplots(figsize=(12, 6))
colors = plt.cm.viridis(np.linspace(0.2, 0.8, 10))
ax.barh(state_totals.index[::-1], state_totals.values[::-1], color=colors)
ax.set_title('Топ-10 штатов по количеству обладателей степени бакалавра', fontsize=13, fontweight='bold')
ax.set_xlabel('Общее количество')
for i, v in enumerate(state_totals.values[::-1]):
    ax.text(v + 500, i, '{:,}'.format(v), va='center', fontsize=9)
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'fig3a_top10_states.png'), dpi=150)
plt.close()
print('Saved: fig3a_top10_states.png')

# ============================================================
# 3b. Круговая — направления
# ============================================================
major_totals = df.groupby('Major_RU')['Count'].sum().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(10, 8))
colors_pie = plt.cm.Set3(np.linspace(0, 1, len(major_totals)))
wedges, texts, autotexts = ax.pie(major_totals, labels=None, autopct='%1.1f%%',
                                   colors=colors_pie, pctdistance=0.85, startangle=90)
ax.legend(major_totals.index, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9)
ax.set_title('Распределение по направлениям подготовки', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'fig3b_pie_majors.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Saved: fig3b_pie_majors.png')

# ============================================================
# 4a. Тепловая карта корреляции
# ============================================================
corr = df[num_cols].corr().round(3)
corr_ru = corr.copy()
corr_ru.index = [col_ru[c] for c in corr.index]
corr_ru.columns = [col_ru[c] for c in corr.columns]

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(corr_ru.values, cmap='RdYlBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(len(corr_ru.columns)))
ax.set_yticks(range(len(corr_ru.columns)))
ax.set_xticklabels(corr_ru.columns, fontsize=9, rotation=45, ha='right')
ax.set_yticklabels(corr_ru.columns, fontsize=9)
for i in range(len(corr_ru)):
    for j in range(len(corr_ru)):
        val = corr_ru.values[i, j]
        c = 'white' if abs(val) > 0.6 else 'black'
        ax.text(j, i, '{:.2f}'.format(val), ha='center', va='center', fontsize=10, color=c, fontweight='bold')
plt.colorbar(im, ax=ax, shrink=0.8)
ax.set_title('Тепловая карта корреляции числовых признаков', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'fig4a_corr_heatmap.png'), dpi=150)
plt.close()
print('Saved: fig4a_corr_heatmap.png')

# ============================================================
# 4b. Тепловая карта: Штат x Направление
# ============================================================
top10_states = df.groupby('State')['Count'].sum().sort_values(ascending=False).head(10).index
pivot = df[df['State'].isin(top10_states)].pivot_table(values='Count', index='State', columns='Major_RU', aggfunc='sum')
fig, ax = plt.subplots(figsize=(14, 7))
sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax, linewidths=0.5)
ax.set_title('Обладатели степени: Топ-10 штатов × направления', fontsize=13, fontweight='bold')
ax.set_xlabel('Направление подготовки')
ax.set_ylabel('Штат')
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'fig4b_state_major_heatmap.png'), dpi=150)
plt.close()
print('Saved: fig4b_state_major_heatmap.png')

# ============================================================
# 5. Нормализованные boxplots
# ============================================================
fig, ax = plt.subplots(figsize=(12, 6))
df_norm = df[num_cols].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
labels_ru = [col_ru[c] for c in num_cols]
bp = ax.boxplot([df_norm[col].dropna() for col in num_cols], labels=labels_ru, patch_artist=True,
                medianprops=dict(color='red', linewidth=2))
colors_box = ['#3498DB', '#2ECC71', '#E74C3C', '#9B59B6', '#F39C12', '#1ABC9C']
for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_title('Нормализованные диаграммы размаха числовых признаков', fontsize=13, fontweight='bold')
ax.set_ylabel('Нормализованное значение (0–1)')
plt.xticks(rotation=30, ha='right', fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'fig5_boxplots_normalized.png'), dpi=150)
plt.close()
print('Saved: fig5_boxplots_normalized.png')

# ============================================================
# 6. Категориальные: Пол и Возраст
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sex_counts = df.groupby('Пол')['Count'].sum()
axes[0].bar(sex_counts.index, sex_counts.values, color=['#3498DB', '#E74C3C'])
axes[0].set_title('Общее количество по полу', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Общее количество')
axes[0].grid(True, alpha=0.3, axis='y')

age_counts = df.groupby('Возраст')['Count'].sum()
age_order = ['25–34', '35–44', '45–64', '65+']
age_counts = age_counts.reindex(age_order)
axes[1].bar(age_counts.index, age_counts.values, color=['#2ECC71', '#F39C12', '#9B59B6', '#1ABC9C'])
axes[1].set_title('Общее количество по возрастной группе', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Общее количество')
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'fig6_categorical.png'), dpi=150)
plt.close()
print('Saved: fig6_categorical.png')

# ============================================================
# 7. Шум
# ============================================================
df_noisy = df.copy()
np.random.seed(42)
noise_income = np.random.normal(0, df['Median_Income'].std() * 0.05, len(df))
df_noisy['Median_Income'] = (df['Median_Income'] + noise_income).astype(int)
noise_emp = np.random.normal(0, 1.5, len(df))
df_noisy['Employment_Rate'] = np.clip(df['Employment_Rate'] + noise_emp, 0, 100).round(1)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(df['Median_Income'], bins=30, alpha=0.5, color='blue', label='Оригинал')
axes[0].hist(df_noisy['Median_Income'], bins=30, alpha=0.5, color='red', label='С шумом')
axes[0].set_title('Медианный доход: до и после добавления шума', fontsize=11, fontweight='bold')
axes[0].set_xlabel('Медианный доход ($)')
axes[0].set_ylabel('Частота')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].hist(df['Employment_Rate'], bins=30, alpha=0.5, color='blue', label='Оригинал')
axes[1].hist(df_noisy['Employment_Rate'], bins=30, alpha=0.5, color='red', label='С шумом')
axes[1].set_title('Уровень занятости: до и после добавления шума', fontsize=11, fontweight='bold')
axes[1].set_xlabel('Уровень занятости (%)')
axes[1].set_ylabel('Частота')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'fig7_noise.png'), dpi=150)
plt.close()
print('Saved: fig7_noise.png')

print()
print('ВСЕ ГРАФИКИ СГЕНЕРИРОВАНЫ НА РУССКОМ ЯЗЫКЕ!')

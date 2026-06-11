# -*- coding: utf-8 -*-
"""
Глава 4: Первичный анализ набора текстовых данных
Датасет: MonoHime/ru_sentiment_dataset
ИСПРАВЛЕНО: подсчёт частоты ПОСЛЕ удаления стоп-слов, предлогов, междометий, цифр
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import os, re, warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = r'C:\Users\марк\.gemini\antigravity\scratch\nlp_dataset'
PLOT_DIR = os.path.join(OUTPUT_DIR, 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)

# ============================================================
# ЗАГРУЗКА ДАТАСЕТА
# ============================================================
print('='*60)
print('ЗАГРУЗКА ДАТАСЕТА: MonoHime/ru_sentiment_dataset')
print('='*60)

from datasets import load_dataset
ds = load_dataset('MonoHime/ru_sentiment_dataset', split='train')
df_full = ds.to_pandas()

label_map = {0: 'Негатив', 1: 'Позитив', 2: 'Нейтрально'}
df_full['category'] = df_full['sentiment'].map(label_map)

# Выборка: 30 отзывов (10 на класс)
df_samples = []
for sent_val in [0, 1, 2]:
    subset = df_full[df_full['sentiment'] == sent_val]
    subset = subset[subset['text'].str.len().between(50, 300)]
    sample = subset.sample(n=10, random_state=42)
    df_samples.append(sample)

df = pd.concat(df_samples).reset_index(drop=True)
df = df[['text', 'category']].copy()

df.to_csv(os.path.join(OUTPUT_DIR, 'ru_sentiment_sample.csv'), index=False, encoding='utf-8')

print('Полный датасет: {} отзывов'.format(len(df_full)))
print('Выборка для анализа: {} отзывов'.format(len(df)))
print('  Позитив: {}'.format(len(df[df['category']=='Позитив'])))
print('  Негатив: {}'.format(len(df[df['category']=='Негатив'])))
print('  Нейтрально: {}'.format(len(df[df['category']=='Нейтрально'])))

# ============================================================
# ПРИМЕР ДАТАСЕТА (5 строк)
# ============================================================
print()
print('='*60)
print('ПРИМЕР ДАТАСЕТА (первые 5 строк)')
print('='*60)
for i in range(5):
    print('{}. [{}] {}'.format(i+1, df['category'].iloc[i], df['text'].iloc[i][:150]))
print()

# ============================================================
# ШАГ 1. ОЧИСТКА ТЕКСТА
# ============================================================
print('='*60)
print('ШАГ 1. Очистка текста')
print('='*60)

def clean_text(text):
    text = text.lower()
    # Удаляем цифры, латиницу, знаки препинания — оставляем только русские буквы и пробелы
    text = re.sub(r'[^а-яё ]', '', text)
    text = ' '.join(text.split())
    return text

df['clean_text'] = df['text'].apply(clean_text)

for i in [0, 10, 20]:
    print('Пример {} [{}]:'.format(i+1, df['category'].iloc[i]))
    print('  До:    {}'.format(df['text'].iloc[i][:120]))
    print('  После: {}'.format(df['clean_text'].iloc[i][:120]))
    print()

# ============================================================
# ШАГ 2. ЛЕММАТИЗАЦИЯ
# ============================================================
print('='*60)
print('ШАГ 2. Лемматизация')
print('='*60)

import pymorphy3
morph = pymorphy3.MorphAnalyzer()

def lemmatize_text(text):
    words = text.split()
    lemmas = [morph.parse(word)[0].normal_form for word in words]
    return ' '.join(lemmas)

df['lemmas'] = df['clean_text'].apply(lemmatize_text)

for i in [0, 10, 20]:
    print('Пример {}:'.format(i+1))
    print('  Очищенный: {}'.format(df['clean_text'].iloc[i][:100]))
    print('  Леммы:     {}'.format(df['lemmas'].iloc[i][:100]))
    print()

# ============================================================
# ШАГ 3. УДАЛЕНИЕ СТОП-СЛОВ, ПРЕДЛОГОВ, МЕЖДОМЕТИЙ, ЦИФР
# ============================================================
print('='*60)
print('ШАГ 3. Удаление стоп-слов, предлогов, междометий')
print('='*60)

import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

stop_words = set(stopwords.words('russian'))

# Дополнительные предлоги, междометия, частицы, которых может не быть в NLTK
extra_stop = {
    # предлоги
    'из', 'от', 'до', 'по', 'за', 'на', 'над', 'под', 'при', 'про', 'через',
    'без', 'для', 'ради', 'между', 'среди', 'около', 'вокруг', 'после',
    'перед', 'кроме', 'вместо', 'насчёт', 'благодаря', 'вопреки',
    # междометия
    'ах', 'ох', 'эх', 'ух', 'ой', 'ай', 'увы', 'ура', 'браво', 'батюшки',
    'ого', 'угу', 'ага', 'хм', 'гм', 'фу', 'тьфу', 'ну', 'ба', 'эй',
    # частицы
    'бы', 'ли', 'же', 'вот', 'вон', 'даже', 'ведь', 'лишь', 'только',
    'уж', 'аж', 'ещё', 'уже', 'именно', 'пусть', 'пускай',
    # прочее
    'это', 'этот', 'тот', 'такой', 'такая', 'такое', 'какой', 'весь',
    'свой', 'мой', 'твой', 'наш', 'ваш', 'который', 'самый',
    'очень', 'просто', 'тоже', 'также',
}
stop_words = stop_words | extra_stop
print('Стоп-слов в словаре (включая доп.): {}'.format(len(stop_words)))

def remove_stopwords(text):
    words = text.split()
    # Удаляем стоп-слова + слова из 1 буквы + цифры (если остались)
    filtered = [w for w in words if w not in stop_words and len(w) > 1 and not w.isdigit()]
    return ' '.join(filtered)

df['no_stopwords'] = df['lemmas'].apply(remove_stopwords)

for i in [0, 10, 20]:
    print('Пример {}:'.format(i+1))
    print('  До:    {}'.format(df['lemmas'].iloc[i][:100]))
    print('  После: {}'.format(df['no_stopwords'].iloc[i][:100]))
    print()

# ============================================================
# ШАГ 4. ПОДСЧЁТ ЧАСТОТЫ СЛОВ (ПОСЛЕ УДАЛЕНИЯ СТОП-СЛОВ!)
# ============================================================
print('='*60)
print('ШАГ 4. Подсчёт частоты слов (после фильтрации)')
print('='*60)

from collections import Counter

all_words = ' '.join(df['no_stopwords']).split()
word_counts = Counter(all_words)

print('Всего слов (после фильтрации): {}'.format(len(all_words)))
print('Уникальных слов: {}'.format(len(word_counts)))
print()
print('Топ-10 самых частых слов (без стоп-слов, предлогов, междометий):')
for word, count in word_counts.most_common(10):
    print('  {}: {}'.format(word, count))

# Столбчатый график
top_words = word_counts.most_common(10)
words_list, counts_list = zip(*top_words)

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(words_list, counts_list, color='steelblue')
ax.set_title('Топ-10 самых частых слов\n(после удаления стоп-слов, предлогов, междометий)', fontsize=13, fontweight='bold')
ax.set_xlabel('Слово', fontsize=12)
ax.set_ylabel('Частота', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'step4_top10.png'), dpi=150)
plt.close()
print('Сохранён: step4_top10.png')

# Облако слов
from wordcloud import WordCloud

all_text = ' '.join(df['no_stopwords'])
wc = WordCloud(width=800, height=400, max_words=30, background_color='white').generate(all_text)
fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(wc, interpolation='bilinear')
ax.axis('off')
ax.set_title('Облако слов (после фильтрации)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'step4_wordcloud.png'), dpi=150)
plt.close()
print('Сохранён: step4_wordcloud.png')

# По категориям
pos_text = ' '.join(df[df['category'] == 'Позитив']['no_stopwords'])
neg_text = ' '.join(df[df['category'] == 'Негатив']['no_stopwords'])

wc_pos = WordCloud(width=400, height=300, background_color='white', max_words=20, colormap='Greens').generate(pos_text)
wc_neg = WordCloud(width=400, height=300, background_color='white', max_words=20, colormap='Reds').generate(neg_text)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].imshow(wc_pos, interpolation='bilinear')
axes[0].axis('off')
axes[0].set_title('Позитивные отзывы', fontsize=14, fontweight='bold')
axes[1].imshow(wc_neg, interpolation='bilinear')
axes[1].axis('off')
axes[1].set_title('Негативные отзывы', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'step4_wordcloud_cat.png'), dpi=150)
plt.close()
print('Сохранён: step4_wordcloud_cat.png')

# ============================================================
# ШАГ 5. TF-IDF
# ============================================================
print()
print('='*60)
print('ШАГ 5. TF-IDF')
print('='*60)

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['no_stopwords'])
feature_names = vectorizer.get_feature_names_out()

print('Размер словаря: {} слов'.format(len(feature_names)))
print('Размер TF-IDF матрицы: {}'.format(tfidf_matrix.shape))
print()
print('Пример для первого текста:')
print('  Текст: {}'.format(df['no_stopwords'].iloc[0][:100]))
vec = tfidf_matrix[0].toarray()[0]
nonzero = np.where(vec > 0)[0]
print('  Ненулевые TF-IDF:')
for idx in nonzero[:8]:
    print('    {} : {:.4f}'.format(feature_names[idx], vec[idx]))

# ============================================================
# ШАГ 6. ИНФОРМАЦИОННЫЙ ПОИСК
# ============================================================
print()
print('='*60)
print('ШАГ 6. Информационный поиск')
print('='*60)

from sklearn.metrics.pairwise import cosine_similarity

def search_texts(query, top_n=3):
    q = clean_text(query)
    q = lemmatize_text(q)
    q = remove_stopwords(q)
    q_vec = vectorizer.transform([q])
    sims = cosine_similarity(q_vec, tfidf_matrix)[0]
    top_idx = sims.argsort()[-top_n:][::-1]
    results = []
    for idx in top_idx:
        results.append({
            'original': df['text'].iloc[idx],
            'similarity': sims[idx],
            'category': df['category'].iloc[idx]
        })
    return results

queries = ['хороший товар качество', 'плохое обслуживание', 'доставка быстро']
for query in queries:
    results = search_texts(query)
    print('Запрос: "{}"'.format(query))
    for i, r in enumerate(results, 1):
        print('  {}. [{:.4f}] [{}] {}'.format(i, r['similarity'], r['category'], r['original'][:90]))
    print()

print('='*60)
print('ВСЕ 6 ШАГОВ ВЫПОЛНЕНЫ!')
print('='*60)

# -*- coding: utf-8 -*-
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

# Маппинг меток
label_map = {0: 'Негатив', 1: 'Позитив', 2: 'Нейтрально'}
df_full['category'] = df_full['sentiment'].map(label_map)

# Берём выборку: 30 отзывов (10 на класс), длина текста 50-300 символов
df_samples = []
for sent_val in [0, 1, 2]:
    subset = df_full[df_full['sentiment'] == sent_val]
    subset = subset[subset['text'].str.len().between(50, 300)]
    sample = subset.sample(n=10, random_state=42)
    df_samples.append(sample)

df = pd.concat(df_samples).reset_index(drop=True)
df = df[['text', 'category']].copy()

# Сохраняем CSV
df.to_csv(os.path.join(OUTPUT_DIR, 'ru_sentiment_sample.csv'), index=False, encoding='utf-8')

print('Полный датасет: {} отзывов'.format(len(df_full)))
print('Выборка для анализа: {} отзывов'.format(len(df)))
print('  Позитив: {}'.format(len(df[df['category']=='Позитив'])))
print('  Негатив: {}'.format(len(df[df['category']=='Негатив'])))
print('  Нейтрально: {}'.format(len(df[df['category']=='Нейтрально'])))
print()
print('Источник: HuggingFace - MonoHime/ru_sentiment_dataset')
print('Ссылка: https://huggingface.co/datasets/MonoHime/ru_sentiment_dataset')

# ============================================================
# ШАГ 1. ОЧИСТКА ТЕКСТА
# ============================================================
print()
print('='*60)
print('ШАГ 1. Очистка текста')
print('='*60)

def clean_text(text):
    text = text.lower()
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
# ШАГ 3. ПОДСЧЁТ ЧАСТОТЫ СЛОВ
# ============================================================
print('='*60)
print('ШАГ 3. Подсчёт частоты слов')
print('='*60)

from collections import Counter

all_words = ' '.join(df['lemmas']).split()
word_counts = Counter(all_words)

print('Всего слов: {}'.format(len(all_words)))
print('Уникальных слов: {}'.format(len(word_counts)))
print()
print('Топ-10 самых частых слов:')
for word, count in word_counts.most_common(10):
    print('  {}: {}'.format(word, count))

# Столбчатый график
top_words = word_counts.most_common(10)
words_list, counts_list = zip(*top_words)

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(words_list, counts_list, color='steelblue')
ax.set_title('Топ-10 самых частых слов', fontsize=14, fontweight='bold')
ax.set_xlabel('Слово', fontsize=12)
ax.set_ylabel('Частота', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'step3_top10.png'), dpi=150)
plt.close()
print('Сохранён: step3_top10.png')

# Облако слов
from wordcloud import WordCloud

all_text = ' '.join(df['lemmas'])
wc = WordCloud(width=800, height=400, max_words=30, background_color='white').generate(all_text)
fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(wc, interpolation='bilinear')
ax.axis('off')
ax.set_title('Облако слов (все отзывы)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'step3_wordcloud.png'), dpi=150)
plt.close()
print('Сохранён: step3_wordcloud.png')

# По категориям
pos_text = ' '.join(df[df['category'] == 'Позитив']['lemmas'])
neg_text = ' '.join(df[df['category'] == 'Негатив']['lemmas'])

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
plt.savefig(os.path.join(PLOT_DIR, 'step3_wordcloud_cat.png'), dpi=150)
plt.close()
print('Сохранён: step3_wordcloud_cat.png')

# ============================================================
# ШАГ 4. УДАЛЕНИЕ СТОП-СЛОВ
# ============================================================
print()
print('='*60)
print('ШАГ 4. Удаление стоп-слов')
print('='*60)

import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

stop_words = set(stopwords.words('russian'))
print('Стоп-слов в словаре: {}'.format(len(stop_words)))

def remove_stopwords(text):
    words = text.split()
    return ' '.join([w for w in words if w not in stop_words])

df['no_stopwords'] = df['lemmas'].apply(remove_stopwords)

for i in [0, 10, 20]:
    print('Пример {}:'.format(i+1))
    print('  До:    {}'.format(df['lemmas'].iloc[i][:100]))
    print('  После: {}'.format(df['no_stopwords'].iloc[i][:100]))
    print()

# Топ-10 после удаления
all_clean = ' '.join(df['no_stopwords']).split()
wc_clean = Counter(all_clean)
print('Топ-10 после удаления стоп-слов:')
for word, count in wc_clean.most_common(10):
    print('  {}: {}'.format(word, count))

top_c = wc_clean.most_common(10)
w_c, c_c = zip(*top_c)
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(w_c, c_c, color='#E74C3C')
ax.set_title('Топ-10 слов (без стоп-слов)', fontsize=14, fontweight='bold')
ax.set_xlabel('Слово', fontsize=12)
ax.set_ylabel('Частота', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'step4_top10_clean.png'), dpi=150)
plt.close()
print('Сохранён: step4_top10_clean.png')

# ============================================================
# ШАГ 5. TF-IDF
# ============================================================
print()
print('='*60)
print('ШАГ 5. TF-IDF')
print('='*60)

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words=list(stop_words))
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

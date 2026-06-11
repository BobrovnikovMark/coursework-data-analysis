# -*- coding: utf-8 -*-
import sys; sys.stdout.reconfigure(encoding='utf-8')
import os, csv
import pandas as pd
import numpy as np

np.random.seed(42)
output_dir = r'C:\Users\марк\.gemini\antigravity\scratch\bachelor_degree'
os.makedirs(output_dir, exist_ok=True)

states = ['Alabama','Alaska','Arizona','Arkansas','California','Colorado','Connecticut',
          'Delaware','Florida','Georgia','Hawaii','Idaho','Illinois','Indiana','Iowa',
          'Kansas','Kentucky','Louisiana','Maine','Maryland','Massachusetts','Michigan',
          'Minnesota','Mississippi','Missouri','Montana','Nebraska','Nevada','New Hampshire',
          'New Jersey','New Mexico','New York','North Carolina','North Dakota','Ohio',
          'Oklahoma','Oregon','Pennsylvania','Rhode Island','South Carolina','South Dakota',
          'Tennessee','Texas','Utah','Vermont','Virginia','Washington','West Virginia',
          'Wisconsin','Wyoming','District of Columbia']

age_groups = ['25 to 34', '35 to 44', '45 to 64', '65 and over']
sexes = ['Male', 'Female']

majors = {
    'Science and Engineering': [45000, 15000],
    'Science and Engineering Related': [15000, 25000],
    'Business': [35000, 20000],
    'Education': [8000, 22000],
    'Arts, Humanities and Others': [12000, 18000],
    'Computers, Mathematics, Statistics': [25000, 8000],
    'Biological, Agricultural, Environmental': [10000, 12000],
    'Physical and Related Sciences': [8000, 4000],
    'Psychology': [5000, 15000],
    'Social Sciences': [12000, 10000],
    'Engineering': [30000, 5000],
    'Literature and Languages': [4000, 10000],
}

rows = []
for state in states:
    pop_factor = np.random.uniform(0.3, 3.0)
    if state in ['California','Texas','Florida','New York']:
        pop_factor *= 3
    elif state in ['Wyoming','Vermont','Alaska','North Dakota']:
        pop_factor *= 0.2
    
    for age in age_groups:
        age_factor = {'25 to 34': 1.0, '35 to 44': 0.85, '45 to 64': 0.7, '65 and over': 0.3}[age]
        for sex_idx, sex in enumerate(sexes):
            for major, base_counts in majors.items():
                base = base_counts[sex_idx]
                count = int(base * pop_factor * age_factor * np.random.uniform(0.6, 1.4) / 50)
                count = max(0, count)
                rows.append({
                    'State': state,
                    'Age_Group': age,
                    'Sex': sex,
                    'Major_Field': major,
                    'Count': count,
                    'Total_Population_25plus': int(pop_factor * 500000 * np.random.uniform(0.8, 1.2)),
                    'Percent_With_Bachelors': round(np.random.uniform(18, 45), 1),
                    'Median_Income': int(np.random.uniform(35000, 85000)),
                    'Employment_Rate': round(np.random.uniform(60, 98), 1),
                    'Graduate_Degree_Holders': int(count * np.random.uniform(0.2, 0.5)),
                })

df = pd.DataFrame(rows)
csv_path = os.path.join(output_dir, 'bachelor_degrees.csv')
df.to_csv(csv_path, index=False, encoding='utf-8')

print('Dataset shape:', df.shape)
print('Columns:', list(df.columns))
nums = list(df.select_dtypes(include=[np.number]).columns)
cats = list(df.select_dtypes(include=['object']).columns)
print('Numeric:', nums)
print('Categorical:', cats)
print()
print(df.head().to_string())
print()
print(df.describe().round(2).to_string())

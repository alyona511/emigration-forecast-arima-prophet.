import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("ЗАГРУЗКА И ПРЕОБРАЗОВАНИЕ ДАННЫХ")
print("="*60)

# 1. Загружаем файл
df = pd.read_csv('data/43514.csv')
print("✅ Файл загружен")

# 2. Оставляем только Россию
df_ru = df[df['Классификатор объектов административно-территориального деления (ОКАТО)'] == '643 Российская Федерация']
print(f"✅ Найдено записей по России: {len(df_ru)}")

# 3. Выбираем поток "С другими зарубежными странами" (эмиграция)
df_emig = df_ru[df_ru['Потоки миграции'] == '57 С другими зарубежными странами']
print(f"✅ Найдено записей по эмиграции: {len(df_emig)}")

# 4. Преобразуем из широкого формата в длинный (годы → строки)
years = [str(y) for y in range(1993, 2025)]
df_long = df_emig.melt(
    id_vars=['Период', 'Потоки миграции'],
    value_vars=years,
    var_name='year',
    value_name='emigrants'
)

# 5. Чистим данные
df_long['year'] = pd.to_numeric(df_long['year'])
df_long['emigrants'] = pd.to_numeric(df_long['emigrants'], errors='coerce')
df_long = df_long.dropna(subset=['emigrants'])
df_long = df_long[df_long['emigrants'] >= 0]

# 6. Сортируем по году
df_long = df_long.sort_values('year').reset_index(drop=True)

# 7. Создаём колонку с датой (для Prophet)
df_long['ds'] = pd.to_datetime(df_long['year'], format='%Y')

# 8. Переименовываем для удобства
df_final = df_long[['ds', 'emigrants']].rename(columns={'emigrants': 'y'})

print("\n=== ИТОГОВЫЙ ДАТАФРЕЙМ ===")
print(df_final.head(10))
print(f"\nВсего строк: {len(df_final)}")
print(f"Период: с {df_final['ds'].min()} по {df_final['ds'].max()}")

# 9. Сохраняем
df_final.to_csv('data/emigration_clean.csv', index=False)
print("\n✅ Данные сохранены в data/emigration_clean.csv")

# 10. График
plt.figure(figsize=(12, 6))
plt.plot(df_final['ds'], df_final['y'], marker='o', color='blue', linewidth=2)
plt.title('Эмиграция из России в другие страны, 1993–2024', fontsize=16)
plt.xlabel('Год')
plt.ylabel('Количество эмигрантов')
plt.grid(True, alpha=0.3)
plt.savefig('output/emigration_trend.png', dpi=300)
plt.show()

print("\n✅ ГОТОВО! Теперь можно строить прогноз.")

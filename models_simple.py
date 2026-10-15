import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("ПРОГНОЗ ЭМИГРАЦИИ: ARIMA vs PROPHET")
print("="*60)

# 1. Загружаем данные
df = pd.read_csv('data/emigration_clean.csv')
df['ds'] = pd.to_datetime(df['ds'])
df = df.sort_values('ds').reset_index(drop=True)

print(f"Загружено {len(df)} записей")
print(f"Период: с {df['ds'].min()} по {df['ds'].max()}")

# 2. Разделяем на обучение и тест
train_size = int(len(df) * 0.8)
train = df[:train_size]
test = df[train_size:]

print(f"Обучающая выборка: {len(train)} лет")
print(f"Тестовая выборка: {len(test)} лет")

# 3. ARIMA
print("\n" + "="*60)
print("ARIMA ПРОГНОЗ")
print("="*60)

model_arima = ARIMA(train['y'], order=(1, 1, 1))
fitted_arima = model_arima.fit()
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf

residuals = fitted_arima.resid

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].plot(residuals)
axes[0, 0].axhline(y=0, color='red', linestyle='--')
axes[0, 0].set_title('Остатки ARIMA')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].hist(residuals, bins=20, edgecolor='black', alpha=0.7)
axes[0, 1].set_title('Распределение остатков')

stats.probplot(residuals, dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('Q-Q Plot')

plot_acf(residuals, ax=axes[1, 1], lags=20)
axes[1, 1].set_title('ACF остатков')

plt.tight_layout()
plt.savefig('output/residuals_diagnostics.png', dpi=300)
plt.show()
print(fitted_arima.summary())

# Прогноз на тест
forecast_arima = fitted_arima.forecast(steps=len(test))

# Прогноз на будущее (2025-2028)
future_arima = fitted_arima.forecast(steps=4)
print("\nПрогноз ARIMA на 2025-2028:")
for year, val in zip(range(2025, 2029), future_arima):
    print(f"  {year}: {int(val)}")

# 4. Prophet
print("\n" + "="*60)
print("PROPHET ПРОГНОЗ")
print("="*60)

df_prophet = train[['ds', 'y']].rename(columns={'ds': 'ds', 'y': 'y'})

model_prophet = Prophet(yearly_seasonality=True)
model_prophet.fit(df_prophet)

# Прогноз на тест + будущее
future = model_prophet.make_future_dataframe(periods=len(test) + 4, freq='YS')
forecast = model_prophet.predict(future)

# Берём прогноз для тестового периода
forecast_test = forecast.set_index('ds').loc[test['ds']]['yhat']

# Берём прогноз на 2025-2028
future_dates = pd.date_range('2025', periods=4, freq='YS')
forecast_future = forecast.set_index('ds').loc[future_dates]['yhat']

print("\nПрогноз Prophet на 2025-2028:")
for year, val in zip(range(2025, 2029), forecast_future):
    print(f"  {year}: {int(val)}")

# 5. ГРАФИКИ
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# График 1: Сравнение на тесте
axes[0].plot(test['ds'], test['y'], label='Реальные данные', color='black', linewidth=2)
axes[0].plot(test['ds'], forecast_arima, label='ARIMA', color='red', linestyle='--', linewidth=2)
axes[0].plot(test['ds'], forecast_test, label='Prophet', color='green', linestyle='--', linewidth=2)
axes[0].set_title('Сравнение моделей на тестовой выборке', fontsize=14)
axes[0].set_xlabel('Год')
axes[0].set_ylabel('Количество эмигрантов')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# График 2: Прогноз на будущее
axes[1].plot(df['ds'], df['y'], label='Исторические данные', color='blue', linewidth=2)
axes[1].plot(future_dates, future_arima, label='ARIMA прогноз', color='red', marker='o', markersize=8, linewidth=2)
axes[1].plot(future_dates, forecast_future, label='Prophet прогноз', color='green', marker='s', markersize=8, linewidth=2)
axes[1].axvline(df['ds'].iloc[-1], color='gray', linestyle=':', linewidth=2, label='Начало прогноза')
axes[1].set_title('Прогноз эмиграции на 2025-2028', fontsize=14)
axes[1].set_xlabel('Год')
axes[1].set_ylabel('Количество эмигрантов')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('output/forecast_final.png', dpi=300)
plt.show()

print("\n✅ Готово! График сохранён в output/forecast_final.png")
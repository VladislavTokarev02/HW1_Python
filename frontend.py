import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import aiohttp
import asyncio


def analyze_city_season(df, city_name):
    city_data = df[df['city'] == city_name]
    city_data = city_data.dropna(subset=['temperature', 'timestamp'])

    # Скользящее среднее и стандартное отклонение (30 дней) по сезону
    city_data['season'] = city_data['season'].astype(str)
    city_data['rolling_mean'] = city_data.groupby('season')['temperature'].transform(
        lambda x: x.rolling(window=30, min_periods=1).mean()
    )
    city_data['rolling_std'] = city_data.groupby('season')['temperature'].transform(
        lambda x: x.rolling(window=30, min_periods=1).std()
    )

    city_data['anomaly'] = (
            (city_data['temperature'] > (city_data['rolling_mean'] + 2 * city_data['rolling_std'])) |
            (city_data['temperature'] < (city_data['rolling_mean'] - 2 * city_data['rolling_std']))
    )

    seasonal_profile = city_data.groupby('season').agg(
        avg_temp=('temperature', 'mean'),
        std_temp=('temperature', 'std')
    ).reset_index()

    rolling_stats = city_data.groupby('season').agg(
        rolling_mean=('rolling_mean', 'mean'),
        rolling_std=('rolling_std', 'mean')
    ).reset_index()



    # 5. Среднее, min, max температуры за всё время
    avg_temp_all = city_data['temperature'].mean()
    min_temp_all = city_data['temperature'].min()
    max_temp_all = city_data['temperature'].max()

    # Собираем результаты
    result = {
        'city': city_name,
        'avg_temp_all': avg_temp_all,
        'min_temp_all': min_temp_all,
        'max_temp_all': max_temp_all,
        'seasonal_profile': seasonal_profile,
        'rolling_stats': rolling_stats,
        'anomalies': city_data[city_data['anomaly'] == True][['temperature', 'anomaly']].to_dict(orient='records')
    }

    return result


def merged_df(df, historical_data):
    merged_data = df.copy()

    for city_name, city_data in historical_data.items():

        seasonal_data = city_data['seasonal_profile']


        for season in seasonal_data['season']:
            season_data = seasonal_data[seasonal_data['season'] == season]
            avg_temp_season = season_data['avg_temp'].values[0]
            std_temp_season = season_data['std_temp'].values[0]

            merged_data.loc[(merged_data['city'] == city_name) & (merged_data['season'] == season),
                            'avg_temp_for_season'] = avg_temp_season
            merged_data.loc[(merged_data['city'] == city_name) & (merged_data['season'] == season),
                            'std_temp_for_season'] = std_temp_season

        # Общие статистики для города
        merged_data.loc[merged_data['city'] == city_name, 'avg_temp_all'] = city_data['avg_temp_all']
        merged_data.loc[merged_data['city'] == city_name, 'min_temp_all'] = city_data['min_temp_all']
        merged_data.loc[merged_data['city'] == city_name, 'max_temp_all'] = city_data['max_temp_all']


    merged_data['rolling_mean'] = merged_data.groupby(['city', 'season'])['temperature'].transform(
        lambda x: x.rolling(window=30, min_periods=1).mean()
    )
    merged_data['rolling_std'] = merged_data.groupby(['city', 'season'])['temperature'].transform(
        lambda x: x.rolling(window=30, min_periods=1).std()
    )

    # Определяем аномалии на основе ±2σ от скользящего среднего и стандартного
    merged_data['anomalies'] = merged_data.apply(
        lambda row: 'yes' if abs(row['temperature'] - row['rolling_mean']) > 2 * row['rolling_std'] else 'no',
        axis=1
    )

    return merged_data


st.title("Анализ температурных данных")

# Загрузка файла
uploaded_file = st.file_uploader("Загрузите файл с данными (CSV)", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Загруженные данные:")
    st.dataframe(data.head())

    required_columns = {'city', 'timestamp', 'temperature', 'season'}
    if not required_columns.issubset(data.columns):
        st.error(f"Файл должен содержать столбцы: {', '.join(required_columns)}")
    else:
        cities = data['city'].unique()
        analysis_results = {city: analyze_city_season(data, city) for city in cities}
        st.session_state_analysis_results = analysis_results
        merged_df = merged_df(data, analysis_results)

        st.write("Результирующий датафрейм с анализом:")
        st.dataframe(merged_df)
else:
    st.info("Загрузите файл для анализа.")


@st.cache_data
def plot_for_cities(df):
    # Интерактивные выборы для выбора города через Streamlit
    city_list = df['city'].unique()
    selected_city = st.selectbox("Выберите город:", city_list)

    # Фильтрация данных по выбранному городу
    city_data = df[df['city'] == selected_city]

    # Создание фигуры
    fig, ax = plt.subplots(figsize=(20, 6))

    # Построение графиков
    ax.plot(city_data['timestamp'], city_data['temperature'], label='Температура', color='tab:blue', linewidth=0.8)
    ax.plot(city_data['timestamp'], city_data['avg_temp_for_season'], label='Средняя сезонная температура',
            color='darkviolet', linestyle='--', linewidth=2)

    # Область вокруг сезонных значений (средняя температура ± стандартное отклонение)
    ax.fill_between(city_data['timestamp'],
                    city_data['avg_temp_for_season'] - city_data['std_temp_for_season'],
                    city_data['avg_temp_for_season'] + city_data['std_temp_for_season'],
                    color='tab:green', alpha=0.6, label='Диапазон сезонной температуры')

    # Отображаем аномалии
    anomalies = city_data[city_data['anomalies'] == 'yes']
    ax.scatter(anomalies['timestamp'], anomalies['temperature'], color='gold', label='Аномалии', zorder=5)

    # Настройка графика
    ax.set_title(f"Температурные данные для {selected_city}")
    ax.set_xlabel('Год')
    ax.set_ylabel('Температура (°C)')
    ax.legend()
    ax.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    st.pyplot(fig)




st.title("Анализ температурных данных с использованием API OpenWeather")
api_key = st.text_input("Введите ваш API-ключ OpenWeather", type="password")

# Список доступных городов
cities = ['New York', 'London', 'Paris', 'Tokyo', 'Moscow', 'Sydney',
          'Berlin', 'Beijing', 'Rio de Janeiro', 'Dubai', 'Los Angeles',
          'Singapore', 'Mumbai', 'Cairo', 'Mexico City']


@st.cache_data
def get_url(city, api_key):
    return f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"


async def fetch_weather(city, api_key):
    async with aiohttp.ClientSession() as session:
        url = get_url(city, api_key)
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                return data['main']['temp']
            else:
                st.error(f"Ошибка при получении данных для города {city}: {response.status}")
                return None


async def get_city_temperature(city, api_key):
    temperature = await fetch_weather(city, api_key)
    if temperature is not None:
        st.success(f"Текущая температура в городе {city}: {temperature}°C")
    return temperature


# Выпадающий список для выбора города и сезона
selected_city = st.selectbox("Выберите город для получения температуры:", cities)
seasons = ['Winter', 'Spring', 'Summer', 'Autumn']
current_season = st.selectbox("Выберите текущий сезон:", seasons)

# Кнопка для запроса данных
if st.button("Получить температуру"):
    if not api_key:
        st.error("Пожалуйста, введите API-ключ.")
    else:

        current_temperature = asyncio.run(get_city_temperature(selected_city, api_key))
        st.session_state.current_temperature = current_temperature




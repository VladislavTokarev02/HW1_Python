import requests
import aiohttp
import asyncio

api_key = "24e5777bfd8af22757aae21bc2d5e98f"
city = "Dubai" # сначала для одного города

url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"

response = requests.get(url)

if response.status_code == 200:
    data = response.json()
    temperature = data['main']['temp']
    print(f"Текущая температура в городе {city}: {temperature}°C")
else:
    print(f"Ошибка: {response.status_code}")


cities = ['New York', 'London', 'Paris', 'Tokyo', 'Moscow', 'Sydney',
          'Berlin', 'Beijing', 'Rio de Janeiro', 'Dubai', 'Los Angeles',
          'Singapore', 'Mumbai', 'Cairo', 'Mexico City']

def get_url(city):
    """Формирование URL для API запроса."""
    return f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"

async def fetch_weather(city):
    """Асинхронная функция для получения погоды для одного города."""
    async with aiohttp.ClientSession() as session:
        url = get_url(city)
        try:
            # Отправляем запрос
            async with session.get(url) as response:
                # Проверяем статус-код ответа
                if response.status == 200:
                    data = await response.json()
                    temperature = data['main']['temp']
                    print(f"Текущая температура в городе {city}: {temperature}°C")
                else:
                    print(f"Ошибка при получении данных для города {city}: {response.status}")
        except aiohttp.ClientError as e:
            print(f"Ошибка сети для города {city}: {e}")

async def main():
    """Основная асинхронная функция для обработки всех городов."""
    # Создаем список задач для всех городов
    tasks = [fetch_weather(city) for city in cities]

    # Выполняем все задачи параллельно
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    # Запускаем все асинхронные задачи
    asyncio.run(main())
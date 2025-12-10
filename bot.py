import os
import asyncio
from aiogram import Bot, Dispatcher, F
from aiogram.types import Message
from dotenv import load_dotenv

# Получаем директорию, где лежит bot.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Строим абсолютный путь к .env
ENV_PATH = os.path.join(BASE_DIR, "config", ".env")

# Загружаем переменные
load_dotenv(ENV_PATH)

# Получаем токен
TOKEN = os.getenv("BOT_TOKEN")
if not TOKEN:
    raise RuntimeError("BOT_TOKEN не найден! Проверьте config/.env")

# Инициализация
bot = Bot(token=TOKEN)
dp = Dispatcher()

@dp.message(F.text == "/start")
async def start(message: Message):
    await message.answer("Я родился")

@dp.message()
async def fallback(message: Message):
    await message.answer("Я пока ничего не умею, но обязательно чему-нибудь научусь)")

async def main():
    print("Бот запущен...")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())

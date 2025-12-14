import os
import asyncio
from aiogram import Bot, Dispatcher, F
from aiogram.types import Message
from dotenv import load_dotenv
import tempfile
from model.detection import draw_dog_bbox, detect_best_dog_bbox
from aiogram.types import FSInputFile

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
async def cmd_start(message: Message):
    await message.answer(
        "Привет! 🐾\n\n"
        "Я - **Dog Breed Bot**, ваш виртуальный помощник по определению пород собак 🐕\n\n"
        "🎯 Просто пришлите фото собаки - и я скажу, кто это!\n"
        "💡 Я знаю **120** пород - от лабрадоров до редких овчарок.\n"
        "❗ Но помните: я - ИИ, а не кинолог. Моя оценка не заменяет эксперта.\n\n"
        "Отправьте фото и мы начнём!",
        parse_mode="Markdown"
    )

@dp.message(F.text == "/help")
async def cmd_help(message: Message):
    await message.answer(
        "📸 Как прислать идеальное фото для точного результата:\n\n"
        "✅ Хорошо:\n"
        "• Собака находится в центре кадра\n"
        "• Дневной свет или яркое освещение\n"
        "• Чёткое, не размытое изображение\n"
        "• Один объект (без других собак/людей в кадре)\n\n"
        "❌ Не очень:\n"
        "• Тёмные или пересвеченные фото\n"
        "• Мелкий план\n"
        "• Сильное размытие или движение\n"
        "• Фото игрушек, картинок с экрана, мультиков\n\n"
        "Готовы? Просто отправьте фото 🐶",
        parse_mode="Markdown"
    )

@dp.message(F.text == "/about")
async def cmd_help(message: Message):
    await message.answer(
        "Скоро здесь появится информация о боте и используемых технологиях",
        parse_mode="Markdown"
    )

@dp.message(F.photo)
async def handle_photo(message: Message):
    await message.answer("🔍 Ищу собаку на фото...")
    try:
        # === 1. Получаем фото ===
        photo = message.photo[-1]
        file_info = await bot.get_file(photo.file_id)
        
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_in:
            await bot.download_file(file_info.file_path, tmp_in.name)
            input_path = tmp_in.name

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_out:
            output_path = tmp_out.name

        # === 2. Рисуем bbox ===
        dog_found = draw_dog_bbox(input_path, output_path)

        # === 3. Отправляем результат ===
        photo_to_send = FSInputFile(output_path)
        
        if dog_found:
            await message.answer_photo(
                photo_to_send,
                caption=(
                    "✅ *Отлично\\!* Я нашёл собаку на фото и выделил её красной рамкой\\.\n\n"
                    "❗ *Важно:* если на фото *несколько собак*, я анализирую только *ту, в которой больше всего уверен*\\.\n\n"
                    "💡 *Совет:*\n"
                    "Если рамка выделила не всю собаку или захватила слишком много фона \\- сделайте новое фото:\n"
                    "• Сфотографируйте *крупнее и чётче*\n"
                    "• В кадре должна быть *только одна собака*\n"
                    "• Используйте *хорошее освещение*\n\n"
                    "Так я дам вам *самый точный ответ*\\! 🐾\n\n"
                    "Скоро я научусь определять породу собаки\\)"
                ),
                parse_mode="MarkdownV2"
            )
        else:
            await message.answer_photo(
                photo_to_send,
                caption=(
                    "🤔 *Хммм... Кажется, на фото нет собаки или я не знаю такую породу\\.*\n\n"
                    "💡 *Совет:*\n"
                    "• Убедитесь, что *освещение хорошее*\n"
                    "• И что *собаку чётко видно*"
                ),
                parse_mode="MarkdownV2"
            )

        # === 4. Удаляем временные файлы ===
        os.unlink(input_path)
        os.unlink(output_path)

    except Exception as e:
        await message.answer("Произошла ошибка при обработке фото")
        print(f"Ошибка в боте: {e}")

@dp.message()
async def fallback(message: Message):
    await message.answer("Отправь фото собаки, я найду ее и выделю рамкой")

async def main():
    print("Бот запущен...")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())

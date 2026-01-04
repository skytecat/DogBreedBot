import os
import asyncio
import gc
from aiogram import Bot, Dispatcher, F
from aiogram.types import Message
from dotenv import load_dotenv
import tempfile
from model.detection import draw_dog_bbox, detect_best_dog_bbox
from model.classification import BreedClassifier, crop_image_by_bbox
from aiogram.types import FSInputFile
from PIL import Image
import logging
from data.breed_translation import BREED_TRANSLATION

def escape(text: str) -> str:
    """
    Экранирует специальные символы для Telegram MarkdownV2.
    """
    escape_chars = r'_*[]()~`>#+-=|{}.!'
    return ''.join('\\' + char if char in escape_chars else char for char in text)

logging.basicConfig(
    filename='/root/breedbot/error.log',
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# === Инициализация глобальных объектов ===
classifier = BreedClassifier()
processing_semaphore = asyncio.Semaphore(1)  # ⚠️ Только 1 одновременная обработка

# Пути и токен
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, "config", ".env")
load_dotenv(ENV_PATH)

TOKEN = os.getenv("BOT_TOKEN")
if not TOKEN:
    raise RuntimeError("BOT_TOKEN не найден! Проверьте config/.env")

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

@dp.message(F.text == "/breeds")
async def send_breeds_list(message: Message):
    # Получаем список пород (на русском)
    breeds_rus = list(BREED_TRANSLATION.values())
    breeds_rus.sort()  # сортируем по алфавиту

    # Ограничиваем длину сообщения (Telegram — 4096 символов)
    text = "🐾 Я знаю следующие породы:\n\n" + "\n".join(f"• {breed}" for breed in breeds_rus)

    if len(text) <= 4096:
        await message.answer(text)
    else:
        # Если слишком длинно — разбиваем на части
        parts = []
        current = "🐾 Я знаю следующие породы:\n\n"
        for breed in breeds_rus:
            line = f"• {breed}\n"
            if len(current) + len(line) > 4096:
                parts.append(current)
                current = line
            else:
                current += line
        if current:
            parts.append(current)

        for part in parts:
            await message.answer(part)

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
        "• Сильное размытие или движение\n\n"
        "❗*Важно:* если на фото *несколько собак*, я анализирую только ту, в которой больше всего уверен\n\n"
        "Готовы? Просто отправьте фото 🐶",
        parse_mode="Markdown"
    )


@dp.message(F.text == "/about")
async def cmd_about(message: Message):
    await message.answer(
        "Архитектура:\n"
        "• Детекция: YOLOv8 (находит bounding box собаки)\n"
        "• Классификация: ResNet18 + Logistic Regression\n\n"
        "*Код и более подробное описание проекта:* [GitHub](https://github.com/skytecat/DogBreedBot)",
        parse_mode="Markdown"
    )


@dp.message(F.photo)
async def handle_photo(message: Message):
    # ⚠️ Ограничиваем одновременные запросы
    if processing_semaphore.locked():
        await message.answer("⏳ Бот сейчас обрабатывает другое изображение. Пожалуйста, подождите немного!")
        return

    async with processing_semaphore:
        await message.answer("🔍 Ищу собаку на фото...")
        input_path = tempfile.mktemp(suffix=".jpg")
        output_path = tempfile.mktemp(suffix=".jpg")

        try:
            # === 1. Получаем фото ===
            photo = message.photo[-1]
            file_info = await bot.get_file(photo.file_id)
            await bot.download_file(file_info.file_path, input_path)

            # === 2. Рисуем bbox ===
            dog_found, confidence, bbox = draw_dog_bbox(input_path, output_path)

            photo_to_send = FSInputFile(output_path)

            if dog_found:
                conf_percent = round(confidence * 100)
                await message.answer_photo(
                    photo_to_send,
                    caption=(
                        "✅ *Отлично\\!* Я нашёл собаку на фото и выделил её красной рамкой\\.\n\n"
                        f"*Уверенность:* {conf_percent}\\%\n\n"
                        "💡 *Совет:*\n"
                        "Если рамка выделила не всю собаку или захватила слишком много фона \\- сделайте новое фото:\n"
                        "• Сфотографируйте *крупнее и чётче*\n"
                        "• В кадре должна быть *только одна собака*\n"
                        "• Используйте *хорошее освещение*\n\n"
                        "Так я дам вам *самый точный ответ*\\! 🐾\n\n"
                    ),
                    parse_mode="MarkdownV2"
                )

                # === 3. Анализ породы ===
                await message.answer("🧠 Анализирую породу собаки...")
                cropped_img = crop_image_by_bbox(input_path, bbox)
                breed, conf = classifier.predict(cropped_img)
                conf_percent = round(conf * 100)

                # ⚠️ №2: явно закрываем PIL-изображение
                if hasattr(cropped_img, 'close'):
                    cropped_img.close()
                del cropped_img

                safe_breed = escape(breed)
                await message.answer(
                    f"🐾 *Порода:* {safe_breed}\n"
                    f"📊 *Уверенность:* {conf_percent}%\n\n"
                    "💡 *Совет:*\n"
                    "Если результат неточный — сделайте новое фото:\n"
                    "• Сфотографируйте *крупнее и чётче*\n"
                    "• В кадре должна быть *только одна собака*\n"
                    "• Используйте *хорошее освещение*",
                    parse_mode="MarkdownV2"
                )
            else:
                await message.answer_photo(
                    photo_to_send,
                    caption=(
                        "🤔 Кажется, на фото нет собаки или я не знаю такую породу\n\n"
                        "💡 *Совет:*\n"
                        "• Убедитесь, что освещение хорошее и собаку чётко видно\n"
                    ),
                    parse_mode="MarkdownV2"
                )

        except Exception as e:
            # Отправляем пользователю краткое сообщение
            await message.answer("Произошла ошибка при обработке фото 🛠️\nПопробуйте другое изображение.")

            # Логируем полный стек ошибки + тип ошибки
            error_msg = f"❌ Ошибка при обработке фото: {type(e).__name__}: {e}"
            print(error_msg)  # Выводим в консоль (для systemd)
            
            import traceback
            tb_str = traceback.format_exc()
            print(tb_str)  # Полный стек — ключ к диагностике!

        finally:
            # ⚠️ №1: надёжное удаление временных файлов
            for path in (input_path, output_path):
                try:
                    if os.path.exists(path):
                        os.unlink(path)
                except OSError as ex:
                    print(f"Не удалось удалить временный файл {path}: {ex}")

            # ⚠️ №2: принудительная сборка мусора
            gc.collect()


@dp.message()
async def fallback(message: Message):
    await message.answer("Отправь фото собаки, я найду ее и определю породу")


async def main():
    print("Бот запущен...")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
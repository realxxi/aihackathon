import os
import json
import requests
import numpy as np
from io import BytesIO
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import torch
import telebot
from telebot import types
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from datetime import datetime
import time
import sqlite3
from contextlib import contextmanager

load_dotenv()

API_TOKEN = os.getenv("TOKEN")
bot = telebot.TeleBot(API_TOKEN)

# Statistika fayllari uchun papka yaratish
STATS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bot_statistics")
TRAINING_IMAGES_DIR = os.path.join(STATS_DIR, "training_images")

# Papkalarni yaratish va huquqlarni tekshirish
def ensure_directories():
    try:
        # Asosiy papkani yaratish
        if not os.path.exists(STATS_DIR):
            os.makedirs(STATS_DIR, mode=0o777)
            print(f"Created directory: {STATS_DIR}")
            
        # Training images papkasini yaratish    
        if not os.path.exists(TRAINING_IMAGES_DIR):
            os.makedirs(TRAINING_IMAGES_DIR, mode=0o777)
            print(f"Created directory: {TRAINING_IMAGES_DIR}")
            
        # Huquqlarni tekshirish
        if not os.access(TRAINING_IMAGES_DIR, os.W_OK):
            print(f"Warning: No write access to {TRAINING_IMAGES_DIR}")
            os.chmod(TRAINING_IMAGES_DIR, 0o777)
            print(f"Changed permissions for {TRAINING_IMAGES_DIR}")
            
    except Exception as e:
        print(f"Error creating directories: {str(e)}")
        raise e

# SQLite ma'lumotlar bazasi
DB_PATH = os.path.join(STATS_DIR, "statistics.db")

# SQLite bilan ishlash uchun context manager
@contextmanager
def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
    finally:
        conn.close()

# Ma'lumotlar bazasini yaratish
def init_database():
    ensure_directories()  # Papkalarni tekshirish
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Foydalanuvchilar jadvali
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY,
            username TEXT,
            first_activity TIMESTAMP,
            last_activity TIMESTAMP,
            total_requests INTEGER DEFAULT 0
        )
        ''')
        
        # Kasalliklar aniqlash tarixi
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS disease_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            disease_name TEXT,
            confidence REAL,
            detected_at TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
        ''')
        
        # Foydalanuvchi tilini saqlash jadvali
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_languages (
            user_id INTEGER PRIMARY KEY,
            language TEXT DEFAULT 'uz',
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
        ''')
        
        conn.commit()

# Foydalanuvchi tilini olish
def get_user_language(chat_id):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT language FROM user_languages WHERE user_id = ?', (chat_id,))
        result = cursor.fetchone()
        return result[0] if result else 'uz'

# Foydalanuvchi tilini saqlash
def set_user_language(chat_id, language):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
        INSERT INTO user_languages (user_id, language) 
        VALUES (?, ?) 
        ON CONFLICT(user_id) DO UPDATE SET language = ?
        ''', (chat_id, language, language))
        conn.commit()

# Foydalanuvchi statistikasini yangilash
def update_user_statistics(user_id, username, disease=None, confidence=None):
    current_time = datetime.now()
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Foydalanuvchi mavjudligini tekshirish
        cursor.execute('SELECT * FROM users WHERE user_id = ?', (user_id,))
        user = cursor.fetchone()
        
        if user is None:
            # Yangi foydalanuvchi qo'shish
            cursor.execute('''
            INSERT INTO users (user_id, username, first_activity, last_activity, total_requests)
            VALUES (?, ?, ?, ?, 1)
            ''', (user_id, username, current_time, current_time))
        else:
            # Mavjud foydalanuvchi ma'lumotlarini yangilash
            cursor.execute('''
            UPDATE users 
            SET last_activity = ?, total_requests = total_requests + 1
            WHERE user_id = ?
            ''', (current_time, user_id))
        
        # Aniqlangan kasallikni saqlash
        if disease and disease != "not_detected":
            cursor.execute('''
            INSERT INTO disease_history (user_id, disease_name, confidence, detected_at)
            VALUES (?, ?, ?, ?)
            ''', (user_id, disease, confidence, current_time))
        
        conn.commit()

# Foydalanuvchi statistikasini olish
def get_user_stats(user_id):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Foydalanuvchi ma'lumotlari
        cursor.execute('''
        SELECT username, first_activity, last_activity, total_requests 
        FROM users WHERE user_id = ?
        ''', (user_id,))
        user_data = cursor.fetchone()
        
        if not user_data:
            return None
            
        # Kasalliklar tarixi
        cursor.execute('''
        SELECT disease_name, COUNT(*) as count 
        FROM disease_history 
        WHERE user_id = ? 
        GROUP BY disease_name
        ORDER BY count DESC
        ''', (user_id,))
        diseases = cursor.fetchall()
        
        return {
            'username': user_data[0],
            'first_activity': user_data[1],
            'last_activity': user_data[2],
            'total_requests': user_data[3],
            'diseases': diseases
        }

# Statistika rasmini generatsiya qilish
def generate_user_statistics_image(user_id):
    # Function removed - grafik chizish funksiyasi olib tashlandi
    return None, None

# Asosiy menyu uchun keyboard yaratish
def get_main_keyboard(lang, user_id=None):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, row_width=2)
    
    stats_btn = types.KeyboardButton("📊 Statistika")
    help_btn = types.KeyboardButton("❓ Yordam")
    lang_btn = types.KeyboardButton("🌐 Til")
    
    markup.add(stats_btn)
    markup.add(help_btn, lang_btn)
    
    # Admin uchun qo'shimcha tugmalar
    if user_id and is_admin(user_id):
        admin_btn = types.KeyboardButton("👨‍💻 Admin panel")
        markup.add(admin_btn)
    
    return markup

@bot.message_handler(commands=['start'])
def send_welcome(message):
    markup = get_language_keyboard()
    lang = get_user_language(message.chat.id)
    
    # Avval til tanlash inline buttonlarini ko'rsatamiz
    bot.send_message(
        message.chat.id,
        "👋 Welcome! / Добро пожаловать! / Xush kelibsiz!",
        reply_markup=markup
    )
    
    # So'ng asosiy menyu keyboard buttonlarini ko'rsatamiz
    keyboard = get_main_keyboard(lang, message.from_user.id)
    bot.send_message(
        message.chat.id,
        messages[lang].get("welcome", "Botdan foydalanish uchun quyidagi tugmalardan foydalaning:"),
        reply_markup=keyboard
    )

    # Foydalanuvchi statistikasini yangilash
    username = message.from_user.username or f"{message.from_user.first_name} {message.from_user.last_name or ''}"
    update_user_statistics(message.from_user.id, username)

@bot.message_handler(func=lambda message: message.text == "❓ Yordam")
def show_help(message):
    lang = get_user_language(message.chat.id)
    bot.reply_to(message, messages[lang]["help_text"], parse_mode='Markdown')

@bot.message_handler(func=lambda message: message.text == "🌐 Til")
def change_language_keyboard(message):
    markup = get_language_keyboard()
    bot.reply_to(message, "Choose your language / Выберите язык / Tilni tanlang:", reply_markup=markup)

@bot.callback_query_handler(func=lambda call: call.data.startswith('lang_'))
def callback_language(call):
    lang = call.data.split('_')[1]
    set_user_language(call.message.chat.id, lang)
    bot.answer_callback_query(call.id)
    
    # Til o'zgartirilgandan so'ng yangi keyboard bilan xabar yuborish
    keyboard = get_main_keyboard(lang, call.from_user.id)
    bot.edit_message_text(
        chat_id=call.message.chat.id,
        message_id=call.message.message_id,
        text=messages[lang]["language_selected"]
    )
    bot.send_message(
        call.message.chat.id,
        messages[lang].get("menu_message", "Botdan foydalanish uchun quyidagi tugmalardan foydalaning:"),
        reply_markup=keyboard
    )

@bot.message_handler(func=lambda message: message.text == "📊 Statistika")
def show_text_stats(message):
    lang = get_user_language(message.chat.id)
    try:
        stats = get_user_stats(message.chat.id)
        if not stats:
            bot.reply_to(message, "Statistika mavjud emas.")
            return
            
        # Statistika matni
        stats_text = f"{messages[lang]['stats_title']}\n\n"
        stats_text += f"👤 Foydalanuvchi: {stats['username']}\n"
        stats_text += f"📅 Birinchi faollik: {stats['first_activity']}\n"
        stats_text += f"🕒 Oxirgi faollik: {stats['last_activity']}\n"
        stats_text += f"📊 Jami so'rovlar: {stats['total_requests']}\n\n"
        
        if stats['diseases']:
            stats_text += f"🦠 {messages[lang]['stats_diseases']}\n"
            for disease, count in stats['diseases']:
                translated_disease = disease_names[lang].get(disease, disease)
                stats_text += f"- {translated_disease}: {count}\n"
        
        bot.reply_to(message, stats_text)
        
    except Exception as e:
        bot.reply_to(message, f"Xatolik yuz berdi: {str(e)}")

@bot.message_handler(func=lambda message: message.text == "👨‍💻 Admin panel")
def show_admin_panel(message):
    if not is_admin(message.from_user.id):
        bot.reply_to(message, "Bu funksiya faqat adminlar uchun mavjud.")
        return
    
    lang = get_user_language(message.chat.id)
    markup = get_admin_keyboard(lang)
    bot.reply_to(
        message,
        "Admin panel:\n\n"
        "• Statistika\n"
        "• Foydalanuvchilar",
        reply_markup=markup
    )

# Admin keyboard yaratish
def get_admin_keyboard(lang):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, row_width=2)
    
    admin_stats_btn = types.KeyboardButton("📊 Admin statistika")
    users_btn = types.KeyboardButton("👥 Foydalanuvchilar")
    back_btn = types.KeyboardButton("🔙 Asosiy menyu")
    
    markup.add(admin_stats_btn, users_btn)
    markup.add(back_btn)
    
    return markup

@bot.message_handler(func=lambda message: message.text == "🔙 Asosiy menyu")
def back_to_main_menu(message):
    lang = get_user_language(message.chat.id)
    keyboard = get_main_keyboard(lang, message.from_user.id)
    bot.send_message(
        message.chat.id,
        messages[lang].get("menu_message", "Asosiy menyu:"),
        reply_markup=keyboard
    )

@bot.message_handler(func=lambda message: message.text == "📊 Admin statistika")
def show_admin_stats(message):
    if not is_admin(message.from_user.id):
        return
        
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Umumiy statistika
            cursor.execute('SELECT COUNT(*) FROM users')
            total_users = cursor.fetchone()[0]
            
            cursor.execute('SELECT SUM(total_requests) FROM users')
            total_requests = cursor.fetchone()[0] or 0
            
            cursor.execute('SELECT COUNT(*) FROM disease_history')
            total_detections = cursor.fetchone()[0]
            
            # Oxirgi 7 kun statistikasi
            cursor.execute('''
                SELECT COUNT(*) 
                FROM disease_history 
                WHERE detected_at >= date('now', '-7 days')
            ''')
            last_week_detections = cursor.fetchone()[0]
            
            # Top 5 kasalliklar
            cursor.execute('''
                SELECT disease_name, COUNT(*) as count
                FROM disease_history
                GROUP BY disease_name
                ORDER BY count DESC
                LIMIT 5
            ''')
            top_diseases = cursor.fetchall()
            
            # Statistika matni
            text = "📊 Admin statistika:\n\n"
            text += f"👥 Jami foydalanuvchilar: {total_users}\n"
            text += f"📝 Jami so'rovlar: {total_requests}\n"
            text += f"🔍 Jami kasallik aniqlashlar: {total_detections}\n"
            text += f"📅 Oxirgi 7 kundagi aniqlashlar: {last_week_detections}\n\n"
            
            text += "🏆 Top 5 kasalliklar:\n"
            for disease, count in top_diseases:
                text += f"• {disease}: {count} marta\n"
            
            bot.reply_to(message, text)
            
    except Exception as e:
        bot.reply_to(message, f"Xatolik yuz berdi: {str(e)}")

@bot.message_handler(func=lambda message: message.text == "👥 Foydalanuvchilar")
def show_users_list(message):
    if not is_admin(message.from_user.id):
        return
        
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT username, first_activity, last_activity, total_requests
                FROM users
                ORDER BY total_requests DESC
                LIMIT 20
            ''')
            users = cursor.fetchall()
            
            text = "👥 Foydalanuvchilar ro'yxati (top 20):\n\n"
            for user in users:
                text += f"👤 {user[0]}\n"
                text += f"📅 Birinchi faollik: {user[1]}\n"
                text += f"🕒 Oxirgi faollik: {user[2]}\n"
                text += f"📊 So'rovlar: {user[3]}\n"
                text += "➖➖➖➖➖➖➖➖\n"
            
            bot.reply_to(message, text)
            
    except Exception as e:
        bot.reply_to(message, f"Xatolik yuz berdi: {str(e)}")

# Ma'lumotlar bazasini ishga tushirish
init_database()

messages = {
    "uz": {
        "welcome": "Assalomu alaykum! O'simlik kasalliklarini aniqlash botiga xush kelibsiz.\nIltimos, tilingizni tanlang:",
        "language_selected": "O'zbek tili tanlandi. Endi tekshirmoqchi bo'lgan o'simlik rasmini yuboring.",
        "processing": "Rasm tahlil qilinmoqda...",
        "disease_detected": "Aniqlangan kasallik: ",
        "accuracy": "Aniqlik darajasi: ",
        "low_accuracy_warning": "Diqqat: Aniqlik darajasi past, mutaxassis bilan maslahatlashing!",
        "recommendations": "Maslahatlar:",
        "error": "Xatolik yuz berdi: ",
        "help_text": "*O'simlik kasalliklarini aniqlash boti*\n\n📸 Botdan foydalanish uchun o'simlik rasmini yuboring\n🌱 Bot o'simlik kasalligini aniqlaydi va maslahat beradi\n📊 Aniqlik darajasi ham ko'rsatiladi\n\n/start - Botni ishga tushirish\n/help - Ushbu yordam xabarini ko'rish\n/language - Tilni o'zgartirish\n/stats - Statistikani ko'rish",
        "stats_title": "Bot statistikasi",
        "stats_users": "Jami foydalanuvchilar: ",
        "stats_requests": "Jami so'rovlar: ",
        "stats_diseases": "Eng ko'p aniqlangan kasalliklar:",
        "stats_send": "Statistika yuborilmoqda..."
    },
    "en": {
        "welcome": "Hello! Welcome to the Plant Disease Detection bot.\nPlease select your language:",
        "language_selected": "English language selected. Now please send a photo of the plant you want to check.",
        "processing": "Analyzing the image...",
        "disease_detected": "Detected disease: ",
        "accuracy": "Accuracy level: %",
        "low_accuracy_warning": "Attention: Low accuracy level, please consult a specialist!",
        "recommendations": "Recommendations:",
        "error": "An error occurred: ",
        "help_text": "*Plant Disease Detection Bot*\n\n📸 Send a plant photo to use the bot\n🌱 The bot will detect plant diseases and provide advice\n📊 Accuracy level is also shown\n\n/start - Start the bot\n/help - View this help message\n/language - Change language\n/stats - View statistics",
        "stats_title": "Bot Statistics",
        "stats_users": "Total users: ",
        "stats_requests": "Total requests: ",
        "stats_diseases": "Most detected diseases:",
        "stats_send": "Sending statistics..."
    },
    "ru": {
        "welcome": "Здравствуйте! Добро пожаловать в бот определения болезней растений.\nПожалуйста, выберите язык:",
        "language_selected": "Выбран русский язык. Теперь отправьте фотографию растения, которое хотите проверить.",
        "processing": "Анализ изображения...",
        "disease_detected": "Обнаруженное заболевание: ",
        "accuracy": "Уровень точности: ",
        "low_accuracy_warning": "Внимание: Низкий уровень точности, проконсультируйтесь со специалистом!",
        "recommendations": "Рекомендации:",
        "error": "Произошла ошибка: ",
        "help_text": "*Бот определения болезней растений*\n\n📸 Отправьте фото растения для использования бота\n🌱 Бот определит болезни растений и даст советы\n📊 Также показывается уровень точности\n\n/start - Запустить бота\n/help - Просмотреть это сообщение помощи\n/language - Изменить язык\n/stats - Просмотр статистики",
        "stats_title": "Статистика бота",
        "stats_users": "Всего пользователей: ",
        "stats_requests": "Всего запросов: ",
        "stats_diseases": "Наиболее обнаруженные болезни:",
        "stats_send": "Отправка статистики..."
    }
}

# To'liq o'simlik kasalliklari nomlarini uch tilda saqlash
disease_names = {
    "uz": {
        "Tomato___Late_blight": "Pomidor - Kechki fitoftoroz",
        "Tomato___healthy": "Pomidor - Sog'lom",
        "Pepper___healthy": "Qalampir - Sog'lom",
        "Tomato___Early_blight": "Pomidor - Erta fitoftoroz",
        "Tomato___Septoria_leaf_spot": "Pomidor - Septoria barg dog'lanishi",
        "Tomato___Bacterial_spot": "Pomidor - Bakterial dog'lanish",
        "Tomato___Target_Spot": "Pomidor - Nishon dog'lanishi",
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Pomidor - Sariq barg o'ralish virusi",
        "Tomato___Tomato_mosaic_virus": "Pomidor - Mozaika virusi",
        "Tomato___Spider_mites Two-spotted_spider_mite": "Pomidor - O'rgimchak kana",
        "Tomato___Leaf_Mold": "Pomidor - Barg mog'ori",
        "Pepper___Bacterial_spot": "Qalampir - Bakterial dog'lanish",
        "Potato___Early_blight": "Kartoshka - Erta fitoftoroz",
        "Potato___Late_blight": "Kartoshka - Kechki fitoftoroz",
        "Potato___healthy": "Kartoshka - Sog'lom",
        "Corn_(maize)___Common_rust_": "Makkajo'xori - Oddiy zang",
        "Corn_(maize)___Northern_Leaf_Blight": "Makkajo'xori - Shimoliy barg kuyishi",
        "Corn_(maize)___healthy": "Makkajo'xori - Sog'lom",
        "Apple___Apple_scab": "Olma - Olma qo'tiri",
        "Apple___Black_rot": "Olma - Qora chirish",
        "Apple___Cedar_apple_rust": "Olma - Kedr zang",
        "Apple___healthy": "Olma - Sog'lom"
    },
    "en": {
        "Tomato___Late_blight": "Tomato - Late blight",
        "Tomato___healthy": "Tomato - Healthy",
        "Pepper___healthy": "Pepper - Healthy",
        "Tomato___Early_blight": "Tomato - Early blight",
        "Tomato___Septoria_leaf_spot": "Tomato - Septoria leaf spot",
        "Tomato___Bacterial_spot": "Tomato - Bacterial spot",
        "Tomato___Target_Spot": "Tomato - Target Spot",
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Tomato - Yellow Leaf Curl Virus",
        "Tomato___Tomato_mosaic_virus": "Tomato - Mosaic virus",
        "Tomato___Spider_mites Two-spotted_spider_mite": "Tomato - Two-spotted spider mite",
        "Tomato___Leaf_Mold": "Tomato - Leaf Mold",
        "Pepper___Bacterial_spot": "Pepper - Bacterial spot",
        "Potato___Early_blight": "Potato - Early blight",
        "Potato___Late_blight": "Potato - Late blight",
        "Potato___healthy": "Potato - Healthy",
        "Corn_(maize)___Common_rust_": "Corn - Common rust",
        "Corn_(maize)___Northern_Leaf_Blight": "Corn - Northern Leaf Blight",
        "Corn_(maize)___healthy": "Corn - Healthy",
        "Apple___Apple_scab": "Apple - Apple scab",
        "Apple___Black_rot": "Apple - Black rot",
        "Apple___Cedar_apple_rust": "Apple - Cedar apple rust",
        "Apple___healthy": "Apple - Healthy"
    },
    "ru": {
        "Tomato___Late_blight": "Помидор - Поздний фитофтороз",
        "Tomato___healthy": "Помидор - Здоровый",
        "Pepper___healthy": "Перец - Здоровый",
        "Tomato___Early_blight": "Помидор - Ранний фитофтороз",
        "Tomato___Septoria_leaf_spot": "Помидор - Септориоз листьев",
        "Tomato___Bacterial_spot": "Помидор - Бактериальная пятнистость",
        "Tomato___Target_Spot": "Помидор - Целевая пятнистость",
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Помидор - Вирус желтого скручивания листьев",
        "Tomato___Tomato_mosaic_virus": "Помидор - Вирус мозаики",
        "Tomato___Spider_mites Two-spotted_spider_mite": "Помидор - Паутинный клещ",
        "Tomato___Leaf_Mold": "Помидор - Плесень листьев",
        "Pepper___Bacterial_spot": "Перец - Бактериальная пятнистость",
        "Potato___Early_blight": "Картофель - Ранний фитофтороз",
        "Potato___Late_blight": "Картофель - Поздний фитофтороз",
        "Potato___healthy": "Картофель - Здоровый",
        "Corn_(maize)___Common_rust_": "Кукуруза - Обычная ржавчина",
        "Corn_(maize)___Northern_Leaf_Blight": "Кукуруза - Северный листовой ожог",
        "Corn_(maize)___healthy": "Кукуруза - Здоровая",
        "Apple___Apple_scab": "Яблоко - Яблочная парша",
        "Apple___Black_rot": "Яблоко - Черная гниль",
        "Apple___Cedar_apple_rust": "Яблоко - Кедровая ржавчина",
        "Apple___healthy": "Яблоко - Здоровое"
    }
}

# Kasalliklarga qarshi choralar lug'ati uch tilda
remedies = {
    "uz": {
        "Pomidor - Kechki fitoftoroz": "Davolash usullari: 1) Zararlangan barglarni yig'ib yo'q qilish, 2) Mis tarkibli fungitsidlar bilan ishlash, 3) O'simlik orasini yaxshi shamollatish",
        "Pomidor - Erta fitoftoroz": "Davolash usullari: 1) Zararlangan qismlarni kesib tashlash, 2) Fungitsidlar bilan davolash, 3) Almashlab ekishni yo'lga qo'yish",
        "Pomidor - Septoria barg dog'lanishi": "Davolash usullari: 1) Zararlangan barglarni olib tashlash, 2) Fungitsid bilan ishlov berish, 3) Yomg'ir suvi tegmaydigan qilib ekish",
        "Pomidor - Bakterial dog'lanish": "Davolash usullari: 1) Zararlangan o'simliklarni yo'q qilish, 2) Mis tarkibli preparatlar bilan davolash, 3) Kasallik tarqalmasligi uchun tomchilab sug'orish",
        "Qalampir - Bakterial dog'lanish": "Davolash usullari: 1) Bakteritsidlar bilan ishlov berish, 2) Zararlangan o'simliklarni yo'q qilish, 3) Yaxshi drenaj tizimini yaratish",
        "Kartoshka - Erta fitoftoroz": "Davolash usullari: 1) Fungitsidlar qo'llash, 2) Kasallangan qismlarni olib tashlash, 3) Almashlab ekish",
        "Kartoshka - Kechki fitoftoroz": "Davolash usullari: 1) Mis asosli fungitsidlar qo'llash, 2) Zararlangan o'simliklarni yo'q qilish, 3) Kartoshkani vaqtida kavlab olish",
        "Makkajo'xori - Oddiy zang": "Davolash usullari: 1) Fungitsidlar bilan davolash, 2) Chidamli navlarni ekish, 3) Zang qarshi kurashuvchi preparatlar qo'llash",
        "Makkajo'xori - Shimoliy barg kuyishi": "Davolash usullari: 1) Fungitsidlar qo'llash, 2) Kasallangan o'simliklarni yo'q qilish, 3) Chidamli navlarni ekish",
        "Olma - Olma qo'tiri": "Davolash usullari: 1) Fungitsidlar bilan ishlov berish, 2) Kasallangan barglarni yig'ib yo'q qilish, 3) Daraxtlar atrofini toza saqlash",
        "Olma - Qora chirish": "Davolash usullari: 1) Zararlangan qismlarni kesib tashlash, 2) Fungitsidlar bilan davolash, 3) Bog'ni sanitariya holatini yaxshilash",
        "Olma - Kedr zang": "Davolash usullari: 1) Zangga qarshi fungitsidlar qo'llash, 2) Erta bahorda profilaktik ishlov berish, 3) Archa daraxtlarini bog' yaqinida ekmaslik"
    },
    "en": {
        "Tomato - Late blight": "Treatment methods: 1) Remove and destroy infected leaves, 2) Treat with copper-based fungicides, 3) Ensure good air circulation between plants",
        "Tomato - Early blight": "Treatment methods: 1) Remove affected parts, 2) Treat with fungicides, 3) Practice crop rotation",
        "Tomato - Septoria leaf spot": "Treatment methods: 1) Remove infected leaves, 2) Apply fungicide, 3) Plant in a way to avoid rainwater contact",
        "Tomato - Bacterial spot": "Treatment methods: 1) Remove infected plants, 2) Treat with copper-based preparations, 3) Use drip irrigation to prevent disease spread",
        "Pepper - Bacterial spot": "Treatment methods: 1) Apply bactericides, 2) Remove infected plants, 3) Create good drainage system",
        "Potato - Early blight": "Treatment methods: 1) Apply fungicides, 2) Remove infected parts, 3) Practice crop rotation",
        "Potato - Late blight": "Treatment methods: 1) Apply copper-based fungicides, 2) Remove infected plants, 3) Harvest potatoes on time",
        "Corn - Common rust": "Treatment methods: 1) Treat with fungicides, 2) Plant resistant varieties, 3) Apply anti-rust preparations",
        "Corn - Northern Leaf Blight": "Treatment methods: 1) Apply fungicides, 2) Remove infected plants, 3) Plant resistant varieties",
        "Apple - Apple scab": "Treatment methods: 1) Treat with fungicides, 2) Collect and destroy infected leaves, 3) Keep the area around trees clean",
        "Apple - Black rot": "Treatment methods: 1) Cut off infected parts, 2) Treat with fungicides, 3) Improve orchard sanitation",
        "Apple - Cedar apple rust": "Treatment methods: 1) Apply anti-rust fungicides, 2) Preventive treatment in early spring, 3) Avoid planting cedar trees near the orchard"
    },
    "ru": {
        "Помидор - Поздний фитофтороз": "Методы лечения: 1) Сбор и уничтожение пораженных листьев, 2) Обработка медьсодержащими фунгицидами, 3) Обеспечение хорошей вентиляции между растениями",
        "Помидор - Ранний фитофтороз": "Методы лечения: 1) Удаление пораженных частей, 2) Обработка фунгицидами, 3) Севооборот",
        "Помидор - Септориоз листьев": "Методы лечения: 1) Удаление зараженных листьев, 2) Обработка фунгицидом, 3) Посадка с защитой от дождевой воды",
        "Помидор - Бактериальная пятнистость": "Методы лечения: 1) Удаление зараженных растений, 2) Обработка медьсодержащими препаратами, 3) Использование капельного полива для предотвращения распространения болезни",
        "Перец - Бактериальная пятнистость": "Методы лечения: 1) Обработка бактерицидами, 2) Удаление зараженных растений, 3) Создание хорошей дренажной системы",
        "Картофель - Ранний фитофтороз": "Методы лечения: 1) Применение фунгицидов, 2) Удаление зараженных частей, 3) Севооборот",
        "Картофель - Поздний фитофтороз": "Методы лечения: 1) Применение медьсодержащих фунгицидов, 2) Удаление пораженных растений, 3) Своевременная уборка картофеля",
        "Кукуруза - Обычная ржавчина": "Методы лечения: 1) Обработка фунгицидами, 2) Посадка устойчивых сортов, 3) Применение противоржавчинных препаратов",
        "Кукуруза - Северный листовой ожог": "Методы лечения: 1) Применение фунгицидов, 2) Удаление зараженных растений, 3) Посадка устойчивых сортов",
        "Яблоко - Яблочная парша": "Методы лечения: 1) Обработка фунгицидами, 2) Сбор и уничтожение зараженных листьев, 3) Поддержание чистоты вокруг деревьев",
        "Яблоко - Черная гниль": "Методы лечения: 1) Обрезка пораженных частей, 2) Обработка фунгицидами, 3) Улучшение санитарного состояния сада",
        "Яблоко - Кедровая ржавчина": "Методы лечения: 1) Применение противоржавчинных фунгицидов, 2) Профилактическая обработка ранней весной, 3) Избегать посадки кедровых деревьев рядом с садом"
    }
}

# Modellarni yuklash
try:
    # Modelni to'g'ridan-to'g'ri yuklash
    model_id = "Hemg/New-plant-diseases-classification"
    model = ViTForImageClassification.from_pretrained(model_id)
    processor = ViTImageProcessor.from_pretrained(model_id)
    print("Modellar muvaffaqiyatli yuklandi.")
except Exception as e:
    print(f"Modellarni yuklashda xatolik: {str(e)}")

# Bashorat qilish funksiyasi
def predict_with_model(image):
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_class_id = logits.argmax(-1).item()
    predicted_class_label = model.config.id2label[predicted_class_id]
    confidence = torch.nn.functional.softmax(logits, dim=-1)[0][predicted_class_id].item()

    return predicted_class_label, confidence

# Til tanlash uchun klaviatura
def get_language_keyboard():
    markup = types.InlineKeyboardMarkup(row_width=3)
    btn_uz = types.InlineKeyboardButton("🇺🇿 O'zbekcha", callback_data='lang_uz')
    btn_en = types.InlineKeyboardButton("🇬🇧 English", callback_data='lang_en')
    btn_ru = types.InlineKeyboardButton("🇷🇺 Русский", callback_data='lang_ru')
    markup.add(btn_uz, btn_en, btn_ru)
    return markup

@bot.message_handler(commands=['help'])
def send_help(message):
    lang = get_user_language(message.chat.id)
    bot.reply_to(message, messages[lang]["help_text"], parse_mode='Markdown')

@bot.message_handler(commands=['language'])
def change_language(message):
    markup = get_language_keyboard()
    bot.reply_to(message, "Choose your language / Выберите язык / Tilni tanlang:", reply_markup=markup)

@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    try:
        lang = get_user_language(message.chat.id)
        bot.reply_to(message, messages[lang]["processing"])

        # Rasmni yuklab olish
        file_info = bot.get_file(message.photo[-1].file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        
        # Rasmni PIL Image formatiga o'tkazish
        image = Image.open(BytesIO(downloaded_file))
        
        # Rasmni model uchun qayta ishlash
        inputs = processor(images=image, return_tensors="pt")
        
        # Bashorat qilish
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            
            # Eng yuqori ehtimollikdagi klassni topish
            predicted_class_idx = logits.argmax(-1).item()
            predicted_class = model.config.id2label[predicted_class_idx]
            
            # Ishonchlilik darajasini hisoblash
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            confidence = probabilities[0][predicted_class_idx].item()
            confidence_percentage = round(confidence * 100, 2)
        
        # Natijalarni tayyorlash
        disease_name = disease_names[lang].get(predicted_class, predicted_class)
        
        # Natijalarni yuborish
        response = messages[lang]["disease_detected"] + disease_name + "\n\n"
        response += messages[lang]["accuracy"] + str(confidence_percentage) + "%\n\n"
        
        if confidence < 0.7:  # 70% dan past aniqlik
            response += messages[lang]["low_accuracy_warning"] + "\n\n"
        
        # Davolash usullarini qo'shish
        if disease_name in remedies[lang]:
            response += messages[lang]["recommendations"] + "\n" + remedies[lang][disease_name]
        
        bot.reply_to(message, response)
        
        # Statistikani yangilash
        username = message.from_user.username or f"{message.from_user.first_name} {message.from_user.last_name or ''}"
        update_user_statistics(message.from_user.id, username, predicted_class, confidence)

    except Exception as e:
        bot.reply_to(message, messages[lang]["error"] + str(e))

# Admin foydalanuvchilar ro'yxati
ADMIN_IDS = [int(id_) for id_ in os.getenv("ADMIN_IDS", "").split(",") if id_]

# Admin tekshirish
def is_admin(user_id):
    return user_id in ADMIN_IDS

# Botni ishga tushirish
if __name__ == "__main__":
    try:
        ensure_directories()  # Papkalarni tekshirish
        print("Bot ishga tushdi...")
        bot.infinity_polling()
    except Exception as e:
        print(f"Error starting bot: {str(e)}")
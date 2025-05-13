# Asosiy imejni tanlash
FROM python:3.9

# Ishchi papkani o'rnatish
WORKDIR /app

# Fayllarni ko'chirish
COPY requirements.txt .
COPY app.py .
COPY .env .

# Kerakli kutubxonalarni o'rnatish
RUN pip install --no-cache-dir -r requirements.txt

# Ilovani ishga tushirish
CMD ["python", "app.py"]

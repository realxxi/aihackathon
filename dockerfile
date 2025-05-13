# Python asosida image
FROM python:3.11-slim

# Ishchi katalog
WORKDIR /app

# Fayllarni konteynerga o‘tkazish
COPY . /app

# Agar mavjud bo‘lsa, kutubxonalarni o‘rnatish
RUN pip install --no-cache-dir -r requirements.txt

# Dasturni ishga tushirish
CMD ["python", "app.py"]

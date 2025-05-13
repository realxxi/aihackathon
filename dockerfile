# Asosiy image
FROM python:3.11-slim

# Ishchi papkani yaratish va unda ishlash
WORKDIR /app

# Fayllarni konteynerga nusxalash
COPY . /app

# Talablarni o‘rnatish (agar mavjud bo‘lsa)
RUN pip install --no-cache-dir -r requirements.txt

# Dasturni ishga tushirish
CMD ["python", "app.py"]

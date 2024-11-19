# Gunakan image Python yang sesuai sebagai base image
FROM python:3.8.0

# Set working directory di dalam container
WORKDIR /app

# Salin file requirements.txt ke dalam container
COPY requirements.txt /app/

# Install dependencies dari requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install NLTK dan unduh resource 'punkt'
RUN pip install nltk
RUN python -c "import nltk; nltk.download('punkt')"

# Salin semua file dari aplikasi ke dalam container
COPY . /app/

# Jalankan aplikasi dengan Gunicorn
CMD ["gunicorn", "app:app"]

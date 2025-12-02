FROM python:3.10-bullseye

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    software-properties-common \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libxkbcommon0 \
    libnss3 \
    libasound2 \
    fonts-dejavu-core || \
    (sleep 5 && apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    software-properties-common \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libxkbcommon0 \
    libnss3 \
    libasound2 \
    fonts-dejavu-core) \
    && rm -rf /var/lib/apt/lists/*

# Install Chromium (needed for Kaleido)
RUN apt-get update && apt-get install -y \
    chromium \
    chromium-driver \
    fonts-liberation \
    --no-install-recommends && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Fix for Kaleido (expects chromium-browser)
RUN ln -s /usr/bin/chromium /usr/bin/chromium-browser

# Tell Kaleido where Chromium is
ENV CHROME_PATH=/usr/bin/chromium
ENV CHROMIUM_PATH=/usr/bin/chromium-browser

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir kaleido

# Copy application
COPY app.py .
COPY models/ ./models/

# Streamlit config
RUN mkdir -p /root/.streamlit
RUN echo "\
[server]\n\
headless = true\n\
port = 8501\n\
address = 0.0.0.0\n\
\n\
[browser]\n\
gatherUsageStats = false\n\
\n\
[theme]\n\
primaryColor = '#667eea'\n\
backgroundColor = '#ffffff'\n\
secondaryBackgroundColor = '#f0f2f6'\n\
textColor = '#262730'\n\
font = 'sans serif'\n\
" > /root/.streamlit/config.toml

EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

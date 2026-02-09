FROM python:3.12-slim

# Install ffmpeg and streamlink runtime deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Persistent data + clips volumes
RUN mkdir -p /app/data /app/clips

VOLUME ["/app/data", "/app/clips"]

# Default: run the bot. Override with "web" to run the testing GUI.
ENV MODE=bot
EXPOSE 8080

CMD ["sh", "-c", "if [ \"$MODE\" = 'web' ]; then python web/app.py; else python main.py; fi"]

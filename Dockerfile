# Use a lightweight, official Python image
FROM python:3.10-slim

WORKDIR /app

ENV HOST=0.0.0.0 \
    PORT=8000

EXPOSE 8000

# Copy requirements and install them
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir -r /app/requirements.txt

# Copy the rest of the application
COPY . /app

EXPOSE 8000

# Create a non-root user for security (matching your original 1000:1000 requirement)
RUN useradd -m -u 1000 appuser
USER 1000:1000

# Start the server
CMD ["python", "app.py"]
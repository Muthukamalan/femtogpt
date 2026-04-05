FROM python:3.14.3-slim-trixie

WORKDIR /app

# Copy requirements and install dependencies
COPY reqs.txt .
RUN pip install --no-cache-dir -r reqs.txt

# Copy only the python files
COPY main.py .

# Expose the port
EXPOSE 7860

# Run the FastAPI application with uvicorn
CMD ["python","main.py"]
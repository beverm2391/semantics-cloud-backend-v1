# Use the official Python image as the base image
# FROM python:3.9.9-alpine
FROM python:3.10-slim-bullseye

# Set the working directory
WORKDIR /semantic-search-backend-v1

# Copy the requirements file and install the dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose the port that the application will run on
EXPOSE 8000

# Set environment variables from the .env file
ENV $(cat .env | xargs)

# Start the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

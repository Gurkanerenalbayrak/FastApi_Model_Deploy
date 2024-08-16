FROM python:3.9.18-slim-bullseye

# Create a directory for the app
RUN mkdir /app

# Set the working directory
WORKDIR /app

# Copy the requirements file and other files
COPY requirements.txt main.py knn_model.pkl LR_model.pkl /app/

# Upgrade pip and install the dependencies
RUN pip install update pip && pip install -r requirements.txt

# Expose the port
EXPOSE 8000

# Run the application
CMD [ "uvicorn", "main:app", "--host=0.0.0.0", "--port=8000" ]

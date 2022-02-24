FROM  python:3.8-slim-buster
COPY ["Dockerfile","./E:/data science/Projects/Cancer Prediction/Web-App/"]
EXPOSE 5000
WORKDIR E:/data science/Projects/Cancer Prediction/Web-App/
RUN pip freeze > requirements.txt
RUN pip install -r requirements.txt
CMD python app.py

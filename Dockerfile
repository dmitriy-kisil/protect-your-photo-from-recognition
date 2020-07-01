ARG CODE_VERSION="3.8-slim"
ARG PROJECT_DIR="protect-your-photo-from-recognition"
FROM python:${CODE_VERSION}
LABEL mantainer="Dmitriy Kisil <email: logart1995@gmail.com>"
COPY ./requirements.txt ./${PROJECT_DIR}/requirements.txt
WORKDIR /${PROJECT_DIR}
RUN pip3 install --no-cache-dir -r requirements.txt
COPY . /${PROJECT_DIR}
EXPOSE 8050
CMD ["python3", "app.py"]
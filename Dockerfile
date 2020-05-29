ARG CODE_VERSION="3.8-slim"
FROM python:${CODE_VERSION}
LABEL mantainer="Dmitriy Kisil <email: logart1995@gmail.com>"
COPY ./requirements.txt ./protect-your-photo-from-recognition/requirements.txt
WORKDIR /protect-your-photo-from-recognition
RUN pip3 install --no-cache-dir -r requirements.txt
COPY . /protect-your-photo-from-recognition
EXPOSE 8050
CMD ["python3", "app.py"]
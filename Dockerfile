ARG CODE_VERSION="3.8-slim"
FROM python:${CODE_VERSION}
LABEL mantainer="Dmitriy Kisil <email: logart1995@gmail.com>"
ADD . /protect-your-photo-from-recognition
WORKDIR /protect-your-photo-from-recognition
# for opencv
RUN apt-get update && apt-get install -y \
  libgtk2.0-dev
# packages that we need
COPY requirements.txt .
RUN pip3 install -r requirements.txt
COPY . .
EXPOSE 8050
CMD ["python3", "app.py"]
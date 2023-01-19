FROM python:3.11
WORKDIR /
COPY . .
RUN pip install --no-cache-dir .[all]
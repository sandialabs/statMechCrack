FROM python:3.12
WORKDIR /
COPY . .
RUN pip install --no-cache-dir .[all]
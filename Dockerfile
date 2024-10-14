FROM python:3.13
WORKDIR /
COPY . .
RUN pip install --no-cache-dir .[all]
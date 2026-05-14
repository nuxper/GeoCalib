FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libimage-exiftool-perl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml viewer.py ./
COPY templates/ templates/
COPY static/ static/
RUN pip install --no-cache-dir .

EXPOSE 6001

ENTRYPOINT ["python", "viewer.py", "--host", "0.0.0.0", "--no-browser"]
CMD ["/data"]

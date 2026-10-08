FROM python:3.11-slim

WORKDIR /usr/src/app
ENV PATH="/root/.local/bin:${PATH}" \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false

RUN apt-get update
RUN apt-get install -y --no-install-recommends \
    curl \
    libegl1 \
    libgl1 \
    libgles2 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libxcb1 \
    unzip \
    && rm -rf /var/lib/apt/lists/* \
    && curl -sSL https://install.python-poetry.org | python3 -

COPY pyproject.toml poetry.lock ./
RUN poetry install --no-root

COPY models models/
COPY src/ src/
COPY tests/ tests/
COPY dataset.zip .

RUN unzip dataset.zip

CMD ["poetry", "run"]
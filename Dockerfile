FROM python:3.11-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build
COPY pyproject.toml README.md constraints.txt ./
COPY src ./src

RUN python -m pip install --upgrade pip wheel setuptools \
    && python -m pip wheel --constraint constraints.txt --wheel-dir /wheels .

FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN addgroup --system hogfm && adduser --system --ingroup hogfm hogfm

WORKDIR /app
COPY --from=builder /wheels /wheels
COPY constraints.txt ./
RUN python -m pip install --no-cache-dir --constraint constraints.txt /wheels/*.whl \
    && rm -rf /wheels

USER hogfm
EXPOSE 8000

CMD ["uvicorn", "hogfm.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

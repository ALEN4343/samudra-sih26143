# SAMUDRA on Hugging Face Spaces (Docker SDK).
#
# The Space holds only this file and README.md. The code, the shipped cases
# (demo-001, kutch-exercise-001) and the coastline are cloned from GitHub at
# build time, so no large files are uploaded to the Space.
#
# To pick up new commits from GitHub: Space -> Settings -> Factory rebuild.
FROM python:3.12-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*

# Spaces run the container as uid 1000; the app writes dossiers and the audit
# log under artifacts/, so the checkout must belong to that user.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    SAMUDRA_HOME=/investigate
WORKDIR /home/user

ARG REPO=https://github.com/ALEN4343/samudra-sih26143.git
ARG REF=real-isro-data
RUN git clone --depth 1 --branch ${REF} ${REPO} app

WORKDIR /home/user/app
# The lightweight demo set: no torch. The dashboards and shipped cases do not
# need it; /ops -> Satellite reports that no model is installed.
RUN pip install --no-cache-dir --user -r requirements-demo.txt
ENV PYTHONPATH=/home/user/app/src

EXPOSE 7860
CMD ["uvicorn", "samudra.api:app", "--host", "0.0.0.0", "--port", "7860"]

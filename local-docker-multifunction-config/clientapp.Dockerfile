# Ausgangsimage
FROM flwr/clientapp:1.14.0

USER app

# Festlegen des Arbeitsverzeichnisses
WORKDIR /app

# Kopieren der Projektdateien
COPY --chown=app:app pyproject.toml .
COPY --chown=app:app dependencies/analysisbackend /app/analysisbackend
COPY --chown=app:app dependencies/hahn_logger /app/hahn_logger

# Installieren der Abhängigkeiten
RUN sed -i '/.*flwr\[simulation\].*/d' pyproject.toml && \
    pip install --no-cache-dir .

# Installation der lokalen Repositories
RUN pip install -e ./analysisbackend

# Festlegen des EntryPoints
ENTRYPOINT ["flwr-clientapp"]

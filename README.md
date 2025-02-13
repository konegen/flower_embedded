# Run Flower in Docker container local with two use cases

## Install dependencies and project

```bash
pip install -e .
```

## Build and Run Docker Containers

To build and start the Flower containers run:

```bash
docker compose up -d
```

## Run with the Simulation Engine

Use `flwr run` to run a local simulation:

```bash
flwr run .
```

## Stop and Remove Docker Containers

To stop and remove the running Flower containers, use:

```bash
docker compose down
```

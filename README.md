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

## Run the Flowre project in the Docker container 

Use `flwr run` to run a local deployment:

```bash
flwr run . local-deployment --stream
```

## Stop and Remove Docker Containers

To stop and remove the running Flower containers, use:

```bash
docker compose down
```

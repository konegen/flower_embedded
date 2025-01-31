## **1. Swarm-Cluster vorbereiten**

### **Auf dem Server (Manager-Node):**

1. **Swarm initialisieren**:
    
    ```bash
    docker swarm init
    ```
    
    - Der Server wird zum **Manager-Node**.
    - Der Befehl gibt einen **Join-Token** zurück, der verwendet wird, damit die Clients dem Cluster beitreten können.
2. **Join-Token anzeigen (falls verloren)**:
    
    ```bash
    docker swarm join-token worker
    ```
    
    - Dieser Befehl zeigt den Token an, mit dem die Clients dem Swarm-Cluster beitreten können.

---

### **Auf den Clients (Worker-Nodes):**

1. **Swarm beitreten**:
Führe auf jedem Client folgenden Befehl aus:
    
    ```bash
    docker swarm join --token <TOKEN> <MANAGER-IP>:2377
    ```
    
    - **`<TOKEN>`**: Der vom Manager generierte Token.
    - **`<MANAGER-IP>`**: Die IP-Adresse des Servers (Manager-Node).
2. **Verbindung überprüfen**:
Auf dem Server kannst du prüfen, ob die Clients erfolgreich beigetreten sind:
    
    ```bash
    docker node ls
    ```
    
    - Alle Nodes (Manager und Worker) sollten in der Liste angezeigt werden.

---

## **2. Hostnamen auf den Raspberry Pis ändern**

### **Auf jedem Raspberry Pi:**

1. **Aktuellen Hostnamen überprüfen:**
    
    ```bash
    hostname
    ```
    
2. **Hostnamen ändern:**
Führe auf jedem Gerät den folgenden Befehl aus, wobei `<neuer-hostname>` durch den gewünschten Namen ersetzt wird:
    
    ```bash
    sudo hostnamectl set-hostname <neuer-hostname>
    ```
    
    Beispiel:
    
    - Auf dem **Manager**:
        
        ```bash
        sudo hostnamectl set-hostname manager-node
        ```
        
    - Auf dem ersten **Worker**:
        
        ```bash
        sudo hostnamectl set-hostname worker-node-1
        ```
        
3. **Neustart:**
Starte den Raspberry Pi neu, damit die Änderungen wirksam werden:
    
    ```bash
    sudo reboot
    ```
    

---

## **3. Docker Images vorbereiten**

### **Option 1: Images auf jedem Server und Client bauen**

Eine Möglichkeit ist, die Docker Images lokal auf dem Server und den Clients direkt zu bauen.

### **Schritte:**

1. **Dockerfile vorbereiten:**
Es sollte sichergestellt werden, dass die entsprechenden `Dockerfile`Dateien für `clientapp` und `serverapp` auf den entsprechenden Geräten verfügbar sind.
2. **Images bauen**
    
    **Für die Client-App:**
    
    ```bash
    docker build -t flwr_clientapp -f clientapp.Dockerfile .
    ```
    
    **Für die Server-App:**
    
    ```bash
    docker build -t flwr_serverapp -f serverapp.Dockerfile .
    ```
    
3. **Verfügbarkeit überprüfen:**
Nach dem Build-Prozess kann die Verfügbarkeit der Images überprüft werden:
    
    ```bash
    docker images
    ```
    

### **Option 2: Images auf dem Host/Server bauen und über eine Registry verteilen**

Alternativ können die Docker Images zentral auf dem Manager-Node oder einer dedizierten Maschine gebaut und in eine Container-Registry (z. B. Harbor) hochgeladen werden. Die Worker-Nodes können die Images dann von dort beziehen.

### **Schritte:**

1. **Images bauen:**
    
    **Für die Client-App:**
    
    ```bash
    docker build -t harbor.dev-hs.de/${project}/flwr_clientapp -f clientapp.Dockerfile .
    ```
    
    **Für die Server-App:**
    
    ```bash
    docker build -t harbor.dev-hs.de/${project}/flwr_serverapp -f serverapp.Dockerfile .
    ```
    
2. **Bei der Registry anmelden:**
Es wird sichergestellt, dass eine Anmeldung bei der Registry (z. B. Harbor) erfolgt ist. Falls erforderlich, wird folgender Befehl ausgeführt:
    
    ```bash
    docker login harbor.dev-hs.de/harbor
    ```
    
    Benutzername und Passwort werden eingegeben, wenn dazu aufgefordert.
    
3. **Images in die Registry pushen:**
Nach dem Build-Prozess werden die Images in die Registry hochgeladen:
    
    **Für die Client-App:**
    
    ```bash
    docker push harbor.dev-hs.de/${project}/flwr_clientapp
    ```
    
    **Für die Server-App:**
    
    ```bash
    docker push harbor.dev-hs.de/${project}/flwr_serverapp
    ```
    
4. **Images aus der Registry pullen:**
Wenn die Docker Images in der Registry liegen, können diese auf den entsprechenden Geräten heruntergeladen werden:
    
    **Für die Client-App:**
    
    ```bash
    docker pull harbor.dev-hs.de/${project}/flwr_clientapp
    ```
    
    **Für die Server-App:**
    
    ```bash
    docker pull harbor.dev-hs.de/${project}/flwr_serverapp
    ```
    
- **Achtung:** Verschiedene Architekturen von Systemen, auf dem das Image gebaut wird und auf dem der Container ausgebaut wird, bspw. x86_64 (`amd64`) Architektur oder ARM (`arm64` oder `armhf`). Mit Crosscompiler Docker Buildx für Multi-Arch-Support möglich:
    
    ### **Aktiviere Buildx** (falls nicht bereits aktiviert)
    
    ```bash
    docker buildx create --use
    docker buildx inspect --bootstrap
    ```
    
    ### **Image bauen und in die Registry pushen**
    
    **Für arm64 oder armhf:**
    
    ```bash
    docker buildx build --platform linux/arm64 -t harbor.dev-hs.de/hand/flwr_serverapp_iris:latest --push .
    ```
    
    **Für x86_64 (amd64):**
    
    ```bash
    docker buildx build --platform linux/amd64 -t harbor.dev-hs.de/hand/flwr_serverapp_iris:latest --push .
    ```
    
    💡 Beachte: Du musst das Image pushen, da Buildx es nicht lokal speichert.
    

---

## **4. Overlay-Netzwerk erstellen**

Erstelle ein **Overlay-Netzwerk**, das zwischen allen Nodes geteilt wird, dies wird in der compose.yaml definiert:

```bash
docker network create \
  --driver overlay \
  my_overlay_network
```

Dieses Netzwerk ermöglicht die Kommunikation zwischen Containern, unabhängig davon, auf welchem Node sie laufen.

---

## **5. Docker Compose-Stack definieren**

Es ist es auch möglich zu definieren, auf welchem Node (Gerät) welche Container laufen sollen.
Beispiel `docker-compose.yml`:

```yaml
version: '3.8'

services:
  superlink:
    image: flwr/superlink:1.14.0
    command:
      - --insecure
      - --isolation
      - process
    ports:
      - 9091:9091
      - 9092:9092
      - 9093:9093
    networks:
      - flwr-overlay-network
    # Optional: wenn definiert werden soll auf welchem Gerät
    # welcher Container laufen sollen
    deploy:
      placement:
        constraints:
          - node.hostname == manager-node  # Auf dem Manager-Node ausführen

  supernode-1:
    image: flwr/supernode:1.14.0
    command:
      - --insecure
      - --superlink
      - superlink:9092
      - --node-config
      - "partition-id=0 num-partitions=2"
      - --clientappio-api-address
      - 0.0.0.0:9094
      - --isolation
      - process
    ports:
      - 9094:9094
    networks:
      - flwr-overlay-network
    depends_on:
      - superlink
    # Optional: wenn definiert werden soll auf welchem Gerät
    # welcher Container laufen sollen
    deploy:
      placement:
        constraints:
          - node.hostname == worker-node-1  # Auf worker-node-1 ausführen

  serverapp:
    # Wenn image lokal gebaut wurde
    image: flwr_serverapp:0.0.1
    # Wenn image aus der Registry (bspw. Harbor) verwendet werden soll
    image: harbor.dev-hs.de/${project}/flwr_serverapp:0.0.1
    command:                         
      - --insecure
      - --serverappio-api-address
      - superlink:9091
    networks:
      - flwr-overlay-network
    depends_on:
      - superlink
    # Optional wenn definiert werden soll auf welchem Gerät
    # welcher Container laufen sollen
    deploy:
      placement:
        constraints:
          - node.hostname == manager-node  # Auf dem Manager-Node ausführen

  clientapp-1:
    # Wenn image lokal gebaut wurde
    image: flwr_clientapp:0.0.1
    # Wenn image aus der Registry (bspw. Harbor) verwendet werden soll
    image: harbor.dev-hs.de/${project}/flwr_clientapp:0.0.1
    command:                         
      - --insecure
      - --clientappio-api-address
      - supernode-1:9094
    networks:
      - flwr-overlay-network
    depends_on:
      - supernode-1
    # Optional: wenn definiert werden soll auf welchem Gerät
    # welcher Container laufen sollen
    deploy:
      placement:
        constraints:
          - node.hostname == worker-node-1  # Auf worker-node-1 ausführen

networks:
  flwr-overlay-network:
    driver: overlay
```

### **Erläuterung:**

- **`deploy.replicas`**: Gibt die Anzahl der Instanzen für jeden Service an.
- **`placement.constraints`**: Steuert, wo die Container ausgeführt werden:
    - `node.role == worker`: Nur auf Worker-Nodes.
    - `node.role == manager`: Nur auf dem Manager.
- **`networks`**: Verbindet die Container über das Overlay-Netzwerk.

---

## **6. Stack starten**

Starte den Stack auf dem Manager-Node:

```bash
docker stack deploy -c docker-compose.yml my_stack
```

- **`my_stack`**: Der Name des Stacks, unter dem die Services verwaltet werden.
- Swarm verteilt die Container automatisch auf die Nodes basierend auf der `docker-compose.yml`.

---

## **7. Überprüfung des Status**

### **Services anzeigen**:

Auf dem Manager-Node:

```bash
docker service ls
```

- Zeigt alle Services im Stack und deren Status (z. B. Anzahl der Replikate).

### **Nodes überprüfen**:

```bash
docker node ls
```

- Zeigt die Nodes im Cluster und deren Status (z. B. `ACTIVE`).

### **Container auf den Nodes anzeigen**:

1. Auf dem Manager:
    
    ```bash
    docker ps
    ```
    
    Zeigt die auf dem Manager laufenden Container an.
    
2. Auf den Worker-Nodes:
    - Melde dich bei den Clients an und führe `docker ps` aus, um zu sehen, welche Container dort laufen.

---

## 8. Flower Run starten

Starten des Flower Runs auf dem Server oder dem Host-Gerät

Beispielhafter Call:

```bash
flwr . run local-deployment --stream
```

## **9. Logs anzeigen**

Logs eines bestimmten Services auf dem Manager-Node:

```bash
docker service logs flwr_clientapp
```

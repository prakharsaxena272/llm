import chromadb

try:
    port_number = 8000
    print(f"DEBUG: Trying to connect to ChromaDB on port {port_number}")

    client = chromadb.HttpClient(host="localhost", port=port_number)

    # Check if the client is working by sending a heartbeat
    print("✅ ChromaDB client initialized successfully!")
    print("🔄 Heartbeat response:", client.heartbeat())

except Exception as e:
    print("❌ Error initializing ChromaDB client:", e)

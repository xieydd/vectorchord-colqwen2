FROM pgvector/pgvector:0.8.0-pg17

RUN set -eux; \
    apt-get update && apt-get install -y \
    wget; \
    apt-get autoremove -y; \
    apt-get clean -y; \
    rm -rf /var/lib/apt/lists/*

RUN wget https://github.com/tensorchord/VectorChord-bm25/releases/download/0.2.0/postgresql-17-vchord-bm25_0.2.0-1_amd64.deb -O /tmp/vchord-bm25.deb; \
    dpkg -i /tmp/vchord-bm25.deb; \
    rm -rf /tmp/*.deb

RUN wget https://github.com/tensorchord/pg_tokenizer.rs/releases/download/0.1.0/postgresql-17-pg-tokenizer_0.1.0-1_amd64.deb -O /tmp/pg-tokenizer.deb; \
    dpkg -i /tmp/pg-tokenizer.deb;

# RUN wget https://github.com/tensorchord/VectorChord/releases/download/0.2.1/postgresql-17-vchord_0.2.1-1_amd64.deb -O /tmp/vchord.deb; \
#     dpkg -i /tmp/vchord.deb
COPY postgresql-17-vchord_0.0.0-1_amd64.deb /tmp/vchord.deb
RUN dpkg -i /tmp/vchord.deb; \
    rm -rf /tmp/*.deb 

CMD ["postgres", "-c" ,"shared_preload_libraries=vchord.so, pg_tokenizer.so"]

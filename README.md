---
title: Wordify
emoji: 🤗
colorFrom: blue
colorTo: blue
python_version: 3.7
sdk: streamlit
sdk_version: 1.0
app_file: app.py
pinned: false
---


# Run without docker
```bash
streamlit run app.py
```

# Debug in Docker
```bash
# create image (if not already present)
make build

# run container with an interactive shell
make dev

# (from within the contained) start the app normally
streamlit run app.py
```

# Run in Docker
```bash
# create image (if not already present)
make build

# run container and serve the app at localhost:4321
make run

# to stop container
make stop
```
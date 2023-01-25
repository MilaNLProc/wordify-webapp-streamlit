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


# Run locally without docker
```bash
streamlit run app.py
```

# Run locally in Docker
```bash
# create image
make build

# run container and serve the app at localhost:4321
make run

# to stop container
make stop
```
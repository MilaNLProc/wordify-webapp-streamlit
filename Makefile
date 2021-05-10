.PHONY: help build dev integration-test push
.DEFAULT_GOAL := help

# Docker image build info
PROJECT:=wordify
BUILD_TAG?=0.0.1

ALL_IMAGES:=src

help:
# http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html
	@echo "python starter project"
	@echo "====================="
	@echo "Replace % with a directory name (e.g., make build/python-example)"
	@echo
	@grep -E '^[a-zA-Z0-9_%/-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

########################################################
## Local development
########################################################

dev: ARGS?=/bin/bash
dev: DARGS?=-v "${CURDIR}":/var/dev
dev: ## run a foreground container
	docker run -it --rm $(DARGS) $(PROJECT) $(ARGS)


notebook: ARGS?=jupyter lab
notebook: DARGS?=-v "${CURDIR}":/var/dev -p 8888:8888 ##notebook shall be run on http://0.0.0.0:8888 by default. Change to a different port (e.g. 8899) if 8888 is used for example 8899:8888
notebook: ## run a foreground container
	docker run -it --rm $(DARGS) $(PROJECT) $(ARGS) \
		--ip=0.0.0.0 \
		--allow-root \
		--NotebookApp.token="" \
		--NotebookApp.password=""

build: DARGS?=
build: ## build the latest image for a project
	docker build $(DARGS) --build-arg BUILD_TAG=${BUILD_TAG} --rm --force-rm -t $(PROJECT):${BUILD_TAG} .

run:
	docker run -d --name $(PROJECT)-${BUILD_TAG}-container -it --rm -p 8501:8501 $(PROJECT):${BUILD_TAG}
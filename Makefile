# Docker image build info
PROJECT:=wordify
BUILD_TAG?=v2.0
sources = src

########################################################
## Local development
########################################################
dev: ARGS?=/bin/bash
dev: DARGS?=-v "${CURDIR}":/var/dev
dev: ## run a foreground container
	docker run -it --rm -p 8501:8501 $(DARGS) $(PROJECT):${BUILD_TAG} $(ARGS)

build: DARGS?=
build: ## build the latest image for a project
	docker build $(DARGS) --build-arg BUILD_TAG=${BUILD_TAG} --rm --force-rm -t $(PROJECT):${BUILD_TAG} .

########################################################
## Deployment
########################################################
run:
	docker run -d --name $(PROJECT)-${BUILD_TAG}-container -it --rm -p 4321:8501 $(PROJECT):${BUILD_TAG}

stop:
	docker stop $(PROJECT)-${BUILD_TAG}-container

format:
	isort $(sources)
	black $(sources)

lint:
	flake8 $(sources)

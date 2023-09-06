PROJECT_NAME := sota_model

BUILD_TAG := latest

export PROJECT_NAME
export BUILD_TAG

help: ## display all options of the Makefile
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

build: ## build the base docker with a neuralforecast environment
	docker build \
	  -f docker/dockerfiles/Dockerfile \
	  --network host \
	  -t $(PROJECT_NAME):$(BUILD_TAG) .

run: stop ## run the sota_model docker
	docker/scripts/run-docker.sh

enter: ## enter a terminal session within a running sota_model docker
	docker/scripts/enter-docker.sh

stop: ## stop the running sota_model docker
	docker/scripts/stop-docker.sh

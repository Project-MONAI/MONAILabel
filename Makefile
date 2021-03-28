TAG=1.0.0
IMAGE=monai/monai-label

init:
	pip3 install -r requirements.txt

proto:
	cd server/internal/grpc/protos && rm *.py && python3 -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. *.proto

docker: clean
	DOCKER_BUILDKIT=1 docker build --network=host -t $(IMAGE):$(TAG) -f Dockerfile .

test:
	python3 -m unittest discover -s tests/unit -v

dist:
	python3 setup.py sdist bdist_wheel

run:
	cd server && uvicorn main:app --reload

clean:
	@echo "Running clean up...."
	@rm -rf build dist *.egg-info
	@find . -type d -name  "__pycache__" -exec rm -r {} +

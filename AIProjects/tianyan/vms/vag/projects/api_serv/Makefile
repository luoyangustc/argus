
all:
	cd src; go build -v api_lib; go install api_lib; go build -v api_serv; go install api_serv

install: all
	@echo

test:
	cd src; go test ./...

clean:
	cd src; go clean -i ./...



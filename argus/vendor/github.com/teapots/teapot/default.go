package teapot

import (
	"net/http"
)

func defaultNotFound(rw http.ResponseWriter, req *http.Request) {
	// TODO: friendly page in dev mode
	handleStatus(rw, http.StatusNotFound)
}

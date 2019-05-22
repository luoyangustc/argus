package util

import (
	"strings"
)

func CleanHost(host string) string {
	if host == "" {
		return host
	}
	for strings.HasSuffix(host, "/") {
		host = strings.TrimSuffix(host, "/")
	}

	if !strings.HasPrefix(host, "http") {
		// use http by default
		host = "http://" + host
	}
	return host
}

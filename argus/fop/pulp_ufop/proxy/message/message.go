package proxy_message

import (
	"fmt"
	"strings"
)

func CreateError(code int, message string) []byte {
	return []byte(fmt.Sprintf(`{"code":%d,"message":"%s"}`, code,
		strings.Replace(message, `"`, `\"`, -1)))
}

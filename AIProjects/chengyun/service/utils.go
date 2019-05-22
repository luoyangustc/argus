package service

import (
	"strings"
)

func CameraMatchFilters(id, licenceID string) bool {
	// if len(id) < 15 {
	// return false
	// }
	// if id[14:16] != "13" && id[14:16] != "11" {
	// return false
	// }
	if strings.Index(licenceID, "沪") == -1 {
		return false
	}
	return true
}

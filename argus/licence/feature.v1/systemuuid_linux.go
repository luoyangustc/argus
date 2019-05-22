package feature

import (
	"bufio"
	"os"
)

func GetSystemUUID() string {
	f, err := os.Open("/sys/class/dmi/id/product_uuid")
	if err != nil {
		return ""
	}
	defer f.Close()

	line, _, err := bufio.NewReader(f).ReadLine()
	if err != nil {
		return ""
	}
	return string(line)
}

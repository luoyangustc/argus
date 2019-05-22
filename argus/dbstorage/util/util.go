package util

import (
	"crypto/sha1"
	"encoding/hex"
	"os"
	"strings"

	"github.com/pkg/errors"
)

func GetSha1(data []byte) string {
	h := sha1.New()
	h.Write(data)
	cipherStr := h.Sum(nil)
	return hex.EncodeToString(cipherStr)
}

func GetTagAndDesc(name string) (tag, desc string) {
	if name != "" {
		if i := strings.LastIndex(name, "."); i >= 0 {
			name = name[0:i]
		}
		blocks := strings.SplitN(name, "_", 2)
		if len(blocks) == 1 {
			return blocks[0], ""
		}
		return blocks[0], blocks[1]
	}
	return "", ""
}

func Substring(source string, start int, end int) string {
	var r = []rune(source)
	length := len(r)

	if start < 0 || end > length || start > end {
		return ""
	}

	if start == 0 && end == length {
		return source
	}

	return string(r[start:end])
}

func CreatePath(path string) error {
	exist, err := PathExists(path)
	if err != nil {
		return errors.Errorf("get directory [%s] error: %v", path, err)
	}
	if !exist {
		//create folder
		err := os.MkdirAll(path, os.ModePerm)
		if err != nil {
			return errors.Errorf("create directory [%s] failed: %v", path, err)
		}
	}
	return nil
}

func PathExists(path string) (bool, error) {
	_, err := os.Stat(path)
	if err == nil {
		return true, nil
	}
	if os.IsNotExist(err) {
		return false, nil
	}
	return false, err
}

func ArrayContains(s []int, e int) bool {
	for _, a := range s {
		if a == e {
			return true
		}
	}
	return false
}

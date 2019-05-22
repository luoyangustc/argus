package util

import (
	"bufio"
	"bytes"
	"crypto/sha1"
	"encoding/hex"
	"encoding/json"
	"regexp"
)

var (
	regHttpUrl = regexp.MustCompile(`(?i)^(http://|https://)`)
	ImageExts  = []string{".jpeg", ".jpg", ".png", ".gif", ".webp", ".bmp", ".tiff", ".cr2", ".jxr", ".psd", ".ico"}
	VideoExts  = []string{".mp4", ".m4v", ".mkv", ".webm", ".mov", ".avi", ".wmv", ".mpg", ".flv"}
)

func Sha1(data string) string {
	sha1 := sha1.New()
	sha1.Write([]byte(data))
	return hex.EncodeToString(sha1.Sum(nil))
}

func ArrayContains(array []string, item string) bool {
	for _, v := range array {
		if v == item {
			return true
		}
	}
	return false
}

func ByteArrayToLines(bs []byte) []string {
	res := []string{}
	scanner := bufio.NewScanner(bytes.NewReader(bs))
	for scanner.Scan() {
		res = append(res, scanner.Text())
	}
	return res
}

func IsHttpUrl(url string) bool {
	return regHttpUrl.MatchString(url)
}

func UnmarshalInterface(from interface{}, to interface{}) error {
	j, err := json.Marshal(from)
	if err != nil {
		return err
	}
	err = json.Unmarshal(j, to)
	return err
}

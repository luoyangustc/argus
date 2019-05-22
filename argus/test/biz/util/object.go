package util

import (
	"errors"
	"fmt"
	"io/ioutil"
	"net/http"
	"os"
	"path/filepath"

	"qiniu.com/argus/test/configs"
	"qiniu.com/argus/test/lib/auth"
)

func GetTsv(file string) ([]byte, error) {
	// fname := fmt.Sprintf("testdata/%s", file)
	url := auth.GetPrivateUrl(
		"http://"+configs.Configs.Atservingprivatebucketz0.Domain+"/"+file,
		configs.Configs.Atservingprivatebucketz0.User.AK,
		configs.Configs.Atservingprivatebucketz0.User.SK)
	resp, err := http.Get(url)
	fmt.Println(url)
	if err != nil || resp.StatusCode != 200 {
		errStr := "获取tsv文件失败，请手动检查地址:\n" + url + "\n"
		return nil, errors.New(errStr)
	}
	buf, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return buf, err
	}
	err = Store(file, buf)
	return buf, err
	// return resp.Body, err
}

func GetLocal(file string) ([]byte, error) {
	fname := fmt.Sprintf("testdata/%s", file)
	return ioutil.ReadFile(fname)
	// fp, err := os.Open(file)
	// if err != nil {
	// 	errStr := "获取tsv文件失败，请手动检查地址:\n" + file + "\n"
	// 	errStr += err.Error()
	// 	err = errors.New(errStr)
	// 	return nil, err
	// }
	// var f io.ReadCloser = fp
	// return f, err
}

////////////////////////////////////////////////////////////////////////////////

func Store(file string, buf []byte) error {
	fname := fmt.Sprintf("testdata/%s", file)
	if os.Getenv("TEST_STORE") == "" {
		return nil
	}
	if err := os.MkdirAll(filepath.Dir(fname), 0766); err != nil {
		return err
	}
	return ioutil.WriteFile(fname, buf, 0666)
}

func StoreUri(file string, uri string) error {
	fname := fmt.Sprintf("testdata/%s", file)
	if os.Getenv("TEST_STORE") == "" {
		return nil
	}
	if err := os.MkdirAll(filepath.Dir(fname), 0766); err != nil {
		return err
	}
	resp, err := http.Get(uri)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		errStr := "获取uri失败，请手动检查地址:\n" + uri + "\n"
		return errors.New(errStr)
	}
	buf, err1 := ioutil.ReadAll(resp.Body)
	if err1 != nil {
		return err
	}
	return ioutil.WriteFile(fname, buf, 0666)
}

func GetImgBuf(uri string) ([]byte, error) {
	if os.Getenv("TEST_ENV") == "" {
		buf, err := GetLocal(uri)
		if err != nil {
			panic(err)
		}
		return buf, err
	}
	resp, err := http.Get(uri)
	if err != nil {
		return []byte{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		errStr := "获取uri失败，请手动检查地址:\n" + uri + "\n"
		return []byte{}, errors.New(errStr)
	}
	return ioutil.ReadAll(resp.Body)
}

// GetHTTP ...
func GetHTTP(uri string) ([]byte, error) {
	// s := qnhttp.New()
	// return s.Get(uri, nil, nil, nil)
	resp, err := http.Get(uri)
	if err != nil {
		return []byte{}, err
	}
	defer resp.Body.Close()
	body, err := ioutil.ReadAll(resp.Body)
	fmt.Println(string(body))

	return body, err
}

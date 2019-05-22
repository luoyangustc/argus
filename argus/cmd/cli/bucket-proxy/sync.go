package main

import (
	"bufio"
	"context"
	"crypto/hmac"
	"crypto/sha1"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"strings"
	"time"

	"qiniupkg.com/api.v7/kodo"
)

type ConfigSRC struct {
	User struct {
		AK string `json:"ak"`
		SK string `json:"sk"`
	} `json:"user"`
}

type ConfigDest struct {
	UpHost string `json:"up_host"`
	User   struct {
		AK string `json:"ak"`
		SK string `json:"sk"`
	} `json:"user"`
}

func syncMain(args []string) {
	if len(args) < 3 {
		fmt.Println("main src_config dest_config files_list")
		fmt.Println("{\"user\": {\"ak\":\"\", \"sk\":\"\"}}")
		fmt.Println("{\"up_host\":\"\", \"user\": {\"ak\":\"\", \"sk\":\"\"}}")
		fmt.Println("src_domain	src_key	dest_bucket	dest_key")
		os.Exit(1)
	}

	var (
		srcConfig  = args[0]
		destConfig = args[1]
		filesList  = args[2]

		src  ConfigSRC
		dest ConfigDest
	)
	{
		bs, err := ioutil.ReadFile(srcConfig)
		if err != nil {
			log.Fatalf("%v", err)
		}
		if err = json.Unmarshal(bs, &src); err != nil {
			log.Fatalf("%v", err)
		}
	}
	{
		bs, err := ioutil.ReadFile(destConfig)
		if err != nil {
			log.Fatalf("%v", err)
		}
		if err = json.Unmarshal(bs, &dest); err != nil {
			log.Fatalf("%v", err)
		}
	}

	file, err := os.Open(filesList)
	if err != nil {
		log.Fatalf("%v", err)
	}
	defer file.Close()

	if err = sync(src, dest, file); err != nil {
		log.Fatalf("%v", err)
	}
}

/*
domain	key	bucket	key
*/
func sync(src ConfigSRC, dest ConfigDest, reader io.Reader) error {

	var (
		cli = kodo.NewWithoutZone(
			&kodo.Config{
				UpHosts:   []string{dest.UpHost},
				AccessKey: dest.User.AK,
				SecretKey: dest.User.SK,
			})

		buckets   = make(map[string]kodo.Bucket)
		getBucket = func(name string) (kodo.Bucket, error) {
			if bucket, ok := buckets[name]; ok {
				return bucket, nil
			}
			bucket, err := cli.BucketWithSafe(name)
			if err != nil {
				return kodo.Bucket{}, err
			}
			buckets[name] = bucket
			return bucket, nil
		}
	)

	scanner := bufio.NewScanner(reader)
	for scanner.Scan() {
		strs := strings.Split(scanner.Text(), "\t")
		if len(strs) < 4 {
			return errors.New("bad line")
		}

		var (
			srcDomain  = strs[0]
			srcKey     = strs[1]
			destBucket = strs[2]
			destKey    = strs[3]
		)

		bucket, err := getBucket(destBucket)
		if err != nil {
			return err
		}

		err = func() error {

			var urlS string
			{
				if strings.HasPrefix(srcKey, "/") {
					urlS = fmt.Sprintf("http://%s%s", srcDomain, srcKey)
				} else {
					urlS = fmt.Sprintf("http://%s/%s", srcDomain, srcKey)
				}
				_url, _ := url.Parse(urlS)
				query := _url.Query()
				query.Del("token")
				query.Set("e", strconv.FormatInt(time.Now().Add(time.Hour).Unix(), 10))
				urlS = fmt.Sprintf("http://%s%s?%s", _url.Host, _url.Path, query.Encode())
				h := hmac.New(sha1.New, []byte(src.User.SK))
				h.Write([]byte(urlS))
				sign := base64.URLEncoding.EncodeToString(h.Sum(nil))
				urlS += "&token=" + fmt.Sprintf("%s:%s", src.User.AK, sign)
			}
			resp, err := http.DefaultClient.Get(urlS)
			if err != nil {
				return err
			}
			defer resp.Body.Close()

			return bucket.Put(context.Background(), nil, destKey,
				resp.Body, resp.ContentLength, nil)
		}()
		if err != nil {
			return err
		}

	}
	return scanner.Err()
}

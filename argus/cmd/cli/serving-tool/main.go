package main

import (
	"bufio"
	"encoding/base64"
	"flag"
	"fmt"
	"net/http"
	"os"

	"github.com/qiniu/log.v1"
	"qiniu.com/auth/qiniumac.v1"
)

func init() {
	if os.Getenv("DEBUG") != "" {
		log.SetOutputLevel(0)
	} else {
		log.SetFlags(log.Llevel | log.Lshortfile | log.Ldate)
	}
}

func main() {

	flag.Parse()

	switch args := flag.Args(); args[0] {
	case "etcd":
		etcdMain(args[1:])
	case "deploy":
		deployMain(args[1:])
	case "auth":
		req, err := http.ReadRequest(bufio.NewReader(os.Stdin))
		if err != nil {
			log.Fatalf("parse request failed. %v", err)
		}
		sign, err := qiniumac.SignRequest([]byte(args[2]), req)
		if err != nil {
			log.Fatalf("sign request failed. %v", err)
		}
		fmt.Printf(
			">> Authorization: Qiniu %s:%s",
			args[1], base64.URLEncoding.EncodeToString(sign),
		)
	}

}

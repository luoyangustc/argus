package shell

import (
	"os"
	"strconv"

	"qiniu.com/argus/tuso/client/tuso_hub"
	"qiniu.com/argus/tuso/proto"
)

var mockUID uint32
var host = ""

func init() {
	switch os.Getenv("ENV") {
	case "LOCAL", "":
		mockUID = 1810652781
		host = "http://localhost:9204"
	case "CS":
		host = "http://ava-tuso-hub.cs.cg.dora-internal.qiniu.io:5001"
	case "PROD":
		host = "http://ava-tuso-hub.xs.cg.dora-internal.qiniu.io:5001"
		mockUID = 1380585377
	default:
		xl.Error("ENV environment variable not set")
	}
	if uid := os.Getenv("MOCK_UID"); uid != "" {
		n, err := strconv.Atoi(uid)
		if err != nil {
			xl.Panicln(err)
		}
		mockUID = uint32(n)
	}

	xl.Info("use env", host, mockUID)
}

func getClient() (proto.UserApi, error) {
	return tuso_hub.NewHack(mockUID, host, 0), nil
}

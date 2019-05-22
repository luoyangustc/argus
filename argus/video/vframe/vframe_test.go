package vframe

// import (
// 	"context"
// 	"net/http"
// 	"os"
// 	"os/exec"
// 	"path"
// 	"strconv"
// 	"testing"
// 	"time"

// 	"github.com/stretchr/testify/assert"

// 	"github.com/qiniu/http/restrpc.v1"
// 	"github.com/qiniu/http/servestk.v1"
// 	xlog "github.com/qiniu/xlog.v1"

// 	STS "qiniu.com/argus/sts/client"
// )

// func TestVframeProxy(t *testing.T) {

// 	var proxy = NewSTSProxy(
// 		"http://127.0.0.1:8080/uri",
// 		STS.NewClient("127.0.0.1:5555", func() string { return xlog.GenReqId() }, nil))
// 	// var proxy = NewURIProxy("http://127.0.0.1:8080/uri")
// 	mux := servestk.New(restrpc.NewServeMux())
// 	{
// 		router := &restrpc.Router{
// 			PatternPrefix: "",
// 			Factory:       restrpc.Factory,
// 			Mux:           mux,
// 		}
// 		router.Register(proxy)
// 	}
// 	go func() { http.ListenAndServe("0.0.0.0:8080", mux) }()
// 	time.Sleep(time.Second * 3)

// 	dir := path.Join(os.TempDir(), strconv.FormatInt(time.Now().UnixNano(), 10))
// 	os.MkdirAll(dir, 0777)

// 	req := VframeRequest{}
// 	req.Data.URI = proxy.URI("http://pili-media.live.panda.tv/psegments/z1.live_panda.4baf113e30e65f6e98d0f853d21bb8a7/1527698820-1527712055.flv")
// 	req.Params.Mode = 1
// 	args, _ := _GenLiveCmdWithFFmpeg(context.Background(), req, dir)
// 	cmd := exec.Command(args[0], args[1:]...)
// 	err := cmd.Start()
// 	assert.NoError(t, err)
// 	err = cmd.Wait()
// 	assert.NoError(t, err)
// }

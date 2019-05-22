package main

import (
	"context"
	"flag"
	"log"
	"os"
	"path"
	"runtime"

	"github.com/qiniu/xlog.v1"
	cconf "qbox.us/cc/config"

	"qiniu.com/argus/atserving/model"
	"qiniu.com/argus/serving_eval/embedEval"
)

func main() {

	runtime.GOMAXPROCS(runtime.NumCPU())

	var (
		xl     = xlog.NewWith("main")
		ctx    = xlog.NewContext(context.Background(), xl)
		config = struct {
			Eval embedEval.Config `json:"eval"`
		}{}
	)

	{
		cconf.Init("f", "serving-eval", "servering-eval.conf")
		if err := cconf.Load(&config); err != nil {
			xl.Fatal("Failed to load configure file!")
		}
	}

	worker, err := embedEval.NewWorker(ctx, config.Eval)
	if err != nil {
		xl.Fatalf("new woeker failed. %v", err)
	}

	{
		dirname := flag.Arg(0)
		xl.Infof(".... %s", dirname)
		dir, err := os.Open(dirname)
		if err != nil {
			log.Fatalf("open dir failed. %s %v", dirname, err)
		}
		defer dir.Close()

		names, err := dir.Readdirnames(0)
		if err != nil {
			log.Fatalf("read dir failed. %s %v", dirname, err)
		}
		for _, name := range names {
			xl.Infof("file: %s", path.Join(dirname, name))
			resp, err := func(name string) (model.EvalResponse, error) {
				ctx := xlog.NewContext(context.Background(), xlog.NewDummy())
				resps, err := worker.Eval(ctx, []model.EvalRequest{
					model.EvalRequest{
						Data: model.Resource{
							URI: "file://" + model.STRING(path.Join(dirname, name)),
						},
					},
				})
				if err != nil {
					return model.EvalResponse{}, err
				}
				return resps[0], nil
			}(name)
			if err != nil {
				log.Fatalf("eval failed. %s %v", name, err)
			}
			log.Printf("eval done. %s %#v\n", name, resp)
		}
	}
}

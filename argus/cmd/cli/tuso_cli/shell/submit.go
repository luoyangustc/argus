package shell

import (
	"context"
	"os"
	"strings"
	"time"

	"github.com/k0kubun/pp"

	"github.com/spf13/cobra"
	"qiniu.com/argus/tuso/proto"
)

var submitJob = &cobra.Command{
	Use: "submit <Hub> <Key/Url>...",
	RunE: func(cmd *cobra.Command, args []string) error {
		if len(args) < 2 {
			return cmd.Help()
		}
		hub := args[0]

		imgs := make([]proto.ImageKeyUrl, 0)
		for _, v := range args[1:] {
			img := proto.ImageKeyUrl{Key: v}
			{
				key := v
				if strings.Contains(key, "http://") || strings.Contains(key, "https://") {
					img.Url = v
					img.Key = ""
				}
			}
			imgs = append(imgs, img)
		}

		cl, err := getClient()
		if err != nil {
			return err
		}
		req := &proto.PostSearchJobReq{
			Hub:    hub,
			Images: imgs,
			Kind:   0,
			TopN:   10,
		}
		pp.Println("submit job", req)
		ctx := context.Background()
		resp, err := cl.PostSearchJob(ctx, req, nil)
		if err != nil {
			return err
		}
		start := time.Now()
		xl.Infof("submit job success, id: %v", resp.JobID)
		os.Stdout.WriteString("searching .")
		var r *proto.GetSearchJobResp
		for {
			var err error
			r, err = cl.GetSearchJob(ctx, &proto.GetSearchJobReq{JobID: resp.JobID}, nil)
			if err != nil {
				xl.Error(err)
				return err
			}
			if r.Status == "WAITING" || r.Status == "DOING" {
				os.Stdout.WriteString(".")
				time.Sleep(time.Second)
				continue
			}
			break
		}
		xl.Info("use time", time.Since(start).Seconds())
		xl.Info(pp.Sprint(r))
		return nil
	},
}

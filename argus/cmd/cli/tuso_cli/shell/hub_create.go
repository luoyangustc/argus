package shell

import (
	"context"

	"github.com/spf13/cobra"
	"qiniu.com/argus/tuso/proto"
)

var hubCreateCmd = &cobra.Command{
	Use: "create <Hub> <Bucket> <Prefix>",
	RunE: func(cmd *cobra.Command, args []string) error {
		if len(args) != 2 && len(args) != 3 {
			return cmd.Help()
		}
		cl, err := getClient()
		if err != nil {
			return err
		}

		req := proto.PostHubReq{
			Name:   args[0],
			Bucket: args[1],
		}
		if len(args) == 3 {
			req.Prefix = args[2]
		}
		err = cl.PostHub(context.Background(), &req, nil)
		if err != nil {
			return err
		}
		xl.Infof("create hub %v success", args[0])
		return nil
	},
}

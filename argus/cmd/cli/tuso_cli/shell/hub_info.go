package shell

import (
	"context"
	"fmt"

	"github.com/pkg/errors"

	"github.com/spf13/cobra"
	"qiniu.com/argus/tuso/proto"
)

var hubInfoCmd = &cobra.Command{
	Use: "info <Hub>",
	RunE: func(cmd *cobra.Command, args []string) error {
		cl, err := getClient()
		if err != nil {
			return err
		}
		if len(args) != 1 {
			return cmd.Help()
		}
		resp, err := cl.GetHub(context.Background(), &proto.GetHubReq{Hub: args[0]}, nil)
		if err != nil {
			return errors.Wrap(err, "cl.GetHub")
		}
		fmt.Print(renderIdent(`
HubName:	{{.HubName}}
Bucket:	{{.Bucket}}
Prefix:	{{.Prefix}}
ImageNum:	{{.Stat.ImageNum}}
				`, resp))
		return nil
	},
}

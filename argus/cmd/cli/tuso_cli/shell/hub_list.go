package shell

import (
	"context"
	"fmt"

	"github.com/pkg/errors"

	"github.com/spf13/cobra"
	"qiniu.com/argus/tuso/proto"
)

var hubListCmd = &cobra.Command{
	Use: "list",
	RunE: func(cmd *cobra.Command, args []string) error {
		cl, err := getClient()
		if err != nil {
			return err
		}
		resp, err := cl.GetHubs(context.Background(), &proto.GetHubsReq{}, nil)
		if err != nil {
			return errors.Wrap(err, "cl.GetHubs")
		}
		fmt.Print(renderIdent(`
HubName	Bucket	Prefix
{{range .Hubs }}{{.HubName}}	{{.Bucket}}	{{.Prefix}}
{{end}}
				`, resp))
		return nil
	},
}

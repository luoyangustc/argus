package main

import (
	"context"
	"strconv"

	xlog "github.com/qiniu/xlog.v1"
	"github.com/spf13/cobra"
	"qiniupkg.com/api.v7/kodo"
)

func BucketCmd() *cobra.Command {

	var bucketCmd = &cobra.Command{Use: "bucket", Args: cobra.MinimumNArgs(1)}

	var statCmd = &cobra.Command{
		Use:  "stat [ UID ] [ BUCKET ] [ KEY ]",
		Args: cobra.MinimumNArgs(3),
		RunE: func(cmd *cobra.Command, args []string) error {

			ctx := context.Background()
			xl := xlog.FromContextSafe(ctx)

			uid64, err := strconv.ParseUint(args[0], 10, 64)
			if err != nil {
				xl.Errorf("UID Parse Err, %+v", err)
				return err
			}
			uid32 := uint32(uid64)

			ak, sk, err := Aksk(&AdminConfig.Qconf, uid32)

			kdcfg := AdminConfig.Kodo
			kdcfg.AccessKey = ak
			kdcfg.SecretKey = sk

			cli := kodo.New(0, &kdcfg)
			bCli, _ := cli.BucketWithSafe(args[1])
			entry, err := bCli.Stat(ctx, args[2])

			xl.Info("================")
			xl.Infof("Entry Stat, %+v, %+v", entry, err)

			return nil
		},
	}

	bucketCmd.AddCommand(statCmd)
	return bucketCmd
}

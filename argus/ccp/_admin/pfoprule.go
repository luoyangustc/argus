package main

import (
	"context"
	"strconv"

	xlog "github.com/qiniu/xlog.v1"
	"github.com/spf13/cobra"
)

// 查询PFOP规则
func PfopRuleCmd() *cobra.Command {

	var pfopRuleCmd = &cobra.Command{Use: "pfoprule", Args: cobra.MinimumNArgs(1)}

	var pRGetCmd = &cobra.Command{
		Use:  "get [ UID ] [ BUCKET ]",
		Args: cobra.MinimumNArgs(2),
		RunE: func(cmd *cobra.Command, args []string) error {

			ctx := context.Background()
			xl := xlog.FromContextSafe(ctx)

			uid64, err := strconv.ParseUint(args[0], 10, 64)
			if err != nil {
				xl.Errorf("UID Parse Err, %+v", err)
				return err
			}
			uid32 := uint32(uid64)

			var prinfo interface{}
			InnerPostByAkSk(ctx, &AdminConfig.Qconf, uid32,
				"http://uc.qbox.me/pfopRules/get",
				struct {
					Bucket string `json:"bucket"`
				}{
					Bucket: args[1],
				}, &prinfo)

			xl.Info("================")
			xl.Info(JsonStr(prinfo))
			return nil
		},
	}

	var pRDeleteCmd = &cobra.Command{
		Use:  "delete [ UID ] [ BUCKET ] [ PFOP_NAME ]",
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

			CountdownWithTips(10, "Attention: PfopRule will be deleted")

			var prdelret interface{}
			InnerPostByAkSk(ctx, &AdminConfig.Qconf, uid32,
				"http://uc.qbox.me/pfopRules/delete",
				struct {
					Bucket string `json:"bucket"`
					Name   string `json:"name"`
				}{
					Bucket: args[1],
					Name:   args[2],
				}, &prdelret)

			xl.Info("================")
			xl.Info(JsonStr(prdelret))
			return nil
		},
	}

	pfopRuleCmd.AddCommand(pRGetCmd)
	pfopRuleCmd.AddCommand(pRDeleteCmd)
	return pfopRuleCmd
}

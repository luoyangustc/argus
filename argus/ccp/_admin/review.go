package main

import (
	"context"
	"fmt"
	"strconv"

	xlog "github.com/qiniu/xlog.v1"
	"github.com/spf13/cobra"
	"qbox.us/errors"
)

func CcpReviewCmd() *cobra.Command {

	var ccpReviewCmd = &cobra.Command{Use: "ccpreview", Args: cobra.MinimumNArgs(1)}

	var resetCppReviewSetCmd = &cobra.Command{
		Use:  "reset [ UID ] [ RULE_ID ] [ RESET_TYPE (soft|hard|remove)]",
		Args: cobra.MinimumNArgs(3),
		RunE: func(cmd *cobra.Command, args []string) error {

			ctx := context.Background()
			xl := xlog.FromContextSafe(ctx)

			uid, err := strconv.ParseUint(args[0], 10, 64)
			if err != nil {
				xl.Errorf("UID Parse Err, %+v", err)
				return err
			}

			if args[2] != "soft" && args[2] != "hard" && args[2] != "remove" {
				return errors.New("Invalid reset_type")
			}

			url := fmt.Sprintf("http://argus-ccp-review.xs.cg.dora-internal.qiniu.io:5001/v1/sets/%s/reset", args[1])

			var ret interface{}
			err = InnerPost(ctx, uint32(uid), url, struct {
				Type       string `json:"type"`
				JobType    string `json:"job_type"`
				SourceType string `json:"source_type"`
			}{
				JobType:    "BATCH",
				SourceType: "KODO",
				Type:       args[2],
			}, &ret)
			if err != nil {
				xl.Errorf("Reset Set Failed: %+v", err)
				return err
			}

			xl.Info("================")
			xl.Info("Reset Rule Success")
			return nil
		},
	}

	ccpReviewCmd.AddCommand(resetCppReviewSetCmd)

	return ccpReviewCmd
}

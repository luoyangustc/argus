package main

import (
	"context"
	"encoding/json"
	"fmt"
	"strconv"
	"time"

	mgoutil "github.com/qiniu/db/mgoutil.v3"
	xlog "github.com/qiniu/xlog.v1"
	"github.com/spf13/cobra"
	"gopkg.in/mgo.v2/bson"
	"qbox.us/errors"
	"qiniu.com/argus/ccp/manager/proto"
)

// 查询PFOP规则
func CcpRuleCmd() *cobra.Command {

	var ccpRuleCmd = &cobra.Command{Use: "ccprule", Args: cobra.MinimumNArgs(1)}

	var cRGetCmd = &cobra.Command{
		Use:  "get [ UID ]",
		Args: cobra.MinimumNArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {

			ctx := context.Background()
			xl := xlog.FromContextSafe(ctx)

			uid64, err := strconv.ParseUint(args[0], 10, 64)
			if err != nil {
				xl.Errorf("UID Parse Err, %+v", err)
				return err
			}
			uid32 := uint32(uid64)

			var rls []interface{}
			InnerGet(ctx, uid32,
				"http://argus-ccp.xs.cg.dora-internal.qiniu.io:5001/v1/rules",
				nil, &rls)

			xl.Info("================")
			for _, rl := range rls {
				xl.Info(JsonStr(rl))
			}
			return nil
		},
	}

	var cRSetCmd = &cobra.Command{
		Use:  "set [ UID ] [ RULE_JSON_STR ]",
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

			rl := proto.Rule{}
			err = json.Unmarshal([]byte(args[1]), &rl)
			if err != nil {
				xl.Errorf("RULE_JSON_STR Unmarshal Err, %+v", err)
				return err
			}
			rl.RuleID = fmt.Sprintf("shell_set_%s", time.Now().Format("20060102150405"))

			xl.Info(JsonStr(rl))
			CountdownWithTips(10, "Attention: Rule will be set")
			err = InnerPost(ctx, uid32,
				"http://argus-ccp.xs.cg.dora-internal.qiniu.io:5001/v1/rules",
				rl, nil)

			if err != nil {
				xl.Errorf("Set Rule Failed: %+v", err)
				return err
			}

			xl.Info("================")
			xl.Info("Set Rule Success")
			return nil
		},
	}

	var cRResetCmd = &cobra.Command{
		Use:  "reset [ UID ] [ RULE_ID ]",
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

			rl := proto.Rule{}
			err = InnerGet(ctx, uid32,
				fmt.Sprintf("http://argus-ccp.xs.cg.dora-internal.qiniu.io:5001/v1/rules/%s", args[1]),
				nil, &rl)

			if err != nil {
				xl.Errorf("Get Rule Failed: %+v", err)
				return err
			}

			rl.RuleID = fmt.Sprintf("%s_%s_reset", rl.RuleID[:8], time.Now().Format("20060102150405"))

			xl.Info(JsonStr(rl))
			CountdownWithTips(10, "Attention: Rule will be set")
			err = InnerPost(ctx, uid32,
				"http://argus-ccp.xs.cg.dora-internal.qiniu.io:5001/v1/rules",
				rl, nil)

			if err != nil {
				xl.Errorf("Reset Rule Failed: %+v", err)
				return err
			}

			xl.Info("================")
			xl.Info("Reset Rule Success")
			return nil
		},
	}

	var cRCountCmd = &cobra.Command{
		Use:  "count",
		Args: cobra.MinimumNArgs(0),
		RunE: func(cmd *cobra.Command, args []string) error {

			ctx := context.Background()
			xl := xlog.FromContextSafe(ctx)

			var (
				colls struct {
					Rules   mgoutil.Collection `coll:"rules"`
					KodoSrc mgoutil.Collection `coll:"kodosrc"`
				}
			)

			{
				sess, err := mgoutil.Open(&colls, &AdminConfig.CcpM.Mgo)

				if err != nil {
					xl.Errorf("Open Mongo Failed: %+v", errors.Detail(err))
					return err
				}
				sess.SetPoolLimit(100)
				defer sess.Close()
			}

			{
				coll := colls.Rules.CopySession()
				defer coll.CloseSession()

				kodoColl := colls.KodoSrc.CopySession()
				defer kodoColl.CloseSession()

				query := bson.M{
					"status": proto.RULE_STATUS_ON,
					"type":   proto.TYPE_STREAM,
				}

				num, err := coll.Find(query).Count()
				if err != nil {
					xl.Errorf("Find Err: %+v", err)
					return err
				}

				query2 := bson.M{
					"status": proto.RULE_STATUS_ON,
					"type":   proto.TYPE_BATCH,
				}

				num2, err := coll.Find(query2).Count()
				if err != nil {
					xl.Errorf("Find Err: %+v", err)
					return err
				}

				xl.Info("================")
				xl.Infof("%d rules(STATUS_ON + STREAM)", num)
				xl.Infof("%d rules(STATUS_ON + BATCH)", num2)

			}
			return nil
		},
	}

	var cRPushBjobCmd = &cobra.Command{
		Use:  "pushbjob [ UID ] [ RULE_ID ] [ KEY ]",
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

			bjobCBUrl := fmt.Sprintf("http://argus-ccp.xs.cg.dora-internal.qiniu.io:5001/v1/msg/bjob/%d/%s",
				uid32, args[1])

			CountdownWithTips(10, "Attention: Bjob Result will be posted")

			var ret interface{}
			err = InnerPost(ctx, uid32, bjobCBUrl, struct {
				Keys []string `json:"keys"`
			}{
				Keys: []string{
					args[2],
				},
			}, &ret)

			xl.Info("================")
			xl.Infof("Push Bjob Result, %+v, %+v", ret, err)
			return nil
		},
	}

	ccpRuleCmd.AddCommand(cRGetCmd)
	ccpRuleCmd.AddCommand(cRResetCmd)
	ccpRuleCmd.AddCommand(cRSetCmd)
	ccpRuleCmd.AddCommand(cRCountCmd)
	ccpRuleCmd.AddCommand(cRPushBjobCmd)
	return ccpRuleCmd
}

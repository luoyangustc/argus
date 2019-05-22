package service

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"strconv"

	restrpc "github.com/qiniu/http/restrpc.v1"
	xlog "github.com/qiniu/xlog.v1"
	"qiniu.com/argus/ccp/manager"
	"qiniu.com/argus/ccp/manager/client"
	"qiniu.com/argus/ccp/manager/convert"
	"qiniu.com/argus/ccp/manager/proto"
)

// 各个模块统一把结果Push到该接口
type MsgService interface {
	PostMsgPfop___(ctx context.Context,
		req *struct {
			CmdArgs []string // UID/RuleID/MimeType
			ReqBody io.Reader
		},
		env *restrpc.Env,
	) error

	PostMsgBjob__(ctx context.Context,
		req *struct {
			CmdArgs []string // UID/RuleID
			ReqBody io.Reader
		},
		env *restrpc.Env,
	) error

	PostMsgManualStream__(ctx context.Context,
		req *struct {
			CmdArgs []string // UID/RuleID
			ReqBody io.Reader
		},
		env *restrpc.Env,
	) error

	PostMsgManualBatch__(ctx context.Context,
		req *struct {
			CmdArgs []string // UID/RuleID
			ReqBody io.Reader
		},
		env *restrpc.Env,
	) error

	PostMsgReview__(ctx context.Context,
		req *struct {
			CmdArgs []string // UID/RuleID
			ReqBody io.Reader
		},
		env *restrpc.Env,
	) error
}

func NewMsgService(rules manager.Rules,
	innerCfg client.InnerConfig,
	manualJobs client.ManualJobs,
	reviewCli client.ReviewClient,
	notifyCallback client.NotifyCallback,
	saveBack client.SaveBack) MsgService {
	return &_MsgService{
		Rules:          rules,
		InnerConfig:    innerCfg,
		ManualJobs:     manualJobs,
		ReviewClient:   reviewCli,
		NotifyCallback: notifyCallback,
		SaveBack:       saveBack,
	}
}

type _MsgService struct {
	manager.Rules
	client.InnerConfig
	client.ManualJobs
	client.ReviewClient
	client.NotifyCallback
	client.SaveBack
}

func (m *_MsgService) PostMsgPfop___(ctx context.Context,
	req *struct {
		CmdArgs []string // UID/RuleID/MimeType
		ReqBody io.Reader
	},
	env *restrpc.Env,
) error {
	xl := xlog.FromContextSafe(ctx)
	bs, _ := ioutil.ReadAll(req.ReqBody)
	xl.Infof("PostMsgPfop___.msg, %+v, %s", req.CmdArgs, string(bs))

	uid64, err := strconv.ParseUint(req.CmdArgs[0], 10, 32)
	if err != nil {
		xl.Errorf("conv uid failed, %+v", err)
		return err
	}
	uid := uint32(uid64)
	ruleID := req.CmdArgs[1]
	mimeType := req.CmdArgs[2]

	rl, err := m.Rules.QueryByRuleID(ctx, uid, ruleID)
	if err != nil {
		xl.Errorf("QueryByRuleID %s, err: %+v", ruleID, err)
		return err
	}

	go func() {
		ctx := context.Background()
		xl := xlog.FromContextSafe(ctx)
		if mimeType == "video" {
			err := m.PfopVideo(ctx, rl, bs)
			if err != nil {
				xl.Errorf("%+v", err)
				return
			}
		} else { // 默认为"image"
			err := m.PfopImage(ctx, rl, bs)
			if err != nil {
				xl.Errorf("%+v", err)
				return
			}
		}
	}()

	return nil
}

func (m *_MsgService) PostMsgBjob__(ctx context.Context,
	req *struct {
		CmdArgs []string // UID/RuleID
		ReqBody io.Reader
	},
	env *restrpc.Env,
) error {
	xl := xlog.FromContextSafe(ctx)
	bs, _ := ioutil.ReadAll(req.ReqBody)
	xl.Infof("PostMsgBjob__.msg, %+v, %s", req.CmdArgs, string(bs))

	uid64, err := strconv.ParseUint(req.CmdArgs[0], 10, 32)
	if err != nil {
		xl.Errorf("conv uid failed, %+v", err)
		return err
	}
	uid := uint32(uid64)
	ruleID := req.CmdArgs[1]

	var result = struct {
		Error string   `json:"error"`
		Keys  []string `json:"keys"`
	}{}
	_ = json.Unmarshal(bs, &result)
	xl.Infof("result %+v", result)
	if result.Error != "" {
		xl.Errorf("RuleInfo: %+v, err: %s", req.CmdArgs, result.Error)
		err := m.Rules.Del(ctx, uid, ruleID) // End Batch Job
		xl.Infof("Rules.Del, %s, %+v", ruleID, err)
	}

	rl, err := m.Rules.QueryByRuleID(ctx, uid, ruleID)
	if err != nil {
		xl.Errorf("QueryByRuleID %s, err: %+v", ruleID, err)
		return err
	}

	go func() {
		ctx := context.Background()
		xl := xlog.FromContextSafe(ctx)

		if rl.Manual.IsOn {
			inSaver := m.InnerConfig.GetInnerSaver(ctx, "")
			err := m.ManualJobs.PushItems(ctx, rl, &client.MJBatchEntries{
				UID:    inSaver.UID,
				Bucket: inSaver.Bucket,
				Keys:   result.Keys,
			})
			if err != nil {
				xl.Errorf("ManualJobs.PushItems %s, %+v, err: %+v",
					ruleID, result.Keys, err)
			}
		} else {
			// Save Back
			err := m.SaveBack.Save(ctx, rl, result.Keys)
			xl.Infof("SaveBack, %s, %+v", ruleID, err)
			err = m.Rules.Del(ctx, uid, ruleID) // End Batch Job
			xl.Infof("Rules.Del, %s, %+v", ruleID, err)
			// TODO:Notify

			if rl.Review.IsOn {
				inUID, ak, sk, bucket, domain := m.SaveBack.GetInnerBucketInfo(ctx)
				kodo := m.SaveBack.GetKodoInfo(ctx)
				entryKeys := convert.ConvBjob2Review(ctx, rl, kodo,
					ak, sk, bucket, domain, result.Keys)
				err := m.ReviewClient.PushItems(ctx, rl, &client.BatchEntries{
					UID:    inUID,
					Bucket: bucket,
					Keys:   entryKeys,
				})
				if err != nil {
					xl.Errorf("ReviewClient.PushItems %s, %+v, err: %+v",
						ruleID, entryKeys, err)
				}
			}
		}
	}()

	return nil
}

func (m *_MsgService) PostMsgManualStream__(ctx context.Context,
	req *struct {
		CmdArgs []string // UID/RuleID
		ReqBody io.Reader
	},
	env *restrpc.Env,
) error {
	xl := xlog.FromContextSafe(ctx)
	bs, _ := ioutil.ReadAll(req.ReqBody)
	xl.Infof("PostMsgManualStream__.msg, %+v, %s", req.CmdArgs, string(bs))

	// TODO: 此处分发给Review及结果通知

	return nil
}

func (m *_MsgService) PostMsgManualBatch__(ctx context.Context,
	req *struct {
		CmdArgs []string // UID/RuleID
		ReqBody io.Reader
	},
	env *restrpc.Env,
) error {
	xl := xlog.FromContextSafe(ctx)
	bs, _ := ioutil.ReadAll(req.ReqBody)
	xl.Infof("PostMsgManualBatch__.msg, %+v, %s", req.CmdArgs, string(bs))

	uid64, err := strconv.ParseUint(req.CmdArgs[0], 10, 32)
	if err != nil {
		xl.Errorf("conv uid failed, %+v", err)
		return err
	}
	uid := uint32(uid64)
	ruleID := req.CmdArgs[1]

	var result = struct {
		Error  string   `json:"error"`
		Keys   []string `json:"keys"`
		UID    uint32   `json:"uid"`
		Bucket string   `json:"bucket"`
	}{}
	_ = json.Unmarshal(bs, &result)
	xl.Infof("result %+v", result)
	if result.Error != "" {
		xl.Errorf("RuleInfo: %+v, err: %s", req.CmdArgs, result.Error)
		err := m.Rules.Del(ctx, uid, ruleID) // End Batch Job
		xl.Infof("Rules.Del, %s, %+v", ruleID, err)
	}

	rl, err := m.Rules.QueryByRuleID(ctx, uid, ruleID)
	if err != nil {
		xl.Errorf("QueryByRuleID %s, err: %+v", ruleID, err)
		return err
	}

	go func() {
		ctx := context.Background()
		xl := xlog.FromContextSafe(ctx)
		// Save Back
		err = m.SaveBack.Save(ctx, rl, result.Keys)
		xl.Infof("SaveBack, %s, %+v", ruleID, err)
		err = m.Rules.Del(ctx, uid, ruleID) // End Batch Job
		xl.Infof("Rules.Del, %s, %+v", ruleID, err)
		// TODO:Notify

		if rl.Review.IsOn {
			inUID, ak, sk, bucket, domain := m.SaveBack.GetInnerBucketInfo(ctx)
			kodo := m.SaveBack.GetKodoInfo(ctx)
			// TODO: ConvBjob2Review
			entries := convert.ConvBjob2Review(ctx, rl, kodo,
				ak, sk, bucket, domain, result.Keys)
			err := m.ReviewClient.PushItems(ctx, rl, &client.BatchEntries{
				UID:    inUID,
				Bucket: bucket,
				Keys:   entries,
			})
			if err != nil {
				xl.Errorf("ReviewClient.PushItems %s, %+v, err: %+v", ruleID, entries, err)
			}
		}
	}()

	return nil
}

func (m *_MsgService) PostMsgReview__(ctx context.Context,
	req *struct {
		CmdArgs []string // UID/RuleID
		ReqBody io.Reader
	},
	env *restrpc.Env,
) error {
	xl := xlog.FromContextSafe(ctx)
	bs, _ := ioutil.ReadAll(req.ReqBody)
	xl.Infof("PostMsgReview__.msg, %+v, %s", req.CmdArgs, string(bs))

	uid64, err := strconv.ParseUint(req.CmdArgs[0], 10, 32)
	if err != nil {
		xl.Errorf("conv uid failed, %+v", err)
		return err
	}
	uid := uint32(uid64)
	ruleID := req.CmdArgs[1]

	rl, err := m.Rules.QueryByRuleID(ctx, uid, ruleID)
	if err != nil {
		xl.Errorf("QueryByRuleID %s, err: %+v", ruleID, err)
		return err
	}

	if rl.NotifyURL != nil {
		// TODO: Notify Msg 格式定义
		err := m.NotifyCallback.PostNotifyMsg(ctx, rl, string(bs))
		xl.Errorf("NotifyCallback failed, %s, %+v", ruleID, err)
	}

	return nil
}

//================================================================

func (m *_MsgService) PfopImage(ctx context.Context,
	rl *proto.Rule, reqBytes []byte) error {
	xl := xlog.FromContextSafe(ctx)

	result := struct {
		UID    uint32 `json:"uid"`
		Bucket string `json:"bucket"`
		Key    string `json:"key"`
		Result struct {
			Result convert.PfopImageResult `json:"result"`
		} `json:"result"`
	}{}

	_ = json.Unmarshal(reqBytes, &result)
	if len(result.Result.Result.Result.Suggestion) > 0 {
		xl.Infof("PfopImage result, %s", client.JsonStr(result))

		if rl.Review.IsOn {
			entry := convert.ConvPfop2Review(rl,
				fmt.Sprintf("qiniu:///%s/%s", result.Bucket, result.Key),
				&result.Result.Result)
			err := m.ReviewClient.PushItem(ctx, rl, entry)
			if err != nil {
				xl.Errorf("ReviewClient.PushItem %s, %+v, err: %+v", rl.RuleID, entry, err)
			}
		}

		// 已按最新结果解析
		return nil
	}

	resultOld := struct {
		UID    uint32 `json:"uid"`
		Bucket string `json:"bucket"`
		Key    string `json:"key"`
		Result struct {
			Result convert.PfopImageResultOld `json:"result"`
		} `json:"result"`
	}{}
	_ = json.Unmarshal(reqBytes, &resultOld)
	xl.Infof("PfopImage resultOld, %s", client.JsonStr(resultOld))

	if rl.Review.IsOn {
		entry := convert.ConvPfopOld2Review(rl,
			fmt.Sprintf("qiniu:///%s/%s", resultOld.Bucket, resultOld.Key),
			&resultOld.Result.Result)
		err := m.ReviewClient.PushItem(ctx, rl, entry)
		if err != nil {
			xl.Errorf("ReviewClient.PushItem %s, %+v, err: %+v", rl.RuleID, entry, err)
		}
	}

	return nil
}

func (m *_MsgService) PfopVideo(ctx context.Context,
	rl *proto.Rule, reqBytes []byte) error {
	xl := xlog.FromContextSafe(ctx)
	result := struct {
		UID    uint32 `json:"uid"`
		Bucket string `json:"bucket"`
		Key    string `json:"key"`
		Result struct {
			Result convert.PfopVideoResult `json:"result"`
		} `json:"result"`
	}{}
	_ = json.Unmarshal(reqBytes, &result)
	xl.Infof("PfopVideo result, %s", client.JsonStr(result))

	if rl.Manual.IsOn {
		// TODO
	}

	if rl.Review.IsOn {
		entry := convert.ConvPfopVideo2Review(rl,
			fmt.Sprintf("qiniu:///%s/%s", result.Bucket, result.Key),
			&result.Result.Result)
		err := m.ReviewClient.PushItem(ctx, rl, entry)
		if err != nil {
			xl.Errorf("ReviewClient.PushItem %s, %+v, err: %+v", rl.RuleID, entry, err)
		}
	}

	return nil
}

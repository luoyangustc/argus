package cap

import (
	"context"
	"errors"

	xlog "github.com/qiniu/xlog.v1"

	"qiniu.com/argus/ccp/manual/model"
	authstub "qiniu.com/auth/authstub.v1"
)

type IService interface {
	PostSets(
		context.Context,
		*model.SetModel,
		*authstub.Env,
	) error

	GetSets_(
		context.Context,
		*struct {
			CmdArgs []string // setID
		},
		*authstub.Env,
	) (
		*model.SetModel,
		error,
	)

	GetSets(
		context.Context,
		*authstub.Env,
	) (
		*model.QuerySetsResp,
		error,
	)

	PostSets_Entries( // 发送多个entrys
		context.Context,
		*model.BatchEntriesReq,
		*authstub.Env,
	) error

	PostSets_Entries_(
		context.Context,
		*struct {
			CmdArgs []string          // setID,entryID
			Entry   *model.EntryModel `json:"entry"`
		},
		*authstub.Env,
	) error

	GetSets_Entries(
		context.Context,
		*struct {
			CmdArgs []string // setID
			Offset  int      `json:"offset"`
			Limit   int      `json:"limit"`
		},
		*authstub.Env,
	) ([]model.EntryModel, error)

	GetSets_Entries_(
		context.Context,
		*struct {
			CmdArgs []string // setID, entryID
		},
		*authstub.Env,
	) (*model.EntryModel, error)
}

func NewService(ctx context.Context, handler IManualHandler) IService {
	return _SetService{
		IManualHandler: handler,
	}
}

var _ IService = &_SetService{}

type _SetService struct {
	IManualHandler
}

func (s _SetService) PostSets(
	ctx context.Context,
	req *model.SetModel,
	auth *authstub.Env,
) error {
	if req == nil {
		return errors.New("invalid CmdArgs")
	}

	var (
		xl = xlog.FromContextSafe(ctx)
	)
	xl.Infof("req param: %#v", req)
	err := s.IManualHandler.InsertSet(ctx, req)
	if err != nil {
		xl.Warnf("post set error: %#v", err.Error())
		return err
	}

	return nil
}

func (s _SetService) GetSets_(
	ctx context.Context,
	req *struct {
		CmdArgs []string // setID
	},
	auth *authstub.Env,
) (
	*model.SetModel,
	error,
) {
	if req == nil || len(req.CmdArgs) <= 0 {
		return nil, errors.New("invalid CmdArgs")
	}
	resp, err := s.IManualHandler.QuerySetById(ctx, req.CmdArgs[0])
	if err != nil {
		return nil, err
	}

	return resp, nil
}

func (s _SetService) GetSets(
	ctx context.Context,
	auth *authstub.Env,
) (
	*model.QuerySetsResp,
	error,
) {
	var (
		xl = xlog.FromContextSafe(ctx)
	)
	resp, err := s.IManualHandler.QuerySets(ctx)
	if err != nil {
		xl.Warnf("querySets error: %#v", err.Error())
		return nil, err
	}
	return resp, nil
}

//存量请求
func (set _SetService) PostSets_Entries(
	ctx context.Context,
	req *model.BatchEntriesReq,
	auth *authstub.Env,
) error {
	var (
		xl = xlog.FromContextSafe(ctx)
	)
	if req == nil || len(req.CmdArgs) <= 0 {
		xl.Errorf("invalid CmdArgs")
		return errors.New("invalid CmdArgs")
	}
	if req.Keys == nil || len(req.Keys) == 0 {
		xl.Errorf("input files is empty: %v", req.Keys)
		return errors.New("invalid keys")
	}

	if req.Uid == 0 || req.Bucket == "" {
		xl.Errorf("input Uid, Bucket is invalid: <%d, %s>", req.Uid, req.Bucket)
		return errors.New("invalid uid or bucket")
	}

	err := set.IManualHandler.InsertEntries(ctx, req.Uid, req.Bucket, req.CmdArgs[0], req.Keys)
	if err != nil {
		xl.Errorf("InsertEntries error: %#v", err.Error())
		return err
	}
	return nil
}

//实时单个entry
func (set _SetService) PostSets_Entries_(
	ctx context.Context,
	req *struct {
		CmdArgs []string          // setID, entryID
		Entry   *model.EntryModel `json:"entry"`
	},
	auth *authstub.Env,
) error {

	if req == nil || len(req.CmdArgs) <= 1 {
		return errors.New("invalid CmdArgs")
	}

	var (
		xl = xlog.FromContextSafe(ctx)
	)

	req.Entry.EntryID = req.CmdArgs[1]
	err := set.InsertEntry(ctx, req.CmdArgs[0], req.Entry)
	if err != nil {
		xl.Warnf("InsertEntry error: %#v", err.Error())
		return err
	}

	return nil
}

func (set _SetService) GetSets_Entries(
	ctx context.Context,
	req *struct {
		CmdArgs []string // setID
		Offset  int      `json:"offset"`
		Limit   int      `json:"limit"`
		//TODO: 增加其他查询过滤条件
		// Suggestion *byte           `json:"suggestion"` // PASS: 0, BLOCK: 1, REVIEW: 2
		// Scenes     map[string]byte `json:"scenes"`     // pulp|terror|politician
		// // switch src
		// // case "KODO":
		// // 	 More = struct { Bucket string `json:"bucket,omitempty"	}
		// Source json.RawMessage `json:"source,omitempty"`
	},
	auth *authstub.Env,
) ([]model.EntryModel, error) {
	if req == nil || len(req.CmdArgs) <= 0 {
		return nil, errors.New("invalid CmdArgs")
	}

	var (
		xl = xlog.FromContextSafe(ctx)
	)

	resp, err := set.QueryEntries(ctx, req.CmdArgs[0], req.Offset, req.Limit)
	if err != nil {
		xl.Warnf("InsertEntry error: %#v", err.Error())
		return nil, err
	}

	return resp, nil
}

func (set _SetService) GetSets_Entries_(
	ctx context.Context,
	req *struct {
		CmdArgs []string // setID, entryID
	},
	auth *authstub.Env,
) (*model.EntryModel, error) {
	if req == nil || len(req.CmdArgs) <= 1 {
		return nil, errors.New("invalid CmdArgs")
	}

	var (
		xl = xlog.FromContextSafe(ctx)
	)

	resp, err := set.QueryEntry(ctx, req.CmdArgs[0], req.CmdArgs[1])
	if err != nil {
		xl.Warnf("InsertEntry error: %#v", err.Error())
		return nil, err
	}

	return resp, nil
}

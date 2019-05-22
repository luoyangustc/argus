package review

import (
	"context"
	"errors"
	"time"

	httputil "github.com/qiniu/http/httputil.v1"
	xlog "github.com/qiniu/xlog.v1"
	"qiniu.com/argus/ccp/review/concerns"
	"qiniu.com/argus/ccp/review/dao"
	"qiniu.com/argus/ccp/review/enums"
	"qiniu.com/argus/ccp/review/misc"
	"qiniu.com/argus/ccp/review/model"
	authstub "qiniu.com/auth/authstub.v1"
)

type (
	IService interface {
		PostSets(
			context.Context,
			*model.Set,
			*authstub.Env,
		) error
		GetSets_(
			context.Context,
			*struct {
				CmdArgs []string // setID
			},
			*authstub.Env,
		) (
			*model.Set,
			error,
		)
		PostSets_Entry(
			context.Context,
			*struct {
				CmdArgs      []string // setID
				*model.Entry `json:",inline"`
			},
			*authstub.Env,
		) error
		PostSets_Entries( // ? overwrite
			context.Context,
			*struct {
				CmdArgs []string // setID
				Uid     uint32   `json:"uid"`
				Bucket  string   `json:"bucket"`
				Keys    []string `json:"keys"`
			},
			*authstub.Env,
		) error
		PostSets_Entries_(
			context.Context,
			*struct {
				CmdArgs                  []string // setID, entryID
				*model.FininalSuggestion `json:",inline"`
			},
			*authstub.Env,
		) error
		PostFetchEntries(
			context.Context,
			*struct {
				*dao.SetFilter   `json:",inline"`
				*dao.EntryFilter `json:",inline"`
				*dao.Paginator   `json:",inline"`
			},
			*authstub.Env,
		) ([]*EntryResponse, error)
		PostSetsCounters(
			context.Context,
			*struct {
				SetIds []string `json:"sets"`
			},
			*authstub.Env,
		) ([]*model.SetCounter, error)
		GetEntries_Cuts(
			context.Context,
			*struct {
				CmdArgs []string // entryId
				Marker  string   `json:"marker"`
				Limit   int      `json:"limit"`
			},
			*authstub.Env,
		) ([]*model.VideoCut, error)
		GetEntries_CutsCount(
			context.Context,
			*struct {
				CmdArgs []string // entryId
			},
			*authstub.Env,
		) (map[string]int, error)
		PostSets_Reset(
			context.Context,
			*struct {
				CmdArgs    []string         // setID
				Type       string           `json:"type"`
				SourceType enums.SourceType `json:"source_type"`
				JobType    enums.JobType    `json:"job_type"`
			},
			*authstub.Env,
		) error
	}
)

func NewService() IService {
	return &_ReviewService{}
}

type _ReviewService struct{}

var (
	invalidParamsErr = errors.New("Invalid Params")
	invalidAuthErr   = errors.New("Invalid Auth")
)

func (_ *_ReviewService) PostSets(
	ctx context.Context,
	set *model.Set,
	env *authstub.Env,
) (err error) {
	defer func(t1 time.Time) {
		var code = 200
		if err != nil {
			code = httputil.DetectCode(err)
		}
		misc.RequestsCounter("PostSets", code).Inc()
		misc.ResponseTime("PostSets", code).Observe(float64(time.Since(t1) / time.Second))
	}(time.Now())

	xl := xlog.FromContextSafe(ctx)

	if set == nil {
		xl.Warnf("no set found")
		return invalidParamsErr
	}

	set.Uid = env.UserInfo.Uid

	if !set.IsValid() {
		xl.Warnf("input set is invalid: %#v", set)
		return invalidParamsErr
	}

	return dao.SetDao.Insert(ctx, set)
}

func (_ *_ReviewService) GetSets_(
	ctx context.Context,
	req *struct {
		CmdArgs []string
	},
	env *authstub.Env,
) (
	set *model.Set,
	err error,
) {
	defer func(t1 time.Time) {
		var code = 200
		if err != nil {
			code = httputil.DetectCode(err)
		}
		misc.RequestsCounter("GetSets_", code).Inc()
		misc.ResponseTime("GetSets_", code).Observe(float64(time.Since(t1) / time.Second))
	}(time.Now())

	if len(req.CmdArgs) == 0 {
		err = invalidParamsErr
		return
	}

	set, err = dao.SetDao.Find(ctx, req.CmdArgs[0])
	return
}

func (_ *_ReviewService) PostSets_Entry(
	ctx context.Context,
	req *struct {
		CmdArgs      []string
		*model.Entry `json:",inline"`
	},
	env *authstub.Env,
) (err error) {
	defer func(t1 time.Time) {
		var code = 200
		if err != nil {
			code = httputil.DetectCode(err)
		}
		misc.RequestsCounter("PostSets_Entry", code).Inc()
		misc.ResponseTime("PostSets_Entry", code).Observe(float64(time.Since(t1) / time.Second))
	}(time.Now())

	if len(req.CmdArgs) == 0 || req.Entry == nil {
		return invalidParamsErr
	}

	xl := xlog.FromContextSafe(ctx)

	entryDao, err := dao.EntrySetCache.GetDao(req.CmdArgs[0])
	if err != nil {
		xl.Errorf("dao.EntrySetCache.GetDao(%s): %v", req.CmdArgs[0], err)
		return err
	}

	entry := req.Entry
	entry.SetId = req.CmdArgs[0]

	if err := entry.Patch(); err != nil {
		xl.Errorf("failed to patching version from: %#v", entry.Version)
		return err
	}

	if err = entryDao.Insert(ctx, entry); err != nil {
		xl.Errorf("entryDao.Insert(%#v): %v", entry, err)
		return err
	}

	go concerns.EntryCounter.CheckEntry(ctx, entry)

	return nil
}

func (_ *_ReviewService) PostSets_Entries(
	ctx context.Context,
	req *struct {
		CmdArgs []string // setID
		Uid     uint32   `json:"uid"`
		Bucket  string   `json:"bucket"`
		Keys    []string `json:"keys"`
	},
	env *authstub.Env,
) (err error) {
	defer func(t1 time.Time) {
		var code = 200
		if err != nil {
			code = httputil.DetectCode(err)
		}
		misc.RequestsCounter("PostSets_Entries", code).Inc()
		misc.ResponseTime("PostSets_Entries", code).Observe(float64(time.Since(t1) / time.Second))
	}(time.Now())

	xl := xlog.FromContextSafe(ctx)

	if len(req.CmdArgs) == 0 || req.CmdArgs[0] == "" {
		xl.Warnf("no set found")
		return invalidParamsErr
	}

	if req.Keys == nil || len(req.Keys) == 0 {
		xl.Warnf("input files is empty: %v", req.Keys)
		return invalidParamsErr
	}

	if req.Uid == 0 || req.Bucket == "" {
		xl.Warnf("input Uid, Bucket is invalid: <%d, %s>", req.Uid, req.Bucket)
		return invalidParamsErr
	}

	jobs := model.NewBatchEntryJobs(req.Uid, req.Bucket, req.CmdArgs[0], req.Keys)

	if err := dao.BatchEntryJobDAO.BatchInsert(ctx, jobs); err != nil {
		xl.Errorf("dao.BatchEntryJobDAO.BatchInsert: <%#v>", err)
		return err
	}

	return nil
}

func (_ *_ReviewService) PostSets_Entries_(
	ctx context.Context,
	req *struct {
		CmdArgs                  []string // setID, entryID
		*model.FininalSuggestion `json:",inline"`
	},
	env *authstub.Env,
) (err error) {
	defer func(t1 time.Time) {
		var code = 200
		if err != nil {
			code = httputil.DetectCode(err)
		}
		misc.RequestsCounter("PostSets_Entries_", code).Inc()
		misc.ResponseTime("PostSets_Entries_", code).Observe(float64(time.Since(t1) / time.Second))
	}(time.Now())

	if len(req.CmdArgs) != 2 {
		return invalidParamsErr
	}

	xl := xlog.FromContextSafe(ctx)

	entryDao, err := dao.EntrySetCache.GetDao(req.CmdArgs[0])
	if err != nil {
		xl.Errorf("dao.EntrySetCache.GetDao(%#v): %v", req.CmdArgs[0], err)
		return err
	}

	entry, err := entryDao.Find(ctx, req.CmdArgs[1])
	if err != nil {
		xl.Errorf("entryDao.Find(%s): %v", req.CmdArgs[1], err)
		return err
	}

	entry.Final = req.FininalSuggestion
	if err = entryDao.Update(ctx, req.CmdArgs[1], entry); err != nil {
		xl.Errorf("entryDao.Find(%s, %#v): %v", req.CmdArgs[1], entry, err)
		return err
	}

	// do notify
	if entry.Final != nil && entry.Final.Suggestion != entry.Original.Suggestion {
		go concerns.NotifySender.Perform(ctx, entry)
	}

	return
}

type EntryResponse struct {
	Bucket       string `json:"bucket"`
	Automatic    bool   `json:"automatic"`
	Manual       bool   `json:"manual"`
	*model.Entry `json:",inline"`
}

func (_ *_ReviewService) PostFetchEntries(
	ctx context.Context,
	req *struct {
		*dao.SetFilter   `json:",inline"`
		*dao.EntryFilter `json:",inline"`
		*dao.Paginator   `json:",inline"`
	},
	env *authstub.Env,
) (ret []*EntryResponse, err error) {
	defer func(t1 time.Time) {
		var code = 200
		if err != nil {
			code = httputil.DetectCode(err)
		}
		misc.RequestsCounter("PostFetchEntries", code).Inc()
		misc.ResponseTime("PostFetchEntries", code).Observe(float64(time.Since(t1) / time.Second))
	}(time.Now())

	xl := xlog.FromContextSafe(ctx)

	// for params validate
	if req.SetFilter == nil ||
		req.EntryFilter == nil ||
		req.Paginator == nil {
		xl.Warnf("miss params %v, %v, %v", req.SetFilter, req.EntryFilter, req.Paginator)
		return nil, invalidParamsErr
	}

	// set uid
	req.SetFilter.Uid = env.UserInfo.Uid

	if !req.SetFilter.IsValid() {
		xl.Warnf("req SetFilter is invalid %#v", req.SetFilter)
		return nil, invalidParamsErr
	}

	setIds, err := req.SetFilter.GetSetIds()
	if err != nil {
		xl.Errorf("req.SetFilter.GetSetIds(%#v): %v", req.SetFilter, err)
		return nil, err
	}

	entryDao, err := req.SetFilter.GetEntryDao()
	if err != nil {
		xl.Errorf("req.SetFilter.GetEntryDao(%#v): %v", req.SetFilter, err)
		return nil, err
	}

	req.EntryFilter.SetIds = setIds

	entries, err := entryDao.Query(ctx, req.EntryFilter, req.Paginator)
	if err != nil {
		xl.Errorf("entryDao.Query(%#v, %#v): %v", req.EntryFilter, req.Paginator, err)
		return nil, err
	}

	// inject set info
	ret = make([]*EntryResponse, len(entries))
	for i, entry := range entries {
		xl.Debugf("preparing converting entry from version: %#v", entry.Version)
		err := entry.Patch()
		if err != nil {
			xl.Errorf("failed to patching version from: %#v", entry.Version)
			continue
		}
		ret[i] = &EntryResponse{
			Entry: entry,
		}

		if set, err := dao.EntrySetCache.MustGet(entry.SetId); err == nil {
			ret[i].Bucket = set.Bucket
			ret[i].Automatic = set.Automatic
			ret[i].Manual = set.Manual
		}
	}

	return
}

func (_ *_ReviewService) PostSetsCounters(
	ctx context.Context,
	req *struct {
		SetIds []string `json:"sets"`
	},
	env *authstub.Env,
) (items []*model.SetCounter, err error) {
	defer func(t1 time.Time) {
		var code = 200
		if err != nil {
			code = httputil.DetectCode(err)
		}
		misc.RequestsCounter("PostSetsCounters", code).Inc()
		misc.ResponseTime("PostSetsCounters", code).Observe(float64(time.Since(t1) / time.Second))
	}(time.Now())

	if req.SetIds == nil || len(req.SetIds) == 0 {
		return nil, invalidParamsErr
	}

	xl := xlog.FromContextSafe(ctx)

	sets, err := dao.SetDao.QueryBySets(ctx, req.SetIds)
	if err != nil {
		xl.Errorf("dao.SetDao.QueryBySets(%v): %v", req.SetIds, err)
		return nil, err
	}

	var (
		sResourceIds, batchSetIds []string
	)

	for _, set := range sets {
		if set.Type == enums.JobTypeStream {
			sResourceIds = append(sResourceIds, set.ResourceId())
		} else {
			batchSetIds = append(batchSetIds, set.SetId)
		}
	}

	// load stream items
	sItems, err := dao.SetCounterDAO.QueryByResourceID(ctx, env.Uid, sResourceIds)
	if err != nil {
		xl.Errorf("dao.SetCounterDAO.QueryByResourceID(%d, %v): %v", env.Uid, sResourceIds, err)
		return nil, err
	}

	// load batch items
	batchSetItems, err := dao.SetCounterDAO.Query(ctx, env.Uid, batchSetIds)
	if err != nil {
		xl.Errorf("dao.SetCounterDAO.Query(%d, %v): %v", env.Uid, batchSetIds, err)
		return nil, err
	}

	sItems = append(sItems, batchSetItems...)

	sItemsMap := make(map[string]*model.SetCounter)
	for _, item := range sItems {
		cItem, ok := sItemsMap[item.ResourceId]
		if ok {
			cItem.MergeWith(item)
		} else {
			sItemsMap[item.ResourceId] = item
		}
	}

	items = make([]*model.SetCounter, 0, len(sets))

	for _, set := range sets {
		if counter, ok := sItemsMap[set.ResourceId()]; ok {
			counter.SetId = set.SetId
			items = append(items, counter)
		}
	}

	return
}

func (_ *_ReviewService) GetEntries_Cuts(
	ctx context.Context,
	req *struct {
		CmdArgs []string // entryId
		Marker  string   `json:"marker"`
		Limit   int      `json:"limit"`
	},
	env *authstub.Env,
) (ret []*model.VideoCut, err error) {
	defer func(t1 time.Time) {
		var code = 200
		if err != nil {
			code = httputil.DetectCode(err)
		}
		misc.RequestsCounter("GetEntries_Cuts", code).Inc()
		misc.ResponseTime("GetEntries_Cuts", code).Observe(float64(time.Since(t1) / time.Second))
	}(time.Now())

	if req.CmdArgs == nil || len(req.CmdArgs) == 0 {
		return nil, invalidParamsErr
	}

	ret, err = dao.VideoCutDAO.Query(ctx, req.CmdArgs[0], dao.NewPaginator(req.Marker, req.Limit))
	return
}

func (_ *_ReviewService) GetEntries_CutsCount(
	ctx context.Context,
	req *struct {
		CmdArgs []string
	},
	env *authstub.Env,
) (ret map[string]int, err error) {
	defer func(t1 time.Time) {
		var code = 200
		if err != nil {
			code = httputil.DetectCode(err)
		}
		misc.RequestsCounter("GetEntries_CutsCount", code).Inc()
		misc.ResponseTime("GetEntries_CutsCount", code).Observe(float64(time.Since(t1) / time.Second))
	}(time.Now())

	if len(req.CmdArgs) == 0 {
		return nil, invalidParamsErr
	}

	n, err := dao.VideoCutDAO.Count(ctx, req.CmdArgs[0])
	if err != nil {
		xl := xlog.FromContextSafe(ctx)
		xl.Errorf("dao.VideoCutDAO.Count(%s): %v", req.CmdArgs[0], err)
		return nil, err
	}

	ret = map[string]int{"total": n}
	return
}

func (_ *_ReviewService) PostSets_Reset(
	ctx context.Context,
	req *struct {
		CmdArgs    []string         // setID
		Type       string           `json:"type"`
		SourceType enums.SourceType `json:"source_type"`
		JobType    enums.JobType    `json:"job_type"`
	},
	env *authstub.Env,
) (err error) {
	defer func(t1 time.Time) {
		var code = 200
		if err != nil {
			code = httputil.DetectCode(err)
		}
		misc.RequestsCounter("PostSets_Reset", code).Inc()
		misc.ResponseTime("PostSets_Reset", code).Observe(float64(time.Since(t1) / time.Second))
	}(time.Now())

	if len(req.CmdArgs) == 0 {
		return invalidParamsErr
	}

	xl := xlog.FromContextSafe(ctx)
	uid := env.UserInfo.Uid
	setID := req.CmdArgs[0]

	// 验证参数
	supportedEntryType := map[string]dao.EntryDAO{
		string(enums.SourceTypeKodo) + string(enums.JobTypeStream): dao.QnIncEntriesDao,
		string(enums.SourceTypeKodo) + string(enums.JobTypeBatch):  dao.QnInvEntriesDao,
		string(enums.SourceTypeApi) + string(enums.JobTypeStream):  dao.ApiIncEntriesDao,
		string(enums.SourceTypeApi) + string(enums.JobTypeBatch):   dao.ApiInvEntriesDao,
	}

	entryDao, ok := supportedEntryType[string(req.SourceType)+string(req.JobType)]
	if !ok {
		xl.Errorf("set reset (%s %s) not supported", req.SourceType, req.JobType)
		return invalidParamsErr
	}

	if req.Type != "soft" && req.Type != "hard" && req.Type != "remove" {
		xl.Errorf("set reset type %s not supported", req.Type)
		return invalidParamsErr
	}

	set, err := dao.SetDao.Find(ctx, setID)
	if err != nil {
		xl.Errorf("set(%s) not found : %v", setID, err)
		return err
	} else if set.Uid != uid {
		// 不允许删除非自己的set
		xl.Errorf("set uid is %d not %d", set.Uid, uid)
		return invalidAuthErr
	}

	if req.Type == "soft" {
		// remove entries
		err = entryDao.Remove(ctx, setID)
		if err != nil && err != dao.ErrNotFound {
			xl.Errorf("entryDao.Remove(%s): %v", setID, err)
			return err
		}

		// update set_counter
		setCounter, err := dao.SetCounterDAO.Find(ctx, setID)
		if err != nil {
			if err != dao.ErrNotFound {
				xl.Errorf("dao.SetCounterDAO.Find(%s): %v", setID, err)
				return err
			}
		} else {
			setCounter.Values = make(map[enums.Scene]int)
			setCounter.Values2 = make(map[enums.Scene]int)
			err = dao.SetCounterDAO.Update(ctx, setCounter, 0)
			if err != nil {
				xl.Errorf("dao.SetCounterDAO.Update(%s, *): %v", setID, err)
				return err
			}
		}

		// update batch_job
		if req.JobType == enums.JobTypeBatch {
			err := dao.BatchEntryJobDAO.UpdateStatusBySetId(ctx, setID, enums.BatchEntryJobStatusNew)
			if err != nil {
				if err != dao.ErrNotFound {
					xl.Errorf("dao.SetCounterDAO.Find(%s): %v", setID, err)
					return err
				}
			}
		}
	} else if req.Type == "hard" {
		// remove entries
		err = entryDao.Remove(ctx, setID)
		if err != nil && err != dao.ErrNotFound {
			xl.Errorf("entryDao.Remove(%s): %v", setID, err)
			return err
		}

		// remove entry_job
		if req.JobType == enums.JobTypeBatch {
			err = dao.BatchEntryJobDAO.RemoveBySetId(ctx, setID)
			if err != nil && err != dao.ErrNotFound {
				xl.Errorf("dao.BatchEntryJobDAO.Remove(%s): %v", setID, err)
				return err
			}
		}

		// remove set_counter
		err = dao.SetCounterDAO.Remove(ctx, uid, setID)
		if err != nil && err != dao.ErrNotFound {
			xl.Errorf("dao.SetCounterDAO.Remove(%s): %v", setID, err)
			return err
		}
	} else if req.Type == "remove" {
		// remove entries
		err = entryDao.Remove(ctx, setID)
		if err != nil && err != dao.ErrNotFound {
			xl.Errorf("entryDao.Remove(%s): %v", setID, err)
			return err
		}

		// remove entry_job
		if req.JobType == enums.JobTypeBatch {
			err = dao.BatchEntryJobDAO.RemoveBySetId(ctx, setID)
			if err != nil && err != dao.ErrNotFound {
				xl.Errorf("dao.BatchEntryJobDAO.Remove(%s): %v", setID, err)
				return err
			}
		}

		// remove set_counter
		err = dao.SetCounterDAO.Remove(ctx, uid, setID)
		if err != nil && err != dao.ErrNotFound {
			xl.Errorf("dao.SetCounterDAO.Remove(%s): %v", setID, err)
			return err
		}

		// remove set
		err = dao.SetDao.Remove(ctx, uid, setID)
		if err != nil && err != dao.ErrNotFound {
			xl.Errorf("dao.SetDao.Remove(%s): %v", setID, err)
			return err
		}
	}

	return nil
}

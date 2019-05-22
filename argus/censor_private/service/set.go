package service

import (
	"context"
	"io/ioutil"
	"strconv"
	"time"

	"gopkg.in/mgo.v2/bson"
	"qiniu.com/argus/censor_private/dao"
	"qiniu.com/argus/censor_private/proto"
	"qiniu.com/argus/censor_private/util"

	restrpc "github.com/qiniu/http/restrpc.v1"
	"github.com/qiniu/uuid"
	xlog "github.com/qiniu/xlog.v1"
)

const MIN_MONITOR_INTERVAL = 15

type SetAddReq struct {
	Id               string           `json:"-"`
	Name             string           `json:"name"`
	Type             proto.SetType    `json:"type"`
	Scenes           []proto.Scene    `json:"scenes"`
	Uri              string           `json:"uri"`
	CutIntervalMsecs int              `json:"cut_interval_msecs"`
	MonitorInterval  int              `json:"monitor_interval"`
	MimeTypes        []proto.MimeType `json:"mime_types"`
}

type SetAddResp struct {
	Id string `json:"id"`
}

func (s *Service) PostSetAdd(
	ctx context.Context,
	req *SetAddReq,
	env *restrpc.Env,
) (*SetAddResp, error) {
	xl := xlog.FromContextSafe(ctx)

	if req.Type == proto.SetTypeTask {
		// task类型由upload接口隐式创建
		return nil, proto.ErrInvalidType
	}
	// validate
	err := s.validateSet(xl, req)
	if err != nil {
		return nil, err
	}

	// add set
	set := &proto.Set{
		Name:             req.Name,
		Uri:              req.Uri,
		Scenes:           req.Scenes,
		MimeTypes:        req.MimeTypes,
		CutIntervalMsecs: req.CutIntervalMsecs,
		MonitorInterval:  req.MonitorInterval,
		Status:           proto.SetStatusRunning,
		Type:             req.Type,
	}
	id, err := s.createSet(ctx, set)
	if err != nil {
		return nil, err
	}

	xl.Infof("create set %#v", set)
	return &SetAddResp{Id: id}, nil
}

func (s *Service) PostSetUpload(
	ctx context.Context,
	env *restrpc.Env,
) (*SetAddResp, error) {
	var (
		xl  = xlog.FromContextSafe(ctx)
		err error
	)
	_ = env.Req.ParseMultipartForm(32 << 20)

	req := &SetAddReq{Type: proto.SetTypeTask, CutIntervalMsecs: 0}
	req.Name = env.Req.FormValue("name")
	for _, v := range env.Req.Form["scenes"] {
		req.Scenes = append(req.Scenes, proto.Scene(v))
	}
	for _, v := range env.Req.Form["mime_types"] {
		req.MimeTypes = append(req.MimeTypes, proto.MimeType(v))
	}
	cutIntervalMsecs := env.Req.FormValue("cut_interval_msecs")
	if len(cutIntervalMsecs) > 0 {
		req.CutIntervalMsecs, err = strconv.Atoi(cutIntervalMsecs)
		if err != nil {
			xl.Errorf("cut interval is not number : %s", cutIntervalMsecs)
			return nil, proto.ErrInvalidCutInterval
		}
	}

	// validate
	err = s.validateSet(xl, req)
	if err != nil {
		return nil, err
	}

	// read file
	formFile, _, err := env.Req.FormFile("file")
	if err != nil {
		xl.Error("file is not uploaded : ", err)
		return nil, proto.ErrInvalidFile
	}
	defer formFile.Close()

	bs, err := ioutil.ReadAll(formFile)
	if err != nil {
		xl.Error("read uploaded file err : ", err)
		return nil, proto.ErrInvalidFile
	}

	// insert in db
	id, err := uuid.Gen(16)
	if err != nil {
		return nil, proto.ErrGenId
	}
	count := dao.InsertEntries(ctx, id, req.MimeTypes, util.ByteArrayToLines(bs))
	if count == 0 {
		return nil, proto.ErrInvalidFile
	}

	// add set
	set := &proto.Set{
		Id:               id,
		Name:             req.Name,
		Scenes:           req.Scenes,
		MimeTypes:        req.MimeTypes,
		CutIntervalMsecs: req.CutIntervalMsecs,
		Status:           proto.SetStatusRunning,
		Type:             req.Type,
	}
	id, err = s.createSet(ctx, set)
	if err != nil {
		return nil, err
	}

	xl.Infof("create set(file uploaded) %#v with %d records", set, count)
	return &SetAddResp{Id: id}, nil
}

func (s *Service) PostSet_Start(
	ctx context.Context,
	req *struct{ CmdArgs []string },
	env *restrpc.Env,
) error {
	xl := xlog.FromContextSafe(ctx)

	id := req.CmdArgs[0]
	if len(id) == 0 {
		return proto.ErrEmptyId
	}

	set, err := dao.SetDao.Find(id)
	if err != nil {
		xl.Errorf("setDao.Find(%s): %v", id, err)
		return err
	}

	status := set.Status
	if status != proto.SetStatusStopped {
		xl.Errorf("set is not stopped : %s, %s", id, status)
		return proto.ErrSetNotStopped
	}

	err = dao.SetDao.Patch(id, bson.M{"status": proto.SetStatusRunning})
	if err != nil {
		xl.Errorf("setDao.Patch(%s, %s): %v", id, proto.SetStatusRunning, err)
		return err
	}

	// start job
	err = s.dispatcher.Run(ctx, set)
	if err != nil {
		xl.Errorf("fail to start job: %#v", set)
		// rollback
		_ = dao.SetDao.Patch(id, bson.M{"status": status})
	}

	return nil
}

func (s *Service) PostSet_Stop(
	ctx context.Context,
	req *struct{ CmdArgs []string },
	env *restrpc.Env,
) error {
	xl := xlog.FromContextSafe(ctx)

	id := req.CmdArgs[0]
	if len(id) == 0 {
		return proto.ErrEmptyId
	}

	set, err := dao.SetDao.Find(id)
	if err != nil {
		xl.Errorf("setDao.Find(%s): %v", id, err)
		return err
	}

	status := set.Status
	if status != proto.SetStatusRunning {
		xl.Errorf("set is not running : %s, %s", id, status)
		return proto.ErrSetNotRunning
	}

	err = dao.SetDao.Patch(id, bson.M{"status": proto.SetStatusStopped})
	if err != nil {
		xl.Errorf("setDao.Patch(%s, %s): %v", id, proto.SetStatusStopped, err)
		return err
	}

	// stop job
	err = s.dispatcher.Stop(ctx, set.Id)
	if err != nil {
		xl.Errorf("fail to stop job: %#v", set)
		// rollback
		_ = dao.SetDao.Patch(id, bson.M{"status": status})
	}

	return nil
}

type SetsReq struct {
	dao.SetFilter
}

type SetsResp struct {
	Datas []*proto.Set `json:"datas"`
}

func (_ *Service) GetSets(
	ctx context.Context,
	req *SetsReq,
	env *restrpc.Env,
) (*SetsResp, error) {
	xl := xlog.FromContextSafe(ctx)

	// get sets
	sets, err := dao.SetDao.Query(&req.SetFilter)
	if err != nil {
		xl.Errorf("setDao entryDao.Query(%#v): %v", req.SetFilter, err)
		return nil, err
	}

	resp := &SetsResp{
		Datas: sets,
	}
	return resp, nil
}

type SetsHistoryResp struct {
	Datas []*proto.SetHistory `json:"datas"`
}

func (_ *Service) GetSet_History(
	ctx context.Context,
	req *struct{ CmdArgs []string },
	env *restrpc.Env,
) (*SetsHistoryResp, error) {
	xl := xlog.FromContextSafe(ctx)

	id := req.CmdArgs[0]
	if len(id) == 0 {
		return nil, proto.ErrEmptyId
	}

	set, err := dao.SetDao.Find(id)
	if err != nil {
		xl.Errorf("setDao.Find(%s): %v", id, err)
		return nil, err
	}

	items := make([]*proto.SetHistory, 0)
	start := set.CreatedAt
	if set.ModifiedAt > 0 {
		start = set.ModifiedAt
	}
	items = append(items, set.GenHistory(start, 0))

	histories, err := dao.SetHistoryDao.Query(id)
	if err != nil {
		xl.Errorf("setHistoryDao.Query(%s): %v", id, err)
		return nil, err
	}

	for _, v := range histories {
		v.Status = proto.SetStatusStopped
	}
	items = append(items, histories...)

	resp := &SetsHistoryResp{
		Datas: items,
	}
	return resp, nil
}

type SetUpdateReq struct {
	CmdArgs []string
	SetAddReq
}

func (s *Service) PostSet_Update(
	ctx context.Context,
	req *SetUpdateReq,
	env *restrpc.Env,
) error {
	xl := xlog.FromContextSafe(ctx)

	id := req.CmdArgs[0]
	if len(id) == 0 {
		return proto.ErrEmptyId
	}

	set, err := dao.SetDao.Find(id)
	if err != nil {
		xl.Errorf("setDao.Find(%s): %v", id, err)
		return err
	}

	req.Id = id
	req.Type = set.Type

	// validate
	err = s.validateSet(xl, &req.SetAddReq)
	if err != nil {
		return err
	}

	// save history
	now := time.Now().Unix()
	start := set.CreatedAt
	if set.ModifiedAt > 0 {
		start = set.ModifiedAt
	}
	history := set.GenHistory(start, now)
	historyId, err := dao.SetHistoryDao.Insert(history)
	if err != nil {
		xl.Errorf("setHistoryDao.Insert(%#v): %v", history, err)
		return err
	}

	defer func() {
		if err != nil {
			// rollback
			_ = dao.SetHistoryDao.Remove(historyId)
			_ = dao.SetDao.Update(id, set)
		}
	}()

	newSet := *set
	newSet.ModifiedAt = now
	newSet.Scenes = req.Scenes
	newSet.CutIntervalMsecs = req.CutIntervalMsecs
	newSet.MonitorInterval = req.MonitorInterval
	newSet.MimeTypes = req.MimeTypes
	newSet.Uri = req.Uri
	newSet.Name = req.Name

	err = dao.SetDao.Update(id, &newSet)
	if err != nil {
		xl.Errorf("setDao.Update(%#v): %v", newSet, err)
		return err
	}

	if newSet.Status == proto.SetStatusRunning {
		// if running, need to update monitor
		err = s.dispatcher.Stop(ctx, id)
		if err != nil {
			xl.Errorf("fail to stop job: %#v", set)
			return err
		}
		err = s.dispatcher.Run(ctx, &newSet)
		if err != nil {
			xl.Errorf("fail to stop job: %#v", newSet)
			return err
		}
	}

	dao.SetCache.MustSet(&newSet)
	return nil
}

func (s *Service) validateSet(xl *xlog.Logger, set *SetAddReq) error {
	// check type
	if len(set.Type) == 0 {
		xl.Error("empty type")
		return proto.ErrEmptyType
	}

	if !set.Type.IsValid() {
		xl.Errorf("invalid type: %v", set.Type)
		return proto.ErrInvalidType
	}

	// check name
	if len(set.Name) == 0 {
		xl.Error("empty name")
		return proto.ErrEmptyName
	}

	f := &dao.SetFilter{Name: set.Name}
	set2, err := dao.SetDao.Query(f)
	if err != nil {
		xl.Errorf("setDao.Query(%#v): %v", f, err)
		return err
	}
	if len(set2) > 0 && set2[0].Id != set.Id {
		xl.Errorf("name exists in other set")
		return proto.ErrExistName
	}

	// check scenes
	if len(set.Scenes) == 0 {
		xl.Error("empty scenes")
		return proto.ErrEmptyScenes
	}
	for _, scenes := range set.Scenes {
		if !scenes.IsValid() || !scenes.IsContained(s.config.Scenes) {
			xl.Errorf("invalid scene: %v", scenes)
			return proto.ErrInvalidScene
		}
	}

	// check mime types
	if len(set.MimeTypes) == 0 {
		xl.Error("empty mime_types")
		return proto.ErrEmptyMimeTypes
	}
	for _, t := range set.MimeTypes {
		if !t.IsValid() || !t.IsContained(s.config.MimeTypes) {
			xl.Errorf("invalid mime_type: %v", t)
			return proto.ErrInvalidMimeType
		}
		// check video cut interval
		if t == proto.MimeTypeVideo &&
			(set.CutIntervalMsecs < 1000 || set.CutIntervalMsecs > 60000) {
			return proto.ErrInvalidCutInterval
		}
	}

	var checkInterval, checkUri bool
	switch set.Type {
	case proto.SetTypeMonitorActive:
		checkInterval = true
		checkUri = true
	case proto.SetTypeMonitorPassive:
		checkUri = true
	default:
	}

	if checkInterval {
		// check monitor interval
		if set.MonitorInterval < MIN_MONITOR_INTERVAL {
			xl.Error("monitor interval too small")
			return proto.ErrMonitorIntervalSmall
		}
	}

	if checkUri {
		// check uri
		if len(set.Uri) == 0 {
			xl.Error("empty uri")
			return proto.ErrEmptyUri
		}

		f = &dao.SetFilter{
			Uri:  set.Uri,
			Type: set.Type,
		}
		set2, err = dao.SetDao.Query(f)
		if err != nil {
			xl.Errorf("setDao.Query(%#v): %v", f, err)
			return err
		}
		if len(set2) > 0 {
			if (len(set.Id) != 0 && set2[0].Id != set.Id) || len(set.Id) == 0 {
				xl.Errorf("uri exists in other set")
				return proto.ErrExistUri
			}
		}
	}

	return nil
}

func (s *Service) createSet(ctx context.Context, set *proto.Set) (string, error) {
	xl := xlog.FromContextSafe(ctx)

	id, err := dao.SetDao.Insert(set)
	if err != nil {
		xl.Errorf("setDao.Insert(%#v): %v", set, err)
		return "", err
	}
	set.Id = id

	// start job
	err = s.dispatcher.Run(ctx, set)
	if err != nil {
		xl.Errorf("fail to start job: %#v", set)
		// rollback
		_ = dao.SetDao.Remove(id)
		return "", err
	}

	return id, nil
}

// func (_ *Service) PostSet_Delete(
// 	ctx context.Context,
// 	req *struct{ CmdArgs []string },
// 	env *restrpc.Env,
// ) error {
// 	xl := xlog.FromContextSafe(ctx)

// 	id := req.CmdArgs[0]
// 	err := dao.SetDao.Remove(id)
// 	if err != nil {
// 		xl.Errorf("setDao.Find(%s): %v", id, err)
// 		return err
// 	}

// 	return nil
// }

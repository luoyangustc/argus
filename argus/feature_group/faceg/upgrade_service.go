package faceg

import (
	"context"
	"fmt"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/pkg/errors"
	xlog "github.com/qiniu/xlog.v1"
	"qiniu.com/argus/argus/com/util"
	"qiniu.com/argus/com/uri"
	FG "qiniu.com/argus/feature_group"
	"qiniu.com/argus/utility/evals"
	authstub "qiniu.com/auth/authstub.v1"
)

type UpgradeLimitsConfig struct {
	MaxUpgradesParallel     int32 `json:"max_upgrades_parallel"`
	MaxConcurrencyInUpgrade int32 `json:"max_concurrency_in_upgrade"`
	MaxConcurrencyInCheck   int32 `json:"max_concurrency_in_check"`
}

type UpgradeConfig struct {
	Config
	UpgradeLimitsConfig
}

type FaceGroupUpgradeService struct {
	_FaceGroupManager
	FG.FeatureAPIs

	UpgradeConfig
	FG.UpgradeInfoInMem
	ch            chan bool
	count_running int32
}

func NewFaceGroupUpgradeService(config UpgradeConfig, manager _FaceGroupManager, featureAPIs FG.FeatureAPIs) *FaceGroupUpgradeService {
	s := &FaceGroupUpgradeService{
		_FaceGroupManager: manager,
		FeatureAPIs:       featureAPIs,
		UpgradeConfig:     config,
		UpgradeInfoInMem: FG.UpgradeInfoInMem{
			Upgrades: make([]FG.Upgrade, 0),
			Mutex:    new(sync.Mutex),
		},
		ch:            make(chan bool, config.MaxUpgradesParallel),
		count_running: 0,
	}

	go s.run() // do the upgrade

	return s
}

func (s *FaceGroupUpgradeService) run() {
	var (
		xl     = xlog.NewWith("upgrade")
		ticker = time.NewTicker(time.Second * 10)
	)
	for {
		select {
		case <-ticker.C:
		case <-s.ch:
		}
		xl.Info("try to upgrade.")

		running := atomic.LoadInt32(&s.count_running)
		if running >= s.MaxUpgradesParallel {
			continue
		}

		max := int(s.MaxUpgradesParallel - running)
		idx := make([]int, 0)
		s.UpgradeInfoInMem.Lock()
		for i, j := 0, 0; i < len(s.UpgradeInfoInMem.Upgrades) && j < max; i++ {
			if s.UpgradeInfoInMem.Upgrades[i].Status == FG.UpgradeStatusWaiting {
				idx = append(idx, i)
				s.UpgradeInfoInMem.Upgrades[i].Status = FG.UpgradeStatusUpgrading
				j++
			}
		}
		s.UpgradeInfoInMem.Unlock()
		xl.Infof("execute upgrade: %d", len(idx))

		for _, i := range idx {
			atomic.AddInt32(&s.count_running, 1)
			go func(ctx context.Context, index int) {
				defer func() {
					atomic.AddInt32(&s.count_running, -1)
					s.ch <- true
				}()

				var (
					ch     = make(chan bool)
					ticker = time.NewTicker(time.Second * 10) // TODO config
				)
				go func(ctx context.Context) {
					defer ticker.Stop()
					for {
						var done = false
						select {
						case <-ticker.C:
							s.UpgradeInfoInMem.Lock()
							s.UpgradeInfoInMem.Upgrades[index].UpdatedAt = time.Now()
							s.UpgradeInfoInMem.Unlock()
						case <-ch:
							done = true
							break
						}
						if done {
							break
						}
					}
				}(ctx)

				defer func() {
					close(ch)
				}()

				err := s.runOneUpgrade(ctx, index)

				s.UpgradeInfoInMem.Lock()
				if err != nil {
					s.UpgradeInfoInMem.Upgrades[index].Error = err.Error()
				}
				s.UpgradeInfoInMem.Upgrades[index].Status = FG.UpgradeStatusFinished
				s.UpgradeInfoInMem.Unlock()

			}(context.Background(), i)
		}
	}
}

func (s *FaceGroupUpgradeService) runOneUpgrade(ctx context.Context, index int) error {
	var (
		xl  = xlog.FromContextSafe(ctx)
		err error
	)

	s.UpgradeInfoInMem.Lock()
	var upgrade = s.UpgradeInfoInMem.Upgrades[index]
	s.UpgradeInfoInMem.Unlock()

	xl.Infof("do upgrade %#v", upgrade)

	fg, err := s._FaceGroupManager.Get(ctx, upgrade.UID, upgrade.ID)
	if err != nil {
		xl.Errorf("get group failed. %s %v", upgrade.ID, err)
		return err
	}
	hid, hub := fg.Hub(ctx)

	itemsIter, err := fg.Iter(ctx)
	if err != nil {
		xl.Errorf("get face iter failed. %s %v", upgrade.ID, err)
		return err
	}
	xl.Infof("get face iter done")

	api, err := s.FeatureAPIs.Get(upgrade.To)
	if err != nil {
		xl.Errorf("get feature api(to) failed. %s %s %v", upgrade.ID, upgrade.To, err)
		return err
	}

	newHid, err := hub.New(ctx, 512*4, upgrade.To) // TODO
	if err != nil {
		return errors.Wrapf(err, "hub.New %v ", upgrade.To)
	}

	var (
		sem   = make(chan struct{}, s.MaxConcurrencyInUpgrade)
		once  sync.Once
		abort = false
	)
	defer itemsIter.Close()
	for {
		if abort { // this is not concurrency safe, however, it's not the point here.
			break
		}

		item, ok := itemsIter.Next(ctx)
		if !ok {
			break
		}

		sem <- struct{}{}
		go func(ctx context.Context, uid, utype uint32, fid, uri string) {
			defer func() {
				<-sem
			}()

			if strings.TrimSpace(uri) == "" {
				xl.Errorf("empty uri. %s", fid)
				once.Do(func() {
					err = fmt.Errorf("empty uri. %s", fid)
					abort = true
				})
				return
			}

			features, e := s.parseFace(ctx, uri, uid, utype, api.(FaceGFeatureAPI))
			if e != nil {
				once.Do(func() {
					err = e
					abort = true
				})
				return
			}

			e = hub.Set(ctx, newHid, FG.FeatureID(fid), features)
			if e != nil {
				once.Do(func() {
					err = errors.Wrapf(e, "hub.Set %v %v", newHid, FG.FeatureID(fid))
					abort = true
				})
			}
		}(util.SpawnContext(ctx), s.Config.Saver.Kodo.UID, 0, item.ID, item.Backup) // here we need to use UID from saver since qiniu:// scheme used as uri.
	}
	for i := 0; i < cap(sem); i++ { // wait for completion
		sem <- struct{}{}
	}

	if err != nil {
		xl.Errorf("drop hub. %s %s %v", upgrade.ID, upgrade.To, err)
		e := hub.Remove(ctx, newHid)
		if e != nil {
			xl.Error("remove", e, newHid)
		}
		return err
	}

	err = itemsIter.Error()
	if err != nil {
		xl.Errorf("drop hub, iter error. %s %s %v", upgrade.ID, upgrade.To, err)
		e := hub.Remove(ctx, newHid)
		if e != nil {
			xl.Error("remove", e, newHid)
		}
		return err
	}

	err = s._FaceGroupManager.Update(ctx, upgrade.UID, upgrade.ID, hid, newHid)
	if err != nil {
		xl.Errorf("upgrade HubID error. gid %v, old hid: %v, new hid: %v", upgrade.ID, hid, newHid)
		e := hub.Remove(ctx, newHid)
		if e != nil {
			xl.Error("remove", e, newHid)
		}
		return err
	}
	xl.Infof("upgrade finished. gid %v, old hid: %v, new hid: %v", upgrade.ID, hid, newHid)
	return nil
}

func (s FaceGroupUpgradeService) parseFace(
	ctx context.Context, uri string, uid, utype uint32, api FaceGFeatureAPI,
) ([]byte, error) {

	var xl = xlog.FromContextSafe(ctx)

	var req evals.SimpleReq
	req.Data.URI = uri
	dResp, err := api.IFaceDetect.Eval(ctx, req, uid, utype)
	if err != nil {
		xl.Errorf("call facex-detect error: %s %v", uri, err)
		return nil, err
	}

	xl.Infof("face detect: %#v", dResp.Result)

	if len(dResp.Result.Detections) != 1 {
		xl.Warnf("not one face: %s %d", uri, len(dResp.Result.Detections))
		return nil, errors.New("not face detected") // TODO
	}

	one := dResp.Result.Detections[0]
	var fReq evals.FaceReq
	fReq.Data.URI = uri
	fReq.Data.Attribute.Pts = one.Pts
	ff, err := api.IFaceFeature.Eval(ctx, fReq, uid, utype)
	if err != nil {
		xl.Errorf("get face feature failed. %s %v", uri, err)
		return nil, err
	}

	xl.Infof("face feature: %d", len(ff))

	return ff, nil
}

func (s *FaceGroupUpgradeService) GetFaceGroupFeature(
	ctx context.Context,
	env *authstub.Env,
) (ret struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Result  []struct {
		ID             string `json:"id"`
		FeatureVersion string `json:"feature_version"`
	} `json:"result"`
}, err error) {
	var (
		uid = env.UserInfo.Uid
	)
	ctx, xl := util.CtxAndLog(ctx, env.W, env.Req)

	ids, err := s._FaceGroupManager.All(ctx, uid)
	if err != nil {
		xl.Errorf("all group failed. %v", err)
		return
	}
	xl.Infof("all group done. %d", len(ids))

	sort.Strings(ids)
	for _, id := range ids {
		var (
			fg      _FaceGroup
			version FG.FeatureVersion
		)
		fg, err = s._FaceGroupManager.Get(ctx, uid, id)
		if err != nil {
			xl.Errorf("get group failed. %s %v", id, err)
			return
		}

		hid, hub := fg.Hub(ctx)
		version, _, err = s.getFeatureAPI(ctx, hub, hid)
		if err != nil {
			xl.Errorf("get feature api failed. %s %s %v", id, hid, err)
			return
		}
		ret.Result = append(ret.Result, struct {
			ID             string `json:"id"`
			FeatureVersion string `json:"feature_version"`
		}{
			ID:             id,
			FeatureVersion: string(version),
		})
	}
	return
}

func (s *FaceGroupUpgradeService) GetFaceGroup_Feature(
	ctx context.Context,
	req *struct {
		CmdArgs []string
	},
	env *authstub.Env,
) (ret struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Result  struct {
		ID             string `json:"id"`
		FeatureVersion string `json:"feature_version"`
	} `json:"result"`
}, err error) {
	var (
		uid = env.UserInfo.Uid
		id  = req.CmdArgs[0]
	)

	ctx, xl := util.CtxAndLog(ctx, env.W, env.Req)

	fg, err := s._FaceGroupManager.Get(ctx, uid, id)
	if err != nil {
		xl.Errorf("get group failed. %s %v", id, err)
		return
	}

	hid, hub := fg.Hub(ctx)
	version, _, err := s.getFeatureAPI(ctx, hub, hid)
	if err != nil {
		xl.Errorf("get feature api failed. %s %s %v", id, hid, err)
		return
	}
	ret.Result.ID = id
	ret.Result.FeatureVersion = string(version)
	return
}

func (s *FaceGroupUpgradeService) GetFaceGroupFeatureUpgrade(
	ctx context.Context,
	req *struct {
		Status string `json:"status"`
	},
	env *authstub.Env,
) (ret struct {
	Code    int          `json:"code"`
	Message string       `json:"message"`
	Result  []FG.Upgrade `json:"result"`
}, err error) {
	var (
		uid = env.UserInfo.Uid
	)

	status, err := FG.UpgradeStatusFromString(req.Status)
	if err != nil {
		return
	}

	s.UpgradeInfoInMem.Lock()
	defer s.UpgradeInfoInMem.Unlock()
	for _, v := range s.UpgradeInfoInMem.Upgrades {
		if v.UID == uid && v.Status == status {
			ret.Result = append(ret.Result, v)
		}
	}
	return
}

func (s *FaceGroupUpgradeService) GetFaceGroup_FeatureUpgrade(
	ctx context.Context,
	req *struct {
		CmdArgs []string
	},
	env *authstub.Env,
) (ret struct {
	Code    int          `json:"code"`
	Message string       `json:"message"`
	Result  []FG.Upgrade `json:"result"`
}, err error) {
	var (
		uid      = env.UserInfo.Uid
		id       = req.CmdArgs[0]
		upgrades []FG.Upgrade
	)

	s.UpgradeInfoInMem.Lock()
	defer s.UpgradeInfoInMem.Unlock()
	for _, v := range s.UpgradeInfoInMem.Upgrades {
		if v.UID == uid && v.ID == id {
			upgrades = append(upgrades, v)
		}
	}
	if len(upgrades) == 0 {
		err = FG.ErrUpgradeNotFound
		return
	}
	ret.Result = upgrades
	return
}

func (s *FaceGroupUpgradeService) PostFaceGroup_FeatureUpgrade_(
	ctx context.Context,
	req *struct {
		CmdArgs []string
	},
	env *authstub.Env,
) (ret struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}, err error) {
	var (
		uid   = env.UserInfo.Uid
		utype = env.UserInfo.Utype
		id    = req.CmdArgs[0]
		to    = FG.FeatureVersion(req.CmdArgs[1])
	)

	ctx, xl := util.CtxAndLog(ctx, env.W, env.Req)

	_, err = s.FeatureAPIs.Get(to)
	if err != nil {
		xl.Errorf("get feature api(to) failed. %s %s %v", id, to, err)
		return
	}

	fg, err := s._FaceGroupManager.Get(ctx, uid, id)
	if err != nil {
		xl.Errorf("get group failed. %s %v", id, err)
		return
	}

	hid, hub := fg.Hub(ctx)
	from, _, err := s.getFeatureAPI(ctx, hub, hid)
	if err != nil {
		xl.Errorf("get feature api(from) failed. %s %s %v", id, hid, err)
		return
	}

	if FG.FeatureVersionCompare(from, to) >= 0 {
		err = FG.ErrUpgradeVersionTooSmall
		return
	}

	s.UpgradeInfoInMem.Lock()
	defer s.UpgradeInfoInMem.Unlock()
	for _, v := range s.UpgradeInfoInMem.Upgrades {
		if v.UID == uid && v.ID == id && (v.Status == FG.UpgradeStatusWaiting || v.Status == FG.UpgradeStatusUpgrading) {
			err = FG.ErrUpgradeAlreadyInProgress
			return
		}
	}

	s.UpgradeInfoInMem.Upgrades = append(s.UpgradeInfoInMem.Upgrades, FG.Upgrade{
		ID:        id,
		UID:       uid,
		UType:     utype,
		From:      from,
		To:        to,
		Status:    FG.UpgradeStatusWaiting,
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	})

	ret.Message = "Upgrade on the way"
	return
}

func (s *FaceGroupUpgradeService) PostFaceGroup_FeatureRerun(
	ctx context.Context,
	req *struct {
		CmdArgs []string
	},
	env *authstub.Env,
) (ret struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}, err error) {
	var (
		uid   = env.UserInfo.Uid
		utype = env.UserInfo.Utype
		id    = req.CmdArgs[0]
	)

	ctx, xl := util.CtxAndLog(ctx, env.W, env.Req)

	fg, err := s._FaceGroupManager.Get(ctx, uid, id)
	if err != nil {
		xl.Errorf("get group failed. %s %v", id, err)
		return
	}

	hid, hub := fg.Hub(ctx)
	from, _, err := s.getFeatureAPI(ctx, hub, hid)
	if err != nil {
		xl.Errorf("get feature api(from) failed. %s %s %v", id, hid, err)
		return
	}

	s.UpgradeInfoInMem.Lock()
	defer s.UpgradeInfoInMem.Unlock()
	for _, v := range s.UpgradeInfoInMem.Upgrades {
		if v.UID == uid && v.ID == id && (v.Status == FG.UpgradeStatusWaiting || v.Status == FG.UpgradeStatusUpgrading) {
			err = FG.ErrUpgradeAlreadyInProgress
			return
		}
	}

	s.UpgradeInfoInMem.Upgrades = append(s.UpgradeInfoInMem.Upgrades, FG.Upgrade{
		ID:        id,
		UID:       uid,
		UType:     utype,
		From:      from,
		To:        from,
		Status:    FG.UpgradeStatusWaiting,
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	})

	ret.Message = "Rerun on the way"
	return
}

func (s *FaceGroupUpgradeService) PostFaceGroup_FeatureCheck(
	ctx context.Context,
	req *struct {
		CmdArgs []string
	},
	env *authstub.Env,
) (ret struct {
	Code    int            `json:"code"`
	Message string         `json:"message"`
	Result  FG.CheckResult `json:"result"`
}, err error) {
	var (
		uid = env.UserInfo.Uid
		id  = req.CmdArgs[0]
	)

	ctx, xl := util.CtxAndLog(ctx, env.W, env.Req)

	fg, err := s._FaceGroupManager.Get(ctx, uid, id)
	if err != nil {
		xl.Errorf("get group failed. %s %v", id, err)
		return
	}

	hid, hub := fg.Hub(ctx)
	version, _, err := s.getFeatureAPI(ctx, hub, hid)
	if err != nil {
		xl.Errorf("get feature api failed. %s %s %v", id, hid, err)
		return
	}

	itemsIter, err := fg.Iter(ctx)
	if err != nil {
		xl.Errorf("get face iter failed. %s %v", id, err)
		return
	}

	c := uri.New(uri.WithUserAkSk(
		s.UpgradeConfig.Config.Saver.Kodo.Config.AccessKey,
		s.UpgradeConfig.Config.Saver.Kodo.Config.SecretKey,
		s.UpgradeConfig.Config.Saver.Kodo.Config.RSHost,
	))

	var (
		sem                    = make(chan struct{}, s.MaxConcurrencyInCheck)
		available, unavailable int32
	)
	defer itemsIter.Close()
	for {
		item, ok := itemsIter.Next(ctx)
		if !ok {
			break
		}

		u := item.Backup
		if len(u) == 0 {
			atomic.AddInt32(&unavailable, 1)
			continue
		}

		sem <- struct{}{}
		go func(_u string) {
			defer func() {
				<-sem
			}()
			resp, _err := c.Get(ctx, uri.Request{URI: _u})
			if _err != nil {
				xl.Warnf("CHECK uri.Get fail. %v %v", _u, _err)
				atomic.AddInt32(&unavailable, 1)
				return
			}
			resp.Body.Close()
			atomic.AddInt32(&available, 1)
		}(u)
	}
	for i := 0; i < cap(sem); i++ { // wait for completion
		sem <- struct{}{}
	}

	ret.Result = FG.CheckResult{
		ID:          id,
		Version:     version,
		Available:   int(available),
		Unavailable: int(unavailable),
	}
	return
}

func (s *FaceGroupUpgradeService) getFeatureAPI(ctx context.Context, hub FG.Hub, hid FG.HubID) (FG.FeatureVersion, FG.FeatureAPI, error) {
	fv, err := hub.FeatureVersion(ctx, hid)
	if err != nil {
		return FG.EmptyFeatureVersion, nil, err
	}

	if len(fv) == 0 { // legacy group
		return s.FeatureAPIs.Default()
	}

	api, err := s.FeatureAPIs.Get(fv)
	return fv, api, err
}

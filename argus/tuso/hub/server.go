package hub

import (
	"context"
	"fmt"
	"regexp"
	"time"

	"github.com/qiniu/db/mgoutil.v3"

	"gopkg.in/mgo.v2"
	"gopkg.in/mgo.v2/bson"
	authstub "qiniu.com/auth/authstub.v1"

	"github.com/pkg/errors"
	"github.com/qiniu/http/httputil.v1"
	"qiniu.com/argus/tuso/client/image_feature"
	"qiniu.com/argus/tuso/client/job_gate"
	"qiniu.com/argus/tuso/proto"
)

var ErrNoImplement = httputil.NewError(400, "api no implement")
var ErrHubExists = httputil.NewError(400, "hub exists")
var ErrHubNotFound = httputil.NewError(400, "hub not found")
var ErrBadStatus = httputil.NewError(400, fmt.Sprintf("bad status, should %v", OptatusEnum))
var ErrBadOp = httputil.NewError(400, fmt.Sprintf("bad op, should %v", OpKindEnum))

type Config struct {
	Mgo            mgoutil.Config                 `json:"mgo"`
	Kodo           KodoConfig                     `json:"kodo"`
	FeatureApi     image_feature.FeatureApiConfig `json:"feature_api"`
	JobGateApi     JobGateApiConfig               `json:"jobgate_api"`
	ConcurrencyNum int                            `json:"concurrency_num"`
}

type JobGateApiConfig struct {
	Host          string `json:"host"`
	TimeoutSecond int    `json:"timeout_second"`
}

func New(cfg Config) (*server, *internalServer, *opLogProcess, error) {
	db := new(db)
	s := &server{
		db: db,
	}
	ins := &internalServer{
		db: db,
	}
	_, err := mgoutil.Open(db, &cfg.Mgo)
	if err != nil {
		return nil, nil, nil, errors.Wrap(err, "mgoutil.Open")
	}
	s.db.createIndex()

	s.job_gate = job_gate.NewHack(0, cfg.JobGateApi.Host, time.Second*time.Duration(cfg.JobGateApi.TimeoutSecond))
	m := image_feature.NewFeatureApi(cfg.FeatureApi)

	uploader := &kodoUploader{
		cfg: cfg.Kodo,
	}
	o := &opLogProcess{
		db:             db,
		api:            m,
		concurrencyNum: cfg.ConcurrencyNum,
		uploader:       uploader,
		batchSize:      2000,
	}
	return s, ins, o, nil
}

type server struct {
	db       *db
	job_gate *job_gate.Client
}

func (s *server) PostImage(context context.Context, req *proto.PostImageReq, env *authstub.Env) (*proto.PostImageResp, error) {
	if err := s.isHubOwner(env, req.Hub); err != nil {
		return nil, err
	}
	op, err := OpKindEnumFromString(req.Op)
	if err != nil {
		return nil, err
	}
	if op != OpKindAdd {
		return nil, ErrNoImplement
	}
	// 检查文件是否存在，是否是图片?
	// 最大图片张数
	resp := new(proto.PostImageResp)
	for _, image := range req.Images {
		ok := true
		err := s.db.insertOpLog(dOpLog{
			Op:         op,
			Status:     OptatusInit,
			HubName:    req.Hub,
			Key:        image.Key,
			CreateTime: time.Now(),
		})
		if err != nil {
			return nil, errors.Wrap(err, "db.OpLog.Insert")
		}
		switch op {
		case OpKindAdd:
			err := s.db.FileMeta.Insert(dFileMeta{
				HubName:           req.Hub,
				Key:               image.Key,
				UpdateTime:        time.Now(),
				Status:            FileMetaStatusInit,
				FeatureFileIndex:  -1,
				FeatureFileOffset: -1,
			})
			if mgo.IsDup(err) {
				ok = false
				resp.ExistsCnt++
			} else if err != nil {
				return nil, errors.Wrap(err, "db.fileMeta.Insert")
			}
		case OpKindUpdate:
			// TODO: 支持oplog 更新、删除op
			var f dFileMeta
			err := s.db.FileMeta.Find(bson.M{"hub_name": req.Hub, "key": image.Key}).One(&f)
			switch err {
			case nil:
				err = s.db.FileMeta.UpdateId(f.ID, bson.M{"$set": bson.M{"status": FileMetaStatusInit, "update_time": time.Now()}})
				if err != nil {
					return nil, errors.Wrap(err, "db.fileMeta.UpdateId")
				}
			case mgo.ErrNotFound:
				ok = false
				resp.NotFoundCnt++
			default:
				return nil, errors.Wrap(err, "db.fileMeta.Find")
			}
		case OpKindDelete:
			var f dFileMeta
			err := s.db.FileMeta.Find(bson.M{"hub_name": req.Hub, "key": image.Key}).One(&f)
			switch err {
			case nil:
				err = s.db.FileMeta.UpdateId(f.ID, bson.M{"$set": bson.M{"status": FileMetaStatusDeleted, "update_time": time.Now()}})
				if err != nil {
					return nil, errors.Wrap(err, "db.fileMeta.UpdateId")
				}
			case mgo.ErrNotFound:
				ok = false
				resp.NotFoundCnt++
			default:
				return nil, errors.Wrap(err, "db.fileMeta.Find")
			}
		}
		if ok {
			resp.SuccessCnt++
		}
	}
	return resp, nil
}

func (s *server) PostSearchJob(ctx context.Context, req *proto.PostSearchJobReq, env *authstub.Env) (*proto.PostSearchJobResp, error) {
	if err := s.isHubOwner(env, req.Hub); err != nil {
		return nil, err
	}
	hubInfo, err := s.db.findHubInfo(req.Hub)
	if err != nil {
		return nil, err
	}
	hubMeta, err := s.db.findHubMeta(req.Hub)
	if err != nil {
		return nil, err
	}
	images := make([]proto.Image, 0)
	for _, v := range req.Images {
		img := proto.Image{
			Uid: env.Uid,
		}
		if v.Url != "" {
			img.Url = v.Url
		} else if v.Key != "" {
			img.Key = v.Key
			img.Bucket = hubInfo.Bucket
		} else {
			return nil, httputil.NewError(400, fmt.Sprintf("bad img: %v", v))
		}
		images = append(images, img)
	}

	r := proto.PostSearchJobReqJob{
		Images:      images,
		Hub:         req.Hub,
		Version:     hubMeta.FeatureVersion,
		TopN:        req.TopN,
		Threshold:   req.Threshold,
		Kind:        req.Kind,
		CallBackURL: req.CallBackURL,
	}

	resp, err := s.job_gate.PostSubmitTusoSearch(ctx, &job_gate.PostSubmitTusoSearchReq{Request: r}, nil)
	if err != nil {
		return nil, errors.Wrap(err, "job_gate.PostSubmitTusoSearch")
	}
	return resp, nil
}

func (s *server) GetSearchJob(ctx context.Context, req *proto.GetSearchJobReq, env *authstub.Env) (*proto.GetSearchJobResp, error) {
	resp, err := s.job_gate.GetQueryTusoSearch(ctx, req, nil)
	if err != nil {
		return nil, errors.Wrap(err, "job_gate.PostSubmitTusoSearch")
	}
	return &resp.Response, nil
}

func checkPostHubReq(req *proto.PostHubReq) (bool, string) {
	if req.Name == "" {
		return false, "The 'name' field cannot be empty"
	}
	if !(len(req.Name) >= 4 && len(req.Name) <= 20) {
		return false, "The hub's name must be large than 4 characters and less than 20 characters"
	}
	if !regexp.MustCompile(`^[a-z][a-z0-9_-]{3,19}$`).MatchString(req.Name) {
		return false, "Invalid hub name"
	}
	return true, ""
}

func (s *server) PostHub(context context.Context, req *proto.PostHubReq, env *authstub.Env) error {
	if valid, msg := checkPostHubReq(req); !valid {
		// HTTP code
		return httputil.NewError(400, msg)
	}
	// TODO: check bucket exists
	err := s.db.createHub(dHub{
		HubName: req.Name,
		UID:     env.Uid,
		Bucket:  req.Bucket,
		Prefix:  req.Prefix,
	})
	if mgo.IsDup(err) {
		return ErrHubExists
	}
	err = s.db.createHubMeta(dHubMeta{
		HubName:          req.Name,
		FeatureVersion:   proto.DefaultFeatureVersion,
		FeatureFileIndex: 0,
	})
	return err
}

func (s *server) GetHubs(ctx context.Context, req *proto.GetHubsReq, env *authstub.Env) (resp *proto.GetHubsResp, err error) {
	r, err := s.db.listHubByUID(env.Uid)
	if err != nil {
		return nil, errors.Wrap(err, "db.listHubByUID")
	}
	resp = &proto.GetHubsResp{}
	for _, v := range r {
		resp.Hubs = append(resp.Hubs, proto.Hub{
			HubName: v.HubName,
			Bucket:  v.Bucket,
			Prefix:  v.Prefix,
		})
	}
	return
}

func (s *server) GetHub(ctx context.Context, req *proto.GetHubReq, env *authstub.Env) (resp *proto.GetHubResp, err error) {
	if err := s.isHubOwner(env, req.Hub); err != nil {
		return nil, err
	}

	hubInfo, err := s.db.findHubInfo(req.Hub)
	if err != nil {
		return nil, err
	}
	n, err := s.db.countFileMetaByUid(req.Hub)
	if err != nil {
		return nil, err
	}
	resp = &proto.GetHubResp{
		HubName: hubInfo.HubName,
		Bucket:  hubInfo.Bucket,
		Prefix:  hubInfo.Prefix,
		Stat: proto.GetHubRespStat{
			ImageNum: n,
		},
	}
	return
}

func (s *server) isHubOwner(env *authstub.Env, hub string) error {
	return s.db.isHubOwner(env.Uid, hub)
}

var _ proto.UserApi = new(server)

package mock

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"time"

	mgoutil "github.com/qiniu/db/mgoutil.v3"
	httputil "github.com/qiniu/http/httputil.v1"
	restrpc "github.com/qiniu/http/restrpc.v1"
	"github.com/qiniu/xlog.v1"
	mgo "gopkg.in/mgo.v2"
)

const (
	defaultCollSessionPoolLimit = 100
)

type M map[string]interface{}

type Config struct {
	MgoConfig            mgoutil.Config `json:"mgo_config"`
	CollSessionPoolLimit int            "coll_session_pool_limit"
}

type _Collections struct {
	Captures mgoutil.Collection `coll:"captures"`
}

type Service struct {
	captureColl *mgoutil.Collection
}

var (
	ErrCaptureExist = httputil.NewError(http.StatusBadRequest, "capture is already exist")
)

func New(cfg Config) (*Service, error) {
	collections := _Collections{}
	mgoSession, err := mgoutil.Open(&collections, &cfg.MgoConfig)
	if err != nil {
		return nil, err
	}

	// ensure index
	if err = collections.Captures.EnsureIndex(mgo.Index{Key: []string{"id"}, Unique: true}); err != nil {
		return nil, fmt.Errorf("groups collections ensure index id err: %s", err.Error())
	}
	if err = collections.Captures.EnsureIndex(mgo.Index{Key: []string{"licence_id"}}); err != nil {
		return nil, fmt.Errorf("groups collections ensure index licence_id err: %s", err.Error())
	}
	if err = collections.Captures.EnsureIndex(mgo.Index{Key: []string{"time"}}); err != nil {
		return nil, fmt.Errorf("groups collections ensure index licence_id err: %s", err.Error())
	}

	if cfg.CollSessionPoolLimit == 0 {
		cfg.CollSessionPoolLimit = defaultCollSessionPoolLimit
	}

	mgoSession.SetPoolLimit(cfg.CollSessionPoolLimit)

	return &Service{
		captureColl: &collections.Captures,
	}, nil
}

type Image struct {
	URI string `json:"uri" bson:"uri"`
	PTS [4]int `json:"pts" bson:"pts"`
}
type Capture struct {
	ID          string    `bson:"id"`
	Time        time.Time `bson:"time"`
	CameraID    string    `bson:"camera_id"`
	CameraInfo  string    `bson:"camera_info"`
	LicenceID   string    `bson:"licence_id"`
	LicenceType string    `bson:"licence_type"`
	Lane        int       `bson:"lane"`
	Result      int       `bson:"result"`
	Score       float32   `bson:"score"`
	Coordinate  struct {
		GPS [2]float32 `bson:"gps"`
	} `bson:"coordinate"`
	Resorce struct {
		Images []Image  `bson:"images"`
		Videos []string `bson:"videos"`
	} `bson:"resource"`
}

type postZhatucheCapture_Req struct {
	CmdArgs     []string
	Time        string  `json:"time"`
	CameraID    string  `json:"camera_id"`
	CameraInfo  string  `json:"camera_info"`
	LicenceID   string  `json:"licence_id"`
	LicenceType string  `json:"licence_type"`
	Lane        int     `json:"lane"`
	Result      int     `json:"result"`
	Score       float32 `json:"score"`
	Coordinate  struct {
		GPS [2]float32 `json:"gps"`
	} `json:"coordinate"`
	Resorce struct {
		Images []Image  `json:"images"`
		Videos []string `json:"videos"`
	} `json:"resource"`
}

func (s *Service) PostZhatucheCapture_(ctx context.Context, args *postZhatucheCapture_Req, env *restrpc.Env) (err error) {
	var (
		id = args.CmdArgs[0]
	)

	if err = s.captureColl.Find(M{"id": id}).One(nil); err == nil {
		return ErrCaptureExist
	}

	if err != mgo.ErrNotFound {
		return httputil.NewError(http.StatusInternalServerError, fmt.Sprintf("fail to add capture, check capture if exist failed due to db error: %s", err.Error()))
	}

	tm, _ := time.Parse("20060102150405", args.Time)
	cap := Capture{
		ID:          id,
		Time:        tm,
		CameraID:    args.CameraID,
		CameraInfo:  args.CameraInfo,
		LicenceID:   args.LicenceID,
		LicenceType: args.LicenceType,
		Lane:        args.Lane,
		Result:      args.Result,
		Score:       args.Score,
	}
	cap.Coordinate.GPS = args.Coordinate.GPS
	cap.Resorce.Images = args.Resorce.Images
	cap.Resorce.Videos = args.Resorce.Videos

	if err = s.captureColl.Insert(cap); err != nil {
		return httputil.NewError(http.StatusInternalServerError, fmt.Sprintf("fail to add capture, insert capture failed due to db error: %s", err.Error()))
	}

	xlog.FromContextSafe(ctx).Debugf("Got capture: %#v", cap)

	return
}

// -----------------------------------------------------------------
type getZhatucheCaptureReq struct {
	CmdArgs []string
}

type getZhatucheCaptureResp struct {
	ID          string  `json:"id"`
	Time        string  `json:"time"`
	CameraID    string  `json:"camera_id"`
	CameraInfo  string  `json:"camera_info"`
	LicenceID   string  `json:"licence_id"`
	LicenceType string  `json:"licence_type"`
	Lane        int     `json:"lane"`
	Result      int     `json:"result"`
	Score       float32 `json:"score"`
	Coordinate  struct {
		GPS [2]float32 `json:"gps"`
	} `json:"coordinate"`
	Resorce struct {
		Images []Image  `json:"images"`
		Videos []string `json:"videos"`
	} `json:"resource"`
}

func (s *Service) GetZhatucheStart_End_(ctx context.Context, args *getZhatucheCaptureReq, env *restrpc.Env) (resp []getZhatucheCaptureResp, err error) {

	var (
		start, end time.Time
	)
	if start, err = time.Parse("20060102150405", args.CmdArgs[0]); err != nil {
		return nil, httputil.NewError(http.StatusBadRequest, "invaid start time")
	}
	if end, err = time.Parse("20060102150405", args.CmdArgs[1]); err != nil {
		return nil, httputil.NewError(http.StatusBadRequest, "invaid end time")
	}

	var captures []Capture
	if err = s.captureColl.Find(M{"time": M{"$gt": start, "$lt": end}}).All(&captures); err != nil {
		err = httputil.NewError(http.StatusInternalServerError, fmt.Sprintf("fail to get captures, get captures failed due to db error: %s", err.Error()))
		return
	}

	for _, cap := range captures {
		res := getZhatucheCaptureResp{
			ID:          cap.ID,
			Time:        cap.Time.Add((-8) * time.Hour).Format("20060102150405"),
			CameraID:    cap.CameraID,
			CameraInfo:  cap.CameraInfo,
			LicenceID:   cap.LicenceID,
			LicenceType: cap.LicenceType,
			Lane:        cap.Lane,
			Result:      cap.Result,
			Score:       cap.Score,
		}
		res.Coordinate.GPS = cap.Coordinate.GPS
		res.Resorce.Images = cap.Resorce.Images
		res.Resorce.Videos = cap.Resorce.Videos
		resp = append(resp, res)
	}

	return
}

// -----------------------------------------------------------------------------------------------
type postZhatucheReportReq struct {
	CmdArgs []string
	Start   string `json:"start"`
	End     string `json:"end"`
	Illegal bool   `json:"illegal"`
}

func (s *Service) PostZhatucheReport(ctx context.Context, args *postZhatucheReportReq, env *restrpc.Env) {
	var (
		start, end time.Time
		err        error
	)
	if start, err = time.Parse("20060102150405", args.Start); err != nil {
		httputil.ReplyErr(env.W, http.StatusBadRequest, "invaid start time")
		return
	}
	if end, err = time.Parse("20060102150405", args.End); err != nil {
		httputil.ReplyErr(env.W, http.StatusBadRequest, "invaid end time")
		return
	}

	var (
		captures []Capture
		query    M
	)
	if args.Illegal {
		query = M{"time": M{"$gt": start, "$lt": end}, "result": 1}
	} else {
		query = M{"time": M{"$gt": start, "$lt": end}}
	}
	if err = s.captureColl.Find(query).All(&captures); err != nil {
		httputil.ReplyErr(env.W, http.StatusInternalServerError, fmt.Sprintf("fail to get captures, get captures failed due to db error: %s", err.Error()))
		return
	}

	var resp string
	for _, cap := range captures {
		res := getZhatucheCaptureResp{
			ID:          cap.ID,
			Time:        cap.Time.Add((-8) * time.Hour).Format("20060102150405"),
			CameraID:    cap.CameraID,
			CameraInfo:  cap.CameraInfo,
			LicenceID:   cap.LicenceID,
			LicenceType: cap.LicenceType,
			Lane:        cap.Lane,
			Result:      cap.Result,
			Score:       cap.Score,
		}
		res.Coordinate.GPS = cap.Coordinate.GPS
		res.Resorce.Images = cap.Resorce.Images
		res.Resorce.Videos = cap.Resorce.Videos
		js, _ := json.Marshal(res)
		resp = resp + string(js) + "\n"
	}

	env.W.Header().Set("Content-Length", strconv.Itoa(len(resp)))
	env.W.Header().Set("Content-Type", "plain/text")
	env.W.WriteHeader(http.StatusOK)
	env.W.Write([]byte(resp))
	return
}

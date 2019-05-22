package service

import (
	"context"
	"net/http"
	"time"

	httputil "github.com/qiniu/http/httputil.v1"
	"github.com/qiniu/http/restrpc.v1"
	"github.com/qiniu/xlog.v1"
)

const (
	licenceTypeYellow = "01"
)

type Config struct {
}

type Service struct {
	Config
	*Manager
}

func New(c Config, mgr *Manager) (*Service, error) {
	srv := &Service{
		Config:  c,
		Manager: mgr,
	}

	return srv, nil
}

func (s *Service) initContext(ctx context.Context, env *restrpc.Env) (*xlog.Logger, context.Context) {
	xl, ok := xlog.FromContext(ctx)
	if !ok {
		xl = xlog.New(env.W, env.Req)
		ctx = xlog.NewContext(ctx, xl)
	}
	return xl, ctx
}

// -----------------------------------------------------------------------------------
type postCapture_Req struct {
	CmdArgs      []string
	Time         string `json:"time"`
	CameraID     string `json:"camera_id,omitempty"`
	CameraInfo   string `json:"camera_info,,omitempty"`
	LicenceID    string `json:"licence_id"`
	LicenceType  string `json:"licence_type"`
	Attribution  string `json:"attribution,omitempty"`
	Lane         int    `json:"lane"`
	LicenceImage string `json:"licence_image"`
	FullImage    string `json:"full_image"`
}

func (s *Service) PostCapture_(ctx context.Context, args *postCapture_Req, env *restrpc.Env) (err error) {

	xl, ctx := s.initContext(ctx, env)

	var (
		capture = Capture{
			ID:          args.CmdArgs[0],
			CameraID:    args.CameraID,
			CameraInfo:  args.CameraInfo,
			LicenceID:   args.LicenceID,
			LicenceType: args.LicenceType,
			Lane:        args.Lane,
		}
	)

	if capture.LicenceType != licenceTypeYellow {
		return
	}

	capture.Time = args.Time
	if len(args.LicenceImage) > 0 {
		capture.Resource.CaptureImages = append(capture.Resource.CaptureImages, Image{URI: args.LicenceImage})
	}
	if len(args.FullImage) > 0 {
		capture.Resource.CaptureImages = append(capture.Resource.CaptureImages, Image{URI: args.FullImage})
	}

	if len(capture.Resource.CaptureImages) > 0 {
		xl.Debugf("Add capture: %#v", capture)
		s.Produce(capture)
	}

	return
}

// -----------------------------------------------------------------------------------

type postCaptureReq struct {
	CmdArgs   []string
	StartTime string `json:"start_time"`
	CameraID  string `json:"camera_id"`
	CameraIP  string `json:"camera_ip"`
	Duration  int    `json:"duration"`
}

func (s *Service) PostCapture(ctx context.Context, args *postCaptureReq, env *restrpc.Env) (err error) {
	return s.Capture(ctx, args.StartTime, args.CameraID, args.CameraIP, args.Duration)
}

// ------------------------------------------------------------------
type postVideoTimeReq struct {
	CmdArgs []string
	Time    string `json:"time"`
}

func (s *Service) PostVideoTime(ctx context.Context, args *postVideoTimeReq, env *restrpc.Env) (err error) {

	tm, err := time.Parse("20060102150405", args.Time)
	if err != nil {
		return httputil.NewError(http.StatusBadRequest, "invalid time")
	}

	for _, camera := range s.Cameras {
		_, err = s.Fetch(ctx, tm, camera.DvrConfig)
		if err != nil {
			return
		}
	}
	return
}

// ------------------------------------------------------------------
type postCaptureArchiveReq struct {
	CmdArgs   []string
	StartTime string `json:"start_time"`
	EndTime   string `json:"end_time"`
	Illegal   bool   `json:"illegal"`
}

func (s *Service) PostCaptureArchive(ctx context.Context, args *postCaptureArchiveReq, env *restrpc.Env) (err error) {
	return s.Archive(ctx, args.StartTime, args.EndTime, args.Illegal)
}

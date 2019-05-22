package main

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"regexp"
	"strings"
	"time"

	httputil "github.com/qiniu/http/httputil.v1"
	restrpc "github.com/qiniu/http/restrpc.v1"
	xlog "github.com/qiniu/xlog.v1"
	"qiniu.com/argus/AIProjects/tianyan/serving"
)

const (
	_DEFAULT_CHECK_INTEVAL = 10
)

type Service struct {
	conf Config

	aiService AIService
	storage   *Storage
}

type AIService struct {
	url     string
	timeout time.Duration
}

func (ai AIService) eval(ctx context.Context, method, uri string, req interface{}, resp interface{}) error {
	client := serving.NewDefaultStubRPCClient(ai.timeout)
	f := func(ctx context.Context) error {
		var err1 error
		err1 = client.CallWithJson(ctx, resp, method, ai.url+uri, req)
		return err1
	}
	return serving.CallWithRetry(ctx, []int{530}, []func(context.Context) error{f, f})
}

func (ai AIService) CreateCluster(
	ctx context.Context, name string, size, precision, dimension int, timeout int,
) (err error) {
	uri := fmt.Sprintf("/cluster/%s/new", name)
	req := struct {
		Size      int `json:"size"`
		Precision int `json:"precision"`
		Dimension int `json:"dimension"`
		Timeout   int `json:"timeout"`
	}{Size: size, Precision: precision, Dimension: dimension, Timeout: timeout}
	return ai.eval(ctx, "POST", uri, &req, nil)
}

func (ai AIService) DestroyCluster(
	ctx context.Context, name string,
) (err error) {
	uri := fmt.Sprintf("/cluster/%s/remove", name)
	return ai.eval(ctx, "POST", uri, nil, nil)
}

func NewService(c Config) (srv *Service, err error) {
	srv = &Service{conf: c}
	srv.aiService = AIService{url: c.AIHost + "/v1/face", timeout: 10 * time.Second}
	srv.storage, err = NewStorage(c.MgoConfig)
	ctx := xlog.NewContext(context.Background(), xlog.NewDummy())
	xl, ctx := srv.initContext(ctx, nil)
	if err != nil {
		xl.Errorf("failed to NewStorage: %v", err)
		return
	}
	err = srv.initService(ctx)
	if err != nil {
		xl.Errorf("failed to initService: %v", err)
		return
	}
	return srv, nil
}

type FfmpegInfo struct {
	CameraID              string `bson:"camera_id"`
	UpstreamAddress       string `bson:"upstream_address"`
	DownstreamAddress     string `bson:"downstream_address"`
	CheckInterval         int    `bson:"check_interval"`
	ClusterID             string `bson:"cluser_id"`
	ClusterFeatureTimeout int    `bson:cluster_feature_timeout`
	AiURL                 string `bson:"ai_url"`
	Pid                   int    `bson:"pid"`
}

func (s *Service) initService(ctx context.Context) error {
	xl, ctx := s.initContext(ctx, nil)
	fInfos, err := s.storage.All()
	if err != nil {
		xl.Error("Initing", "Failed to get ffmpegInfos from storage")
		return err
	}
	for _, fInfo := range fInfos {
		args := s.buildFfmpegArgs(ctx, nil, fInfo)
		stdout, err := os.Create(fmt.Sprintf("out-%v.log", fInfo.CameraID))
		if err != nil {
			xl.Warnf("Camera %v process failed create log file", fInfo.CameraID)
			return errors.New("Camera process failed to start")
		}

		stderr, err := os.Create(fmt.Sprintf("err-%v.log", fInfo.CameraID))
		if err != nil {
			xl.Warnf("Camera %v process failed create log file", fInfo.CameraID)
			return errors.New("Camera process failed to start")
		}
		cmd := exec.Command(s.conf.FfmpegCmd, args...)

		if fInfo.ClusterID != "" {
			err = s.createCluster(ctx, nil, fInfo.ClusterID, fInfo.ClusterFeatureTimeout)
			if err != nil {
				xl.Error("Initing", "Failed to init cluster")
				return err
			}
		}

		cmd.Stdout = stdout
		cmd.Stderr = stderr
		cmd.Start()
		fInfo.Pid = cmd.Process.Pid
		s.storage.Update(fInfo.CameraID, fInfo)
		xl.Infof("Starting camera %v process %v", fInfo.CameraID, fInfo.Pid)
	}
	return nil
}

func (s *Service) initContext(ctx context.Context, env *restrpc.Env) (*xlog.Logger, context.Context) {
	xl, ok := xlog.FromContext(ctx)
	if !ok {
		xl = xlog.New(env.W, env.Req)
		ctx = xlog.NewContext(ctx, xl)
	}
	return xl, ctx
}

// Service API
type StreamStartReq struct {
	CmdArgs               []string
	UpstreamAddress       string   `json:"upstream_address"`
	Groups                []string `json:"groups"`
	CheckInterval         int      `json:"check_interval"`
	EnableClusterSearch   bool     `json:"enable_cluster_search"`
	ClusterFeatureTimeout int      `json:"cluster_feature_timeout"`
	ForceStart            bool     `json:"force"`
}
type StreamStartResp struct {
	DownstreamAddress string `json:"downstream_address"`
	ClusterID         string `json:"cluster_id,omitempty"`
}

func (s *Service) PostCamera_Start(ctx context.Context, args *StreamStartReq, env *restrpc.Env) (resp StreamStartResp, err error) {
	xl, ctx := s.initContext(ctx, env)
	cameraID := args.CmdArgs[0]
	if !regexp.MustCompile(`[A-Za-z0-9]{3,32}$`).MatchString(cameraID) {
		err = httputil.NewError(http.StatusBadRequest, "invalid camera ID")
		return
	}
	if len(args.Groups) == 0 && !args.EnableClusterSearch {
		err = httputil.NewError(http.StatusBadRequest, "invalid groups params")
		return
	}
	if args.UpstreamAddress == "" {
		err = httputil.NewError(http.StatusBadRequest, "invalid groups upstream_address")
		return
	}
	ffmpegInfo := FfmpegInfo{
		CameraID:              cameraID,
		UpstreamAddress:       args.UpstreamAddress,
		CheckInterval:         args.CheckInterval,
		ClusterFeatureTimeout: args.ClusterFeatureTimeout,
		DownstreamAddress:     fmt.Sprintf("%s/%s", s.conf.DownstreamHost, cameraID),
	}
	if ffmpegInfo.CheckInterval == 0 {
		ffmpegInfo.CheckInterval = _DEFAULT_CHECK_INTEVAL
	}

	categoryStr := "group" // group search api is /face/group/<group_ids>/search
	groups := args.Groups
	if args.EnableClusterSearch {
		ffmpegInfo.ClusterID = "cluster000" + cameraID
		groups = append([]string{ffmpegInfo.ClusterID}, groups...)
		categoryStr = "cluster" // cluster search api is /face/cluster/<cluser_id,group_ids>/search
	}
	groupsStr := strings.Join(groups, ",")
	ffmpegInfo.AiURL = fmt.Sprintf("%s/v1/face/%s/%s/search", s.conf.AIHost, categoryStr, groupsStr)

	e := s.startFfmpegProcess(ctx, env, ffmpegInfo, args.ForceStart)
	if e != nil {
		err = httputil.NewError(http.StatusBadRequest, e.Error())
		return
	}
	xl.Debugf("camera: %v started", cameraID)
	resp = StreamStartResp{
		DownstreamAddress: ffmpegInfo.DownstreamAddress,
		ClusterID:         ffmpegInfo.ClusterID,
	}
	return
}

func (s *Service) PostCamera_Stop(ctx context.Context, args *struct{ CmdArgs []string }, env *restrpc.Env) error {
	xl, ctx := s.initContext(ctx, env)
	cameraID := args.CmdArgs[0]
	err := s.stopFfmpegProcess(ctx, env, cameraID)
	if err != nil {
		return httputil.NewError(http.StatusBadRequest, err.Error())
	}
	xl.Debugf("camera: %v stoped", cameraID)
	return nil
}

func (s *Service) createCluster(ctx context.Context, env *restrpc.Env, clusterID string, clusterFeatureTimeout int) error {
	xl, ctx := s.initContext(ctx, env)
	args := s.conf.ClusterArgs
	err := s.aiService.CreateCluster(ctx, clusterID, args.Size, args.Precision, args.Dimension, clusterFeatureTimeout)

	if err != nil {
		if strings.Contains(err.Error(), "is already exist") {
			e := s.destroyCluster(ctx, env, clusterID)
			if e != nil {
				xl.Errorf("fail to destory cluster before create feature cluster %v, err: %v", clusterID, err)
				return e
			}
			return s.createCluster(ctx, env, clusterID, clusterFeatureTimeout)
		}
		xl.Errorf("fail to create feature cluster %s, size: %d, precision: %d, dimension: %d, err: %v", clusterID, args.Size, args.Precision, args.Dimension, err)
		return err
	}
	xl.Debugf("Create feature cluster %s, size: %d, precision: %d, dimension: %d", clusterID, args.Size, args.Precision, args.Dimension)
	return nil
}

func (s *Service) destroyCluster(ctx context.Context, env *restrpc.Env, clusterID string) error {
	xl, ctx := s.initContext(ctx, env)
	if err := s.aiService.DestroyCluster(ctx, clusterID); err != nil {
		if !strings.Contains(err.Error(), "not found") {
			xl.Errorf("PostFaceCluster_Remove: call feature-search.Destroy failed, error: %v", err)
			return err
		}
	}
	return nil
}

func (s *Service) buildFfmpegArgs(ctx context.Context, env *restrpc.Env, fInfo FfmpegInfo) []string {
	xl, _ := s.initContext(ctx, env)
	ffmpegArgs := fmt.Sprintf(s.conf.FfmpegArgsTpl, fInfo.UpstreamAddress, fInfo.CheckInterval, fInfo.AiURL, fInfo.DownstreamAddress)
	xl.Debugf("Camera proceess start args: %s", ffmpegArgs)
	return strings.Split(ffmpegArgs, " ")
}

func (s *Service) startFfmpegProcess(ctx context.Context, env *restrpc.Env, fInfo FfmpegInfo, forceStart bool) error {
	xl, ctx := s.initContext(ctx, env)
	count, err := s.storage.Count()
	cameraID := fInfo.CameraID
	if err != nil {
		xl.Warnf("Failed to count process num, camera %s failed to start", cameraID)
		return errors.New("Camera failed to check limitation")
	}
	if count >= s.conf.MaxProcessNum {
		xl.Warnf("Camera hit limitation, camera %s failed to start", cameraID)
		return errors.New("Camera hit limitation, cannot create more")
	}
	{
		_, exist, err := s.storage.Get(cameraID)
		if err != nil {
			xl.Warnf("Camera %s proceess failed to check storage", cameraID)
			return errors.New("Failed to check storage")
		}
		if exist {
			if !forceStart {
				xl.Warnf("Camera %s proceess already running", cameraID)
				return errors.New("Camera is already running")
			} else {
				err = s.stopFfmpegProcess(ctx, env, cameraID)
				if err != nil {
					xl.Warnf("Camera %s proceess already running", cameraID)
					return errors.New("Camera is already running")
				}
			}
		}
	}
	stdout, err := os.Create(fmt.Sprintf("out-%v.log", fInfo.CameraID))
	if err != nil {
		xl.Warnf("Camera %v process failed create log file", fInfo.CameraID)
		return errors.New("Camera process failed to start")
	}

	stderr, err := os.Create(fmt.Sprintf("err-%v.log", fInfo.CameraID))
	if err != nil {
		xl.Warnf("Camera %v process failed create log file", fInfo.CameraID)
		return errors.New("Camera process failed to start")
	}
	args := s.buildFfmpegArgs(ctx, env, fInfo)
	{
		if fInfo.ClusterID != "" {
			err := s.createCluster(ctx, env, fInfo.ClusterID, fInfo.ClusterFeatureTimeout)
			if err != nil {
				xl.Warnf("Camera %s start cluster failed", cameraID)
				return errors.New("Camera start cluster failed")
			}
		}
	}

	cmd := exec.Command(s.conf.FfmpegCmd, args...)
	cmd.Stdout = stdout
	cmd.Stderr = stderr
	cmd.Start()
	go func() {
		cmd.Wait()
	}()
	time.Sleep(time.Second * 2)
	if cmd.Process == nil {
		xl.Warnf("Camera %v process failed to start", cameraID)
		return errors.New("Camera process failed to start")
	}
	if cmd.ProcessState != nil && cmd.ProcessState.Exited() {
		xl.Warnf("Camera %v process exited unexpectedly: %v", cameraID)
		return errors.New("Camera process exited unexpectedly")
	}
	fInfo.Pid = cmd.Process.Pid
	err = s.storage.Add(fInfo)
	if err != nil {
		xl.Warnf("Camera %v failed add to storage", cameraID)
		time.AfterFunc(time.Nanosecond, func() { cmd.Process.Kill() })
		cmd.Wait()
		return err
	}
	xl.Infof("Starting camera %s process %d", cameraID, fInfo.Pid)
	return nil
}

func (s *Service) stopFfmpegProcess(ctx context.Context, env *restrpc.Env, cameraID string) error {
	xl, ctx := s.initContext(ctx, env)
	fInfo, exist, err := s.storage.Get(cameraID)
	if err != nil {
		xl.Warnf("Camera %s proceess failed to check storage", cameraID)
		return errors.New("Failed to check storage")
	}
	if !exist {
		xl.Warnf("Camera %s proceess not exist", cameraID)
		return errors.New("Camera not running")
	}
	{
		if fInfo.ClusterID != "" {
			err := s.destroyCluster(ctx, env, fInfo.ClusterID)
			if err != nil {
				xl.Warnf("Camera %s stop cluster failed: %v", cameraID, err)
				return errors.New("Camera stop cluster failed")
			}
		}
	}
	{
		p, err := os.FindProcess(fInfo.Pid)
		if err != nil {
			xl.Warnf("Camera %v process not found", cameraID)
			return errors.New("Camera failed to locate process")
		}
		xl.Infof("Killing camera %s process %v", cameraID, fInfo.Pid)
		err = s.storage.Remove(cameraID)
		if err != nil {
			xl.Warnf("Camera %v process failed to remove from storage", cameraID)
			return errors.New("Camera failed to remove from storgae")
		}
		// NOTE: if kill directly without wait, the process turns info defunct
		time.AfterFunc(time.Nanosecond, func() { p.Kill() })
		p.Wait()
	}
	return nil
}

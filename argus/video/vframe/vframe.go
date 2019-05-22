package vframe

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/fsnotify/fsnotify"
	xlog "github.com/qiniu/xlog.v1"
)

const (
	DEFAULT_INTERVAL                = 5.0
	VIDEO_CLASSIFY_DEFAULT_INTERVAL = 0.1
)

// Job ...
type Job interface {
	Cuts() <-chan CutResponse
	Error() error
	Stop()
}

type JobClean interface {
	Job
	Clean()
}

// Vframe ...
type Vframe interface {
	Run(context.Context, VframeRequest) (Job, error)
}

//----------------------------------------------------------------------------//

type _Vframe interface {
	run(context.Context, VframeRequest) error
}

type _Watcher interface {
	watch(context.Context, chan<- CutResponse) error
}

type _Job struct {
	ch chan CutResponse

	_Vframe
	_Watcher

	ctx    context.Context
	cancel context.CancelFunc
	req    VframeRequest
	dir    string

	err error
}

func newVframeJob(
	ctx context.Context, req VframeRequest, dir string, _v _Vframe, _w _Watcher,
) (JobClean, error) {
	ctx, cancel := context.WithCancel(ctx)
	job := &_Job{
		ch:      make(chan CutResponse),
		_Vframe: _v, _Watcher: _w,
		ctx: ctx, cancel: cancel,
		req: req,
		dir: dir,
	}

	go job.run()
	return job, nil
}

func (j *_Job) Cuts() <-chan CutResponse { return j.ch }
func (j *_Job) Stop()                    { j.cancel() }
func (j *_Job) Error() error             { return j.err }
func (j *_Job) Clean() {
	os.RemoveAll(j.dir) // delete the directory
}

func (j *_Job) run() {

	ctx, cancel := context.WithCancel(j.ctx)
	defer cancel()

	var (
		xl   = xlog.FromContextSafe(ctx)
		wait = new(sync.WaitGroup)

		err1 = new(error)
	)

	wait.Add(1)
	go func(ctx context.Context, err1 *error) {
		xl := xlog.FromContextSafe(ctx)
		defer func() {
			if err := recover(); err != nil {
				xl.Error(err)
			}
			wait.Done()
			xl.Info("watch done.")

			if *err1 != nil {
				cancel()
			}
		}()
		if err := j._Watcher.watch(ctx, j.ch); err != nil {
			xl.Errorf("watch failed. %v", err)
			*err1 = err
		}
	}(xlog.NewContext(ctx, xlog.FromContextSafe(ctx).Spawn()), err1)

	defer func() {
		if err := recover(); err != nil {
			xl.Error(err)
		}
	}()
	if err := j._Vframe.run(ctx, j.req); err != nil {
		xl.Errorf("run vframe failed. %v", err)
		cancel()
		j.err = err
	}

	xl.Info("vframe done.")
	wait.Wait()
	if j.err == nil {
		j.err = *err1
	}
	xl.Info("close channel")
	close(j.ch)
}

////////////////////////////////////////////////////////////////////////////////

type VframeConfig struct {
	Dir string `json:"dir"`
}

type vframe struct {
	VframeConfig
	URIProxy

	newCmd          func(context.Context, func() ([]string, error)) Cmd
	genCmdArgs      func(context.Context, VframeRequest, string, URIProxy) ([]string, error)
	newWatchControl func(context.Context, VframeRequest) _WatchControl
}

func NewVframe(conf VframeConfig, proxy URIProxy) Vframe {
	return vframe{VframeConfig: conf, URIProxy: proxy}
}
func NewLive(conf VframeConfig, proxy URIProxy) Vframe {
	// genCmdArgs := _GenLiveCmdWithFFmpeg
	genCmdArgs := _GenLiveCmdWithGenpic
	return vframe{VframeConfig: conf, URIProxy: proxy,
		newCmd:     NewLiveCmd(),
		genCmdArgs: genCmdArgs,
		newWatchControl: func(ctx context.Context, req VframeRequest) _WatchControl {
			timeout := int64(time.Hour / time.Millisecond)
			if req.Live != nil && req.Live.Timeout > 0 {
				timeout = int64(req.Live.Timeout * 1000)
			}
			return newWatchControl(ctx, timeout)
		},
	}
}

func (f vframe) Run(ctx context.Context, req VframeRequest) (Job, error) {

	dir := path.Join(f.Dir, xlog.GenReqId())
	if err := os.MkdirAll(dir, 0755); err != nil {
		return nil, err
	}
	const (
		_FinishFile = "finish.end"
	)
	_vf := &_vframe{URIProxy: f.URIProxy, dir: dir, finishFile: _FinishFile}
	if f.newWatchControl == nil {
		_vf._WatchControl = newVoidWatchControl()
	} else {
		_vf._WatchControl = f.newWatchControl(ctx, req)
	}
	_vf.newCmd = f.newCmd
	if f.genCmdArgs != nil {
		_vf._genVframeCmd = f.genCmdArgs
	} else {
		if os.Getenv("DEBUG") == "TRUE" {
			_vf._genVframeCmd = _GenVframeCmdWithFFmpeg
		} else {
			_vf._genVframeCmd = _vf.genVframeCmd
		}
	}

	return newVframeJob(ctx, req, dir, _vf, _vf)
}

////////////////////////////////////////////////////////////////////////////////

type _vframe struct {
	URIProxy
	dir        string
	finishFile string

	newCmd        func(context.Context, func() ([]string, error)) Cmd                      // TODO init
	_genVframeCmd func(context.Context, VframeRequest, string, URIProxy) ([]string, error) // 方便UT

	_WatchControl
}

func (vf _vframe) genVframeCmd(
	ctx context.Context, req VframeRequest, outDir string, proxy URIProxy,
) ([]string, error) {

	var (
		xl = xlog.FromContextSafe(ctx)
	)

	cmd := []string{
		"gen_pic",
	}

	if req.Params.StartTime > 0 {
		cmd = append(cmd, "-ss", fmt.Sprintf("%.3f", req.Params.StartTime))
	}

	if req.Params.Duration > 0 {
		cmd = append(cmd, "-t", fmt.Sprintf("%.3f", req.Params.Duration))
	}
	skipArgs := []string{}
	if req.Params.GetMode() == MODE_KEY {
		skipArgs = append(skipArgs, "-skip_frame", "nokey")
	}
	cmd = append(cmd, skipArgs...)
	if proxy == nil {
		cmd = append(cmd, "-i", req.Data.URI)
	} else {
		cmd = append(cmd, "-i", proxy.URI(req.Data.URI))
	}

	if req.Params.GetMode() == MODE_INTERVAL && req.Params.Interval != 0 {
		cmd = append(cmd, "-r", fmt.Sprintf("%f", float64(1)/req.Params.Interval))
	}

	cmd = append(cmd, "-prefix", outDir+"/")
	xl.Info("vframe cmd:", cmd)

	return cmd, nil
}

func (vf *_vframe) run(ctx context.Context, req VframeRequest) error {

	xl := xlog.FromContextSafe(ctx)

	// if _, e := os.Stat(outDir); e != nil && os.IsExist(e) {
	// 	os.Mkdir(outDir, 0755)
	// }

	if err := CheckMode(req.Params.GetMode()); err != nil {
		return err
	}

	if req.Params.GetMode() == MODE_INTERVAL {
		if req.Params.Interval == 0 {
			req.Params.Interval = DEFAULT_INTERVAL
		} else {
			if err := CheckInterval(req.Params.Interval); err != nil {
				return err
			}
		}
	}

	if err := CheckStartTime(req.Params.StartTime); err != nil {
		return err
	}

	if err := CheckDuration(req.Params.Duration); err != nil {
		return err
	}

	var cmd Cmd
	if vf.newCmd == nil {
		vf.newCmd = NewVodCmd()
	}
	cmd = vf.newCmd(ctx, func() ([]string, error) {
		args, _ := vf._genVframeCmd(ctx, req, vf.dir, vf.URIProxy)
		return args, nil
	})

	if err := cmd.Start(); err != nil {
		return err
	}

	{
		var kill int32 = 0
		done := make(chan bool)
		go func() {
			// xl := xlog.NewWith("CMD")
			select {
			case <-done:
			case <-ctx.Done():
				atomic.StoreInt32(&kill, 1)
				_ = cmd.Kill()
			}
			// _end: // NOTHING
		}()
		defer func() {
			if atomic.LoadInt32(&kill) == 0 {
				done <- true
			}
		}()
	}
	if err := cmd.Wait(); err != nil {
		xl.Warnf("cmd finish. %v", err)
		err = GenVframeCmdError(err)
		return err
	}
	xl.Info("cmd finish.")

	_ = os.Mkdir(path.Join(vf.dir, vf.finishFile), 0755)
	// finish
	return nil
}

func (vf _vframe) parseFilename(str string) int64 {
	lines := strings.Split(str, "/")
	if len(lines) < 1 {
		return -1
	}

	nums := strings.Split(lines[len(lines)-1], ".")
	if len(nums) < 2 {
		return -1
	}
	i, _ := strconv.ParseInt(nums[0], 10, 64)
	return i
}

func (vf *_vframe) watch(ctx context.Context, ch chan<- CutResponse) error {

	xl := xlog.FromContextSafe(ctx)
	watcher, err := fsnotify.NewWatcher()
	if err != nil {
		xl.Errorf("New fsnotify watcher failed, err: %v", err)
		return err
	}
	defer watcher.Close()

	err = watcher.Add(vf.dir)
	if err != nil {
		xl.Errorf("add image dir to watcher failed, err: %v", err)
		return err
	}

	var done = make(chan bool)
	go func(ctx context.Context) {
		xl := xlog.FromContextSafe(ctx)
		defer func() {
			if err := recover(); err != nil {
				xl.Error(err)
			}
		}()

		var last *CutResponse
		for {
			select {
			case event, ok := <-watcher.Events:
				if !ok {
					goto _end
				}
				if event.Op&fsnotify.Create == fsnotify.Create {
					if event.Name == path.Join(vf.dir, vf.finishFile) {
						if last != nil {
							ch <- *last
						}
						done <- true
						continue
						// goto _end
					}
					xl.Debug("New vframe file [", event.Name, "] created")
					var resp CutResponse
					resp.Result.Cut =
						struct {
							Offset int64  `json:"offset"`
							URI    string `json:"uri"`
						}{
							Offset: vf.parseFilename(event.Name),
							URI:    "file://" + event.Name,
						}
					resp = vf._WatchControl.Update(resp)
					if last != nil {
						ch <- *last
					}
					last = &resp
				}
			case err, ok := <-watcher.Errors:
				if !ok {
					goto _end
				}
				// TODO error handle
				xl.Warnf("watch error: %v", err)
				// return err
			}
		}
	_end: // NOTHING
	}(spawnContext(ctx))

	select {
	case err := <-vf._WatchControl.Wait():
		return err
	case <-ctx.Done():
		return ctx.Err()
	case <-done:
	}
	close(done)

	return nil
}

////////////////////////////////////////////////////////////////////////////////

type MockJob struct {
	CH chan CutResponse
}

func (mock MockJob) Cuts() <-chan CutResponse {
	return mock.CH
}
func (mock MockJob) Error() error { return nil }
func (mock MockJob) Stop()        {}

type MockVframe struct {
	NewJob func(VframeRequest) Job
}

func (mock MockVframe) Run(ctx context.Context, req VframeRequest) (Job, error) {
	return mock.NewJob(req), nil
}

//----------------------------------------------------------------------------//

type Cmd interface {
	Start() error
	Wait() error
	Kill() error
}

var _ Cmd = new(VodCmd)

type VodCmd struct {
	genVframeCmd func() ([]string, error) // 方便UT
	cmd          *exec.Cmd
}

func NewVodCmd() func(context.Context, func() ([]string, error)) Cmd {
	return func(ctx context.Context, gen func() ([]string, error)) Cmd {
		return &VodCmd{genVframeCmd: gen}
	}
}

func (c *VodCmd) Start() error {
	args, _ := c.genVframeCmd()
	c.cmd = exec.Command(args[0], args[1:]...)
	return c.cmd.Start()
}

func (c *VodCmd) Wait() error { return c.cmd.Wait() }
func (c *VodCmd) Kill() error { return c.cmd.Process.Kill() }

type LiveCmd struct {
	genVframeCmd func() ([]string, error) // 方便UT
	done         chan error
	ctx          context.Context
	cancel       context.CancelFunc
}

func NewLiveCmd() func(context.Context, func() ([]string, error)) Cmd {
	return func(ctx context.Context, gen func() ([]string, error)) Cmd {
		ctx, cancel := context.WithCancel(ctx)
		return &LiveCmd{
			genVframeCmd: gen, done: make(chan error),
			ctx: ctx, cancel: cancel,
		}
	}
}

func (c *LiveCmd) Start() error {
	go func() {
		var xl = xlog.NewWith("LIVE")
		var err error
		defer func() { c.done <- err }()
		for {
			args, _ := c.genVframeCmd()
			cmd := exec.Command(args[0], args[1:]...)
			err1 := cmd.Start()
			if err1 != nil {
				xl.Warnf("start live cmd failed. %v", err1)
				continue
			}

			done := make(chan error)
			go func() { done <- cmd.Wait() }()
			select {
			case <-c.ctx.Done():
				_ = cmd.Process.Kill()
				return
			case err1 := <-done:
				xl.Warnf("live cmd failed. %v", err1)
			}
		}
	}()
	return nil
}

func (c *LiveCmd) Wait() error { return <-c.done }
func (c *LiveCmd) Kill() error {
	c.cancel()
	return nil
}

//----------------------------------------------------------------------------//

type _WatchControl interface {
	Wait() <-chan error
	Update(CutResponse) CutResponse
	Stop()
}

var _ _WatchControl = &watchControl{}

type watchControl struct {
	timeout int64
	last    int64

	ch chan error

	ctx    context.Context
	cancel context.CancelFunc
}

func newWatchControl(ctx context.Context, timeout int64) *watchControl {
	ch := make(chan error)
	ctx, cancel := context.WithCancel(ctx)
	c := &watchControl{timeout: timeout, ch: ch, ctx: ctx, cancel: cancel}
	atomic.StoreInt64(&c.last, time.Now().UnixNano()/int64(time.Millisecond))
	go func() {
		ticker := time.NewTicker(time.Second * 10)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case now := <-ticker.C:
				xlog.NewWith("C").Infof("%d %d %d", now.UnixNano()/int64(time.Millisecond), atomic.LoadInt64(&c.last), timeout)
				if now.UnixNano()/int64(time.Millisecond)-atomic.LoadInt64(&c.last) > timeout {
					ch <- ErrTimeout
					return
				}
			}
		}
	}()
	return c
}

func (c watchControl) Wait() <-chan error { return c.ch }
func (c *watchControl) Update(cut CutResponse) CutResponse {
	cut.Result.Cut.Offset = time.Now().UnixNano() / int64(time.Millisecond)
	atomic.StoreInt64(&c.last, cut.Result.Cut.Offset)
	return cut
}
func (c watchControl) Stop() { c.cancel() }

var _ _WatchControl = voidWatchControl{}

type voidWatchControl struct {
	ch chan error
}

func newVoidWatchControl() voidWatchControl                   { return voidWatchControl{ch: make(chan error)} }
func (c voidWatchControl) Wait() <-chan error                 { return c.ch }
func (c voidWatchControl) Update(cut CutResponse) CutResponse { return cut }
func (c voidWatchControl) Stop()                              {}

//----------------------------------------------------------------------------//

func _GenVframeCmdWithFFmpeg(
	ctx context.Context, req VframeRequest, outDir string, proxy URIProxy,
) ([]string, error) {

	var (
		xl = xlog.FromContextSafe(ctx)
	)
	xl.Infof("Vframe Req: %#v", req)

	cmd := []string{
		"ffmpeg",
	}
	skipArgs := []string{}
	if *req.Params.Mode == 1 {
		skipArgs = append(skipArgs, "-skip_frame", "nokey", "-vsync", "2")
	}
	cmd = append(cmd, skipArgs...)
	if proxy == nil {
		cmd = append(cmd, "-i", req.Data.URI)
	} else {
		cmd = append(cmd, "-i", proxy.URI(req.Data.URI))
	}

	if req.Params.GetMode() == MODE_INTERVAL && req.Params.Interval != 0 {
		cmd = append(cmd, "-vf", "fps=fps="+fmt.Sprintf("%f", float64(1)/req.Params.Interval))
	}

	cmd = append(cmd, outDir+"/%d.jpg")
	xl.Info("vframe cmd:", cmd)

	return cmd, nil
}

var _ = _GenLiveCmdWithFFmpeg // JUST FOR LINT

func _GenLiveCmdWithFFmpeg(
	ctx context.Context, req VframeRequest, outDir string, proxy URIProxy,
) ([]string, error) {

	var (
		xl = xlog.FromContextSafe(ctx)
	)
	xl.Infof("Vframe Req: %#v", req)

	cmd := []string{
		"ffmpeg",
	}
	// skipArgs := []string{}
	// if req.Params.Mode == 1 {
	// 	skipArgs = append(skipArgs, "-skip_frame", "nokey")
	// }
	// cmd = append(cmd, skipArgs...)
	cmd = append(cmd, "-i", req.Data.URI)

	if req.Params.GetMode() == MODE_INTERVAL && req.Params.Interval != 0 {
		cmd = append(cmd, "-vf", "fps=fps="+fmt.Sprintf("%f", float64(1)/req.Params.Interval))
	}
	cmd = append(cmd, "-timeout", "5")

	cmd = append(cmd, outDir+"/%d.jpg")
	xl.Info("vframe cmd:", cmd)

	return cmd, nil
}

func _GenLiveCmdWithGenpic(
	ctx context.Context, req VframeRequest, outDir string, proxy URIProxy,
) ([]string, error) {

	var (
		xl = xlog.FromContextSafe(ctx)
	)

	cmd := []string{
		"gen_pic",
	}

	if req.Params.StartTime > 0 {
		cmd = append(cmd, "-ss", fmt.Sprintf("%.3f", req.Params.StartTime))
	}

	if req.Params.Duration > 0 {
		cmd = append(cmd, "-t", fmt.Sprintf("%.3f", req.Params.Duration))
	}
	skipArgs := []string{}
	if req.Params.GetMode() == MODE_KEY {
		skipArgs = append(skipArgs, "-skip_frame", "nokey")
	}
	cmd = append(cmd, skipArgs...)
	// if vf.URIProxy == nil {
	cmd = append(cmd, "-i", req.Data.URI)
	// } else {
	// cmd = append(cmd, "-i", vf.URIProxy.URI(req.Data.URI))
	// }

	if req.Params.GetMode() == MODE_INTERVAL && req.Params.Interval != 0 {
		cmd = append(cmd, "-r", fmt.Sprintf("%f", float64(1)/req.Params.Interval))
	}

	cmd = append(cmd, "-prefix", outDir+"/")
	xl.Info("vframe cmd:", cmd)

	return cmd, nil
}

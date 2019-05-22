package segment

import (
	"bufio"
	"context"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"strconv"
	"strings"

	xlog "github.com/qiniu/xlog.v1"
)

const (
	DEFAULT_DURATION = 10.0
)

// Job ...
type Job interface {
	Clips() <-chan ClipResponse
	Error() error
	Stop()
}

type Segment interface {
	Run(context.Context, SegmentRequest) (Job, error)
}

//----------------------------------------------------------------------------//

type _Segment interface {
	run(context.Context, SegmentRequest) error
}

type _Walker interface {
	walk(context.Context, chan<- ClipResponse) error
}

type _Job struct {
	ch chan ClipResponse

	_Segment
	_Walker

	ctx    context.Context
	cancel context.CancelFunc
	req    SegmentRequest

	err error
}

func newSegmentJob(
	ctx context.Context, req SegmentRequest, _s _Segment, _w _Walker,
) (Job, error) {
	ctx, cancel := context.WithCancel(ctx)
	job := &_Job{
		ch:       make(chan ClipResponse),
		_Segment: _s, _Walker: _w,
		ctx: ctx, cancel: cancel,
		req: req,
	}

	go job.run()
	return job, nil
}

func (j *_Job) Clips() <-chan ClipResponse { return j.ch }
func (j *_Job) Stop()                      { j.cancel() }
func (j *_Job) Error() error               { return j.err }

func (j *_Job) run() {

	var (
		ctx = j.ctx
		xl  = xlog.FromContextSafe(ctx)
	)

	defer func() {
		if err := recover(); err != nil {
			xl.Error(err)
		}
	}()
	if err := j._Segment.run(ctx, j.req); err != nil {
		xl.Errorf("run segment failed. %v", err)
		j.err = err
	}

	xl.Info("segment done.")

	if err := j._Walker.walk(ctx, j.ch); err != nil {
		xl.Errorf("walk failed. %v", err)
		j.err = err
	}

	xl.Info("close channel")
	close(j.ch)
}

////////////////////////////////////////////////////////////////////////////////

type SegmentConfig struct {
	Dir string `json:"dir"`
}

type segment struct {
	SegmentConfig
	URIProxy
}

func NewSegment(conf SegmentConfig, proxy URIProxy) Segment {
	return segment{SegmentConfig: conf, URIProxy: proxy}
}

func (s segment) Run(ctx context.Context, req SegmentRequest) (Job, error) {

	dir := path.Join(s.Dir, xlog.GenReqId())
	if err := os.MkdirAll(dir, 0755); err != nil {
		return nil, err
	}
	_s := &_segment{URIProxy: s.URIProxy, dir: dir}
	_s._genSegmentCmd = _s.genSegmentCmd

	return newSegmentJob(ctx, req, _s, _s)
}

type _segment struct {
	URIProxy
	dir string

	_genSegmentCmd func(context.Context, SegmentRequest, string) ([]string, error) // 方便UT
}

func (s _segment) genSegmentCmd(
	ctx context.Context, req SegmentRequest, outDir string,
) ([]string, error) {

	var (
		xl = xlog.FromContextSafe(ctx)
	)

	cmd := []string{
		"ffmpeg",
	}

	// cmd = append(cmd, "-ignore_editlist", "1")
	cmd = append(cmd, "-i", s.URIProxy.URI(req.Data.URI))
	cmd = append(cmd, "-sn", "-f", "segment")
	cmd = append(cmd, "-vcodec", "copy")

	cmd = append(cmd, "-segment_time", strconv.Itoa(10))              // 按时长分片
	cmd = append(cmd, "-segment_list", outDir+"/"+"segment_list.csv") // 分片结果信息，包括文件名，起始时间点
	cmd = append(cmd, "-y", outDir+"/"+"segment_%6d.ts")              // 输出文件

	xl.Info("segment cmd:", cmd)

	return cmd, nil
}

func (s *_segment) run(ctx context.Context, req SegmentRequest) error {

	xl := xlog.FromContextSafe(ctx)

	// if _, e := os.Stat(outDir); e != nil && os.IsExist(e) {
	// 	os.Mkdir(outDir, 0755)
	// }

	if err := CheckMode(req.Params.Mode); err != nil {
		return err
	}

	// if err := CheckDuration(req.Params.Duration); err != nil {
	// 	return err
	// }

	args, _ := s._genSegmentCmd(ctx, req, s.dir)

	cmd := exec.Command(args[0], args[1:]...)
	if err := cmd.Start(); err != nil {
		return err
	}
	{
		done := make(chan bool)
		go func() {
			for {
				select {
				case <-done:
					goto _end
				case <-ctx.Done():
					cmd.Process.Kill()
				}
			}
		_end: // NOTHING
		}()
		defer func() {
			done <- true
		}()
	}
	if err := cmd.Wait(); err != nil {
		xl.Warnf("cmd finish. %v", err)
		err = GenSegmentCmdError(err)
		return err
	}
	xl.Info("cmd finish.")

	return nil
}

func (s *_segment) walk(ctx context.Context, ch chan<- ClipResponse) error {

	xl := xlog.FromContextSafe(ctx)

	filename := filepath.Join(s.dir, "segment_list.csv")
	f, err := os.Open(filename) // 读取分片结果信息，包括文件名，起始时间点
	if err != nil {
		xl.Infof("open file %q failed:%s", filename, err.Error)
		return err
	}
	defer func() {
		f.Close()
		os.Remove(filename)
	}()

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		text := strings.Split(scanner.Text(), ",")
		var resp ClipResponse
		resp.Result.Clip = struct {
			OffsetBegin int64  `json:"offset_begin"`
			OffsetEnd   int64  `json:"offset_end"`
			URI         string `json:"uri"`
		}{
			OffsetBegin: func() int64 {
				f64, _ := strconv.ParseFloat(text[1], 64)
				return int64(f64 * 1000)
			}(),
			OffsetEnd: func() int64 {
				f64, _ := strconv.ParseFloat(text[2], 64)
				return int64(f64 * 1000)
			}(),
			URI: "file://" + s.dir + "/" + text[0],
		}
		ch <- resp
	}

	return nil
}

////////////////////////////////////////////////////////////////////////////////

type MockJob struct {
	CH chan ClipResponse
}

func (mock MockJob) Clips() <-chan ClipResponse {
	return mock.CH
}
func (mock MockJob) Error() error { return nil }
func (mock MockJob) Stop()        {}

type MockSegment struct {
	NewJob func(SegmentRequest) Job
}

func (mock MockSegment) Run(ctx context.Context, req SegmentRequest) (Job, error) {
	return mock.NewJob(req), nil
}

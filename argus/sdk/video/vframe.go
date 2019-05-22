package video

import (
	"container/list"
	"context"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/fsnotify/fsnotify"

	httputil "github.com/qiniu/http/httputil.v1"
	xlog "github.com/qiniu/xlog.v1"
	"qiniu.com/argus/com/util"
)

type Vframe interface {
	Next(context.Context, int) ([]CutRequest, bool)
	Error() error
	Close()
}

type VframeWithResult interface {
	Vframe
	Get(context.Context, int64) CutRequest
	Set(context.Context, CutResponse) error
}

////////////////////////////////////////////////////////////////////////////////

// 运行cmd，截帧输出至本地文件系统
var _ Vframe = &vframeV1{}

type vframeV1 struct {
	workspace     string
	parseFilename func(string) (int64, error)
	cuts          chan string
	cancel        context.CancelFunc
	err           error
	*sync.Mutex
}

var (
	GenpicErrInvalidParameters = httputil.NewError(http.StatusBadRequest, "invalid parameters")
	GenpicErrCannotFindVideo   = httputil.NewError(http.StatusFailedDependency, "cannot find the video")
	GenpicErrCannotOpenFile    = httputil.NewError(http.StatusBadRequest, "cannot open the file")
	GenpicErrCannotAllowMemory = httputil.NewError(http.StatusInternalServerError, "cannot allow memory")
)

func formatGenpicError(err error) error {
	switch {
	case strings.Compare(err.Error(), "exit status 1") == 0:
		return GenpicErrInvalidParameters
	case strings.Compare(err.Error(), "exit status 2") == 0,
		strings.Compare(err.Error(), "exit status 3") == 0:
		return GenpicErrCannotFindVideo
	case strings.Compare(err.Error(), "exit status 4") == 0:
	case strings.Compare(err.Error(), "exit status 5") == 0,
		strings.Compare(err.Error(), "exit status 6") == 0,
		strings.Compare(err.Error(), "exit status 7") == 0,
		strings.Compare(err.Error(), "exit status 9") == 0,
		strings.Compare(err.Error(), "exit status 10") == 0,
		strings.Compare(err.Error(), "exit status 11") == 0:
		return GenpicErrCannotOpenFile
	case strings.Compare(err.Error(), "exit status 8") == 0:
		return GenpicErrCannotAllowMemory
	}
	return err
}

func NewGenpic(
	ctx context.Context,
	workspace string, // 本地文件系统目录，用于存放中间结果文件，需要独立目录
	url string, // 视频文件地址
	mode int, // mode = 0，按固定时间截帧；mode = 1，按关键帧截帧
	intervalMS int64, // mode = 0 的情况下，截帧的时间间隔
	startTimeMS int64, // 视频起始位置
	durationMS int64, // 处理的视频时长
	ignoreExit bool, // 是否忽略cmd退出，用于直播场景
) Vframe {

	xl := xlog.FromContextSafe(ctx)

	args := []string{"gen_pic"}
	if startTimeMS > 0 {
		args = append(args, "-ss", fmt.Sprintf("%.3f", float64(startTimeMS)/1000.0))
	}
	if durationMS > 0 {
		args = append(args, "-t", fmt.Sprintf("%.3f", float64(durationMS)/1000.0))
	}
	if mode == 1 {
		args = append(args, "-skip_frame", "nokey")
	}
	args = append(args, "-i", url)
	if mode == 0 && intervalMS != 0 {
		args = append(args, "-r", fmt.Sprintf("%f", float64(1000)/float64(intervalMS)))
	}
	if !strings.HasSuffix(workspace, "/") {
		workspace = workspace + "/"
	}
	args = append(args, "-prefix", workspace)
	xl.Info("vframe cmd:", args)

	return newVframeV1(ctx, args, formatGenpicError,
		ignoreExit, workspace,
		func(name string) (int64, error) {
			strs := strings.Split(name, "/")
			if len(strs) < 1 {
				return -1, nil
			}
			nums := strings.Split(strs[len(strs)-1], ".")
			if len(nums) < 2 {
				return -1, nil
			}
			return strconv.ParseInt(nums[0], 10, 64)
		})
}

func newVframeV1(
	ctx context.Context,
	cmdArgs []string, formatCmdError func(error) error,
	ignoreExit bool,
	workspace string, parseFilename func(string) (int64, error),
) *vframeV1 {
	ctx, cancel := context.WithCancel(ctx)
	v := &vframeV1{
		workspace:     workspace,
		parseFilename: parseFilename,
		cancel:        cancel,
		cuts:          make(chan string, 512),
		Mutex:         new(sync.Mutex),
	}

	if err := os.MkdirAll(workspace, 0755); err != nil {
		close(v.cuts)
		v.err = err
		return v
	}

	do := func(f func(context.Context) error, id int) {
		go func(ctx context.Context) {
			var xl = xlog.FromContextSafe(ctx)
			var err error
			defer func() {
				err1 := recover()
				if err1 != nil {
					err = fmt.Errorf("panic: %v", err1)
					xl.Stack(err)
				}
				if err != nil {
					v.Lock()
					defer v.Unlock()
					if v.err == nil {
						v.err = err
					}
					v.cancel()
				}
			}()
			err = f(ctx)
		}(util.SpawnContext2(ctx, id))
	}

	endFile := "finish.end"
	begin := make(chan bool, 1)
	do(func(ctx context.Context) (err error) {
		select {
		case <-begin:
		case <-ctx.Done():
			return
		}

		err = v.runCmd(ctx, cmdArgs, ignoreExit)
		if err == nil {
			_ = os.Mkdir(path.Join(workspace, endFile), 0755)
		}
		if err != nil {
			if formatCmdError != nil {
				err = formatCmdError(err)
			}
			xlog.FromContextSafe(ctx).Warnf("cmd err: %v", err)
		}
		return
	}, 0)
	do(func(ctx context.Context) (err error) {
		defer func() {
			close(v.cuts)
		}()
		var xl = xlog.FromContextSafe(ctx)
		watcher, err := fsnotify.NewWatcher()
		if err != nil {
			xl.Errorf("New fsnotify watcher failed, err: %v", err)
			return err
		}
		defer watcher.Close()

		err = watcher.Add(workspace)
		if err != nil {
			xl.Errorf("add image dir to watcher failed, err: %v", err)
			return err
		}

		begin <- true // 先开启watcher，再执行cmd

		isEndFile := func(name string) bool { return path.Join(workspace, endFile) == name }
		err = v.watch(ctx, watcher, isEndFile, func(name string) {
			v.cuts <- name
		})
		return
	}, 1)

	return v
}

func (v *vframeV1) Next(ctx context.Context, n int) ([]CutRequest, bool) {
	f := func(name string) CutRequest {
		offsetMS, _ := v.parseFilename(name)
		bs, _ := ioutil.ReadFile(name)
		return CutRequest{OffsetMS: offsetMS, Body: bs}
	}
	reqs := make([]CutRequest, 0, n)
	select {
	case name, ok := <-v.cuts:
		if !ok {
			return nil, false
		}
		reqs = append(reqs, f(name))
	case <-ctx.Done():
		return nil, false
	}
	for len(reqs) < n {
		select {
		case name, ok := <-v.cuts:
			if !ok {
				goto ForEnd
			}
			reqs = append(reqs, f(name))
		default:
			goto ForEnd
		}
	}
ForEnd:
	return reqs, true
}
func (v *vframeV1) Error() error { return v.err }
func (v *vframeV1) Close() {
	v.cancel()
	_ = os.RemoveAll(v.workspace)
}

func (v vframeV1) runCmd(ctx context.Context, args []string, ignoreExit bool) error {
	xl := xlog.FromContextSafe(ctx)
	for {
		cmd := exec.Command(args[0], args[1:]...)
		err := cmd.Start()
		if err != nil {
			xl.Warnf("start cmd failed. %v %v", args, err)
			return err
		}
		done := make(chan error, 1)
		go func() { done <- cmd.Wait() }()
		select {
		case <-ctx.Done():
			_ = cmd.Process.Kill()
			return nil
		case err := <-done:
			if ignoreExit {
				xl.Warnf("cmd exit. %v %v", args, err)
			} else {
				return err
			}
		}
	}
}

// 本方案在内存不够的情况下，可能导致截帧丢失
// Tip：监听的是Create事件（非Close事件），所以只能依靠后一帧产生判定前一帧生成完
func (v vframeV1) watch(
	ctx context.Context, watcher *fsnotify.Watcher,
	isEndFile func(string) bool, appendFile func(string),
) error {

	xl := xlog.FromContextSafe(ctx)

	var last *string
	for {
		select {
		case <-ctx.Done():
			watcher.Close()
		case event, ok := <-watcher.Events:
			if !ok {
				goto _END
			}
			if event.Op&fsnotify.Create == fsnotify.Create {
				xl.Debug("New vframe file [", event.Name, "] created")
				if last != nil {
					appendFile(*last)
				}
				if isEndFile(event.Name) {
					watcher.Close()
					last = nil
				} else {
					last = &event.Name
				}
			}
		case err, ok := <-watcher.Errors:
			if !ok {
				goto _END
			}
			xl.Warnf("watch error: %v", err)
			// return err
		}
	}
_END:

	return nil
}

//----------------------------------------------------------------------------//

// 运行cmd，截帧回调至http服务器，同时接受帧处理结果
var _ VframeWithResult = &vframeV2{}

type vframeV2Item struct {
	CutRequest
	Waiting bool
	ch      chan CutResponse
}

type vframeV2 struct {
	cuts   *list.List // vframeV2Item
	msgs   chan *list.Element
	cancel context.CancelFunc
	done   int32
	err    error
	*sync.Mutex
}

func NewFfmpegai(
	ctx context.Context,
	url string, // 视频文件地址
	interval int64, // 截帧的帧数间隔
	timeoutMS int64,
	downsteam string,
) VframeWithResult {

	var genCmdArgs = func(hook string) []string {
		var xl = xlog.FromContextSafe(ctx)
		var cmd = []string{"ffmpeg_ai"}
		cmd = append(cmd, "-copyts")
		if strings.HasPrefix(url, "rtsp://") {
			cmd = append(cmd, "-rtsp_flags", "prefer_tcp")
		}
		cmd = append(cmd, "-re")
		cmd = append(cmd, "-i", url)
		cmd = append(cmd, "-an")
		cmd = append(cmd, "-c", "copy")
		if timeoutMS <= 0 {
			timeoutMS = 30 * 1000
		}
		if strings.HasPrefix(url, "rtsp://") {
			cmd = append(cmd, "-stimeout", strconv.Itoa(int(timeoutMS)*1000))
		} else {
			cmd = append(cmd, "-rw_timeout", strconv.Itoa(int(timeoutMS)*1000))
		}

		// 固定帧数间隔
		h264_metadata := "h264_metadata=sei_user_data='086f3693-b7b3-4f2c-9653-21492feee5b8+frm_check_interval-%d+%s'"
		cmd = append(cmd, "-bsf:v", fmt.Sprintf(h264_metadata, interval, hook))

		if len(downsteam) > 0 {
			if strings.HasPrefix(downsteam, "rtsp://") {
				cmd = append(cmd, "-f", "rtsp", "-rtsp_flags", "prefer_tcp", downsteam)
			} else {
				cmd = append(cmd, "-f", "flv", downsteam)
			}
		} else {
			cmd = append(cmd, "-f", "null", "/dev/null")
		}
		xl.Debug("live cmd:", strings.Join(cmd, " "))
		return cmd
	}

	return newVframeV2(ctx, genCmdArgs)
}

func newVframeV2(
	ctx context.Context,
	genCmdArgs func(string) []string,
) *vframeV2 {
	v := &vframeV2{
		cuts:  list.New(),
		msgs:  make(chan *list.Element, 64),
		Mutex: new(sync.Mutex),
	}
	ctx, v.cancel = context.WithCancel(ctx)

	var (
		xl = xlog.FromContextSafe(ctx)
	)
	url, err := v.genHttpRpc(ctx, v.process)
	if err != nil {
		xl.Warnf("fail to gen http callback, error: %s", err.Error())
		v.done = 1
		v.err = err
		return v
	}

	args := genCmdArgs(url)
	go func(ctx context.Context) {
		var xl = xlog.FromContextSafe(ctx)
		var err error
		defer func() {
			err1 := recover()
			if err1 != nil {
				err = fmt.Errorf("panic: %v", err1)
				xl.Error(err)
			}
			if err != nil {
				v.Lock()
				defer v.Unlock()
				if v.err == nil {
					v.err = err
				}
			}
			v.cancel()
		}()
		for {
			cmd := exec.Command(args[0], args[1:]...)
			err = cmd.Start()
			if err != nil {
				xl.Warnf("start cmd failed. %v %v", args, err)
				break
			}
			done := make(chan error, 1)
			go func() { done <- cmd.Wait() }()
			select {
			case <-ctx.Done():
				_ = cmd.Process.Kill()
				goto ForEnd
			case err = <-done:
				xl.Warnf("cmd exit. %v %v", args, err)
				goto ForEnd // 由工具决定是否结束
			}
		}
	ForEnd:
	}(util.SpawnContext2(ctx, 1))

	return v
}

func (v *vframeV2) Next(ctx context.Context, n int) ([]CutRequest, bool) {
	reqs := make([]CutRequest, 0, n)
	v.Lock()
	select {
	case msg := <-v.msgs:
		item := msg.Value.(vframeV2Item)
		item.Waiting = true
		reqs = append(reqs, item.CutRequest)
	default:
	}
	done := v.done
	v.Unlock()
	if len(reqs) == 0 && done == 1 {
		return nil, false
	}

	if len(reqs) == 0 {
		select {
		case msg := <-v.msgs:
			item := msg.Value.(vframeV2Item)
			item.Waiting = true
			reqs = append(reqs, item.CutRequest)
		case <-ctx.Done():
			return nil, false
		}
	}

	for len(reqs) < n {
		select {
		case msg := <-v.msgs:
			item := msg.Value.(vframeV2Item)
			item.Waiting = true
			reqs = append(reqs, item.CutRequest)
		default:
			goto ForEnd
		}
	}
ForEnd:

	xlog.FromContextSafe(ctx).Debugf("next: %d|%d", len(reqs), n)
	return reqs, true
}
func (v *vframeV2) Error() error { return v.err }
func (v *vframeV2) Close()       { v.cancel() }
func (v *vframeV2) Get(ctx context.Context, offsetMS int64) CutRequest {
	v.Lock()
	defer v.Unlock()
	for e := v.cuts.Front(); e != nil; e = e.Next() {
		if e.Value.(vframeV2Item).CutRequest.OffsetMS == offsetMS {
			return e.Value.(vframeV2Item).CutRequest
		}
	}
	return CutRequest{}
}

func (v *vframeV2) Set(ctx context.Context, resp CutResponse) error {
	var ch chan CutResponse
	func() {
		v.Lock()
		defer v.Unlock()
		for e := v.cuts.Front(); e != nil; e = e.Next() {
			if e.Value.(vframeV2Item).CutRequest.OffsetMS == resp.OffsetMS {
				ch = v.cuts.Remove(e).(vframeV2Item).ch
				break
			}
		}
	}()
	if ch != nil {
		ch <- resp
	}
	return nil
}

func (v *vframeV2) process(ctx context.Context, body []byte) (interface{}, error) {
	offsetMS := time.Now().UnixNano() / int64(time.Millisecond)
	ch := make(chan CutResponse, 1)
	var e *list.Element
	func() {
		v.Lock()
		defer v.Unlock()
		e = v.cuts.PushBack(vframeV2Item{
			CutRequest: CutRequest{OffsetMS: offsetMS, Body: body},
			ch:         ch})
	}()
	v.msgs <- e
	xlog.FromContextSafe(ctx).Debugf("push cut body: %d %d", offsetMS, len(body))
	select {
	case resp := <-ch:
		return resp.Result, resp.Error
	case <-ctx.Done():
		return nil, nil
	}
}
func (v vframeV2) genHttpRpc(
	ctx context.Context,
	process func(context.Context, []byte) (interface{}, error),
) (string, error) {

	var (
		ls  net.Listener
		err error
		uri string
	)
	if ls, err = net.Listen("tcp", "127.0.0.1:0"); err != nil {
		return "", err
	}
	uri = fmt.Sprintf("http://127.0.0.1:%d", ls.Addr().(*net.TCPAddr).Port)

	srv := &http.Server{
		Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			ctx := util.SpawnContext2(ctx, int(time.Now().UnixNano()))
			body, _ := ioutil.ReadAll(r.Body)
			if len(body) == 0 {
				return
			}
			select {
			case <-ctx.Done():
				w.WriteHeader(http.StatusInternalServerError)
				return
			default:
			}
			resp, err := process(ctx, body)
			if err != nil {
				w.WriteHeader(http.StatusInternalServerError)
				return
			}
			v, _ := json.Marshal(resp)
			w.Header().Set("Content-Type", "application/json")
			w.Write(v)
		}),
	}

	go func() {
		_ = srv.Serve(ls)
	}()

	go func(ctx context.Context) {
		<-ctx.Done()
		_ = srv.Shutdown(ctx)
	}(util.SpawnContext2(ctx, 1))

	return uri, nil
}

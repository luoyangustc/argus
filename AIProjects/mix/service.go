package mix

import (
	"context"
	"encoding/base64"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"net/http"
	"os"
	"path"
	"strconv"
	"sync/atomic"
	"time"

	"github.com/qiniu/http/httputil.v1"
	"github.com/qiniu/http/restrpc.v1"
	"github.com/qiniu/rpc.v3"
	"github.com/qiniu/xlog.v1"

	"qiniu.com/argus/atserving/model"
)

type ServiceConfig struct {
	RunningMax    int32  `json:"running_max"`
	Queue         int    `json:"queue"`
	OutputFile    string `json:"output_file"`
	OutputLog     string `json:"output_log"`
	OutputMetrics string `json:"output_metrics"`
	Host          string `json:"host"`

	RecognizeTar string        `json:"recognize_tar"`
	Recognize    FaceSetConfig `json:"recognize"`
}

type Metrics struct {
	Total    uint64  `json:"total"`
	Done     uint64  `json:"done"`
	Waiting  uint64  `json:"waiting"`
	Censor   uint64  `json:"censor"`
	Skip     uint64  `json:"skip"`
	Error    uint64  `json:"error"`
	LastQPS  float64 `json:"last_qps"`
	LastRate float64 `json:"last_rate"`
	QPS      float64 `json:"qps"`
	Rate     float64 `json:"rate"`

	Normal     uint64 `json:"normal"`
	Pulp       uint64 `json:"pulp"`
	Terror     uint64 `json:"terror"`
	Politicion uint64 `json:"politicion"`
	March      uint64 `json:"march"`
	Text       uint64 `json:"text"`

	Runtime uint64 `json:"runtime"`
}

type Request struct {
	URI  model.STRING `json:"uri"`
	Meta PicMeta      `json:"meta"`
}

type Service struct {
	ServiceConfig
	Metrics

	queue chan Request
	retCH chan PicResult

	Recognize *FaceSet
}

func NewService(conf ServiceConfig) *Service {
	s := &Service{ServiceConfig: conf, retCH: make(chan PicResult, 256)}
	s.queue = make(chan Request, s.Queue)

	s.Recognize = &FaceSet{FaceSetConfig: conf.Recognize}
	if len(conf.RecognizeTar) > 0 {
		_ = s.Recognize.fetchFiles(context.Background(), conf.RecognizeTar)
	}
	_ = s.Recognize.Init(context.Background())

	_ = s.Recognize.Reload(context.Background(),
		func(ctx context.Context, body []byte) ([]byte, error) {
			xl := xlog.FromContextSafe(ctx)
			resp, err := s.postEval(ctx,
				model.STRING("data:application/octet-stream;base64,"+
					base64.StdEncoding.EncodeToString(body)),
			)
			if err != nil {
				return nil, err
			}
			ret0 := resp.(PicResp)

			if ret0.Code != 0 && ret0.Code != 200 {
				return nil, errors.New(fmt.Sprintf("bad code. %d", ret0.Code))
			}
			xl.Infof("%#v", ret0)
			if ret0.Result.Result.Face != 1 || ret0.Result.Result.Facenum == 0 {
				return nil, errors.New("no face")
			}
			return s.formatFeature(ret0.Result.Result.Faces[0].Features), nil
		})

	go s.run()
	var i int32
	for i = 0; i < s.RunningMax; i++ {
		go s.work(context.Background(), int(i))
	}
	return s
}

func (s *Service) initContext(ctx context.Context, env *restrpc.Env) (*xlog.Logger, context.Context) {
	xl, ok := xlog.FromContext(ctx)
	if !ok {
		xl = xlog.New(env.W, env.Req)
		ctx = xlog.NewContext(ctx, xl)
	}
	return xl, ctx
}

type PicMeta struct {
	PID     int    `json:"pid"`
	ZipName string `json:"zip_name"`
	BcpName string `json:"bcp_name"`
	DataID  string `json:"data_id"`
	Line    int    `json:"line"`
	PicName string `json:"pic_name"`
}

type PicResult struct {
	Code     int           `json:"code"`
	Message  string        `json:"message"`
	Meta     PicMeta       `json:"meta"`
	Skip     bool          `json:"skip"`
	Label    int           `json:"label"`
	Score    float32       `json:"score"`
	FSize    int           `json:"fsize"`
	Duration time.Duration `json:"duration"`
}
type PicResp struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Result  struct {
		Result struct {
			Normal     int     `json:"normal"`
			March      int     `json:"march"`
			Text       int     `json:"text"`
			Face       int     `json:"face"`
			BK         int     `json:"bk"`
			Pulp       int     `json:"pulp"`
			MarchSocre float32 `json:"march_score"`
			TextSocre  float32 `json:"text_score"`
			BKSocre    float32 `json:"bk_score"`
			PulpSocre  float32 `json:"pulp_score"`

			Facenum int `json:"facenum"`
			Faces   []struct {
				Pts      [][]int   `json:"pts"`
				Features []float32 `json:"features"`
			} `json:"faces"`
		} `json:"result"`
	} `json:"result"`
}

func (s *Service) run() {

	_ = os.MkdirAll(s.OutputFile, 0755)
	_ = os.MkdirAll(s.OutputLog, 0755)

	var (
		retFile    *os.File
		retCurrent time.Time

		logFile    *os.File
		logCurrent time.Time

		metricsFile *os.File

		xl = xlog.NewWith("RUN")
	)
	_ = xl

	defer func() {
		if retFile != nil {
			retFile.Close()
		}
		if logFile != nil {
			logFile.Close()
		}
	}()
	go func() {
		var (
			start = time.Now()
			err   error
			last  = s.Metrics
		)
		if s.OutputMetrics != "" {
			metricsFile, err = os.OpenFile(s.OutputMetrics, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
			if err != nil {
				xl.Warnf("fail to create metrics file, error: %v", err)
			}
			defer func() {
				if metricsFile != nil {
					metricsFile.Close()
				}
			}()
		}
		for {
			time.Sleep(5 * time.Second)
			atomic.StoreUint64(&s.Runtime, uint64(time.Since(start).Seconds()))
			atomic.StoreUint64(&s.Waiting, uint64(len(s.queue)))
			s.LastQPS = (float64)(atomic.LoadUint64(&s.Censor)-last.Censor) / 5.0
			s.QPS = (float64)(atomic.LoadUint64(&s.Censor)) / (float64)(atomic.LoadUint64(&s.Runtime))
			if s.Censor-last.Censor == 0 {
				s.LastRate = 0
			} else {
				s.LastRate = 1 - (float64)(atomic.LoadUint64(&s.Normal)-last.Normal)/(float64)(atomic.LoadUint64(&s.Censor)-last.Censor)
			}
			if s.Censor == 0 {
				s.Rate = 0
			} else {
				s.Rate = 1 - float64(atomic.LoadUint64(&s.Normal))/float64(atomic.LoadUint64(&s.Censor))
			}

			if metricsFile != nil {
				_, _ = metricsFile.Write([]byte(time.Now().Format("20060102150405")))
				_, _ = metricsFile.Write([]byte("\t"))
				_, _ = metricsFile.Write([]byte(strconv.FormatUint(atomic.LoadUint64(&s.Total), 10)))
				_, _ = metricsFile.Write([]byte("\t"))
				_, _ = metricsFile.Write([]byte(strconv.FormatUint(atomic.LoadUint64(&s.Done), 10)))
				_, _ = metricsFile.Write([]byte("\t"))
				_, _ = metricsFile.Write([]byte(strconv.FormatUint(atomic.LoadUint64(&s.Waiting), 10)))
				_, _ = metricsFile.Write([]byte("\t"))
				_, _ = metricsFile.Write([]byte(strconv.FormatUint(atomic.LoadUint64(&s.Censor), 10)))
				_, _ = metricsFile.Write([]byte("\t"))
				_, _ = metricsFile.Write([]byte(strconv.FormatUint(atomic.LoadUint64(&s.Skip), 10)))
				_, _ = metricsFile.Write([]byte("\t"))
				_, _ = metricsFile.Write([]byte(strconv.FormatUint(atomic.LoadUint64(&s.Error), 10)))
				_, _ = metricsFile.Write([]byte("\t"))
				_, _ = metricsFile.Write([]byte(strconv.FormatFloat(s.LastQPS, 'f', 3, 64)))
				_, _ = metricsFile.Write([]byte("\t"))
				_, _ = metricsFile.Write([]byte(strconv.FormatFloat(s.LastRate, 'f', 3, 64)))
				_, _ = metricsFile.Write([]byte("\t"))
				_, _ = metricsFile.Write([]byte(strconv.FormatFloat(s.QPS, 'f', 3, 64)))
				_, _ = metricsFile.Write([]byte("\t"))
				_, _ = metricsFile.Write([]byte(strconv.FormatFloat(s.Rate, 'f', 3, 64)))
				_, _ = metricsFile.Write([]byte("\t"))
				_, _ = metricsFile.Write([]byte(strconv.FormatUint(atomic.LoadUint64(&s.Normal), 10)))
				_, _ = metricsFile.Write([]byte("\t"))
				_, _ = metricsFile.Write([]byte(strconv.FormatUint(atomic.LoadUint64(&s.Pulp), 10)))
				_, _ = metricsFile.Write([]byte("\t"))
				_, _ = metricsFile.Write([]byte(strconv.FormatUint(atomic.LoadUint64(&s.Terror), 10)))
				_, _ = metricsFile.Write([]byte("\t"))
				_, _ = metricsFile.Write([]byte(strconv.FormatUint(atomic.LoadUint64(&s.Politicion), 10)))
				_, _ = metricsFile.Write([]byte("\t"))
				_, _ = metricsFile.Write([]byte(strconv.FormatUint(atomic.LoadUint64(&s.March), 10)))
				_, _ = metricsFile.Write([]byte("\t"))
				_, _ = metricsFile.Write([]byte(strconv.FormatUint(atomic.LoadUint64(&s.Text), 10)))
				_, _ = metricsFile.Write([]byte("\t"))
				_, _ = metricsFile.Write([]byte(strconv.FormatUint(atomic.LoadUint64(&s.Runtime), 10)))
				_, _ = metricsFile.Write([]byte("\t"))
				_, _ = metricsFile.Write([]byte("\n"))
			}
			last = s.Metrics
		}
	}()

	for {
		var err error
		ret := <-s.retCH

		if ret.Code != 0 && ret.Code == 200 || ret.Message != "" {
			xl.Errorf("error eval: %v", ret)
			atomic.AddUint64(&s.Error, 1)
		} else {
			atomic.AddUint64(&s.Censor, 1)
			switch ret.Label {
			case 0:
				atomic.AddUint64(&s.Normal, 1)
			case 1:
				atomic.AddUint64(&s.March, 1)
			case 2:
				atomic.AddUint64(&s.Text, 1)
			case 3:
				atomic.AddUint64(&s.Politicion, 1)
			case 4:
				atomic.AddUint64(&s.Terror, 1)
			case 5:
				atomic.AddUint64(&s.Pulp, 1)
			}
		}

		if s.OutputFile != "" {
			if retFile == nil {
				now := time.Now()
				retCurrent = time.Date(now.Year(), now.Month(), now.Day(),
					now.Hour(), now.Minute(), 0, 0, now.Location())
				retFile, err = os.OpenFile(
					path.Join(s.OutputFile, retCurrent.Format("200601021504")),
					os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
				if err != nil {
					xl.Warnf("%v", err)
				}
			} else {
				now := time.Now()
				current := time.Date(now.Year(), now.Month(), now.Day(),
					now.Hour(), now.Minute(), 0, 0, now.Location())
				if current.After(retCurrent) {
					_ = retFile.Close()
					retCurrent = current
					retFile, err = os.OpenFile(
						path.Join(s.OutputFile, retCurrent.Format("200601021504")),
						os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
					if err != nil {
						xl.Warnf("%v", err)
					}
				}
			}
			if !ret.Skip && ret.Label != 0 {
				_, _ = retFile.Write([]byte(strconv.Itoa(ret.Meta.PID)))
				_, _ = retFile.Write([]byte("\t"))
				_, _ = retFile.Write([]byte(ret.Meta.ZipName))
				_, _ = retFile.Write([]byte("\t"))
				_, _ = retFile.Write([]byte(ret.Meta.BcpName))
				_, _ = retFile.Write([]byte("\t"))
				_, _ = retFile.Write([]byte(ret.Meta.DataID))
				_, _ = retFile.Write([]byte("\t"))
				_, _ = retFile.Write([]byte(strconv.Itoa(ret.Meta.Line)))
				_, _ = retFile.Write([]byte("\t"))
				_, _ = retFile.Write([]byte(ret.Meta.PicName))
				_, _ = retFile.Write([]byte("\t"))
				_, _ = retFile.Write([]byte(strconv.Itoa(ret.Label)))
				_, _ = retFile.Write([]byte("\t"))
				_, _ = retFile.Write([]byte(strconv.FormatFloat(float64(ret.Score), 'f', 6, 64)))
				_, _ = retFile.Write([]byte("\n"))
			}
		}

		if s.OutputLog != "" {
			if logFile == nil {
				now := time.Now()
				logCurrent = time.Date(now.Year(), now.Month(), now.Day(),
					now.Hour(), 0, 0, 0, now.Location())
				logFile, _ = os.OpenFile(
					path.Join(s.OutputLog, logCurrent.Format("2006010215")),
					os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
			} else {
				now := time.Now()
				current := time.Date(now.Year(), now.Month(), now.Day(),
					now.Hour(), 0, 0, 0, now.Location())
				if current.After(logCurrent) {
					_ = logFile.Close()
					logCurrent = current
					logFile, _ = os.OpenFile(
						path.Join(s.OutputLog, logCurrent.Format("2006010215")),
						os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
				}
			}

			_, _ = logFile.Write([]byte(time.Now().Format("20060102150405")))
			_, _ = logFile.Write([]byte("\t"))
			_, _ = logFile.Write([]byte(strconv.Itoa(ret.Meta.PID)))
			_, _ = logFile.Write([]byte("\t"))
			_, _ = logFile.Write([]byte(ret.Meta.ZipName))
			_, _ = logFile.Write([]byte("\t"))
			_, _ = logFile.Write([]byte(ret.Meta.BcpName))
			_, _ = logFile.Write([]byte("\t"))
			_, _ = logFile.Write([]byte(ret.Meta.DataID))
			_, _ = logFile.Write([]byte("\t"))
			_, _ = logFile.Write([]byte(strconv.Itoa(ret.Meta.Line)))
			_, _ = logFile.Write([]byte("\t"))
			_, _ = logFile.Write([]byte(ret.Meta.PicName))
			_, _ = logFile.Write([]byte("\t"))
			_, _ = logFile.Write([]byte(strconv.FormatBool(ret.Skip)))
			_, _ = logFile.Write([]byte("\t"))
			_, _ = logFile.Write([]byte(strconv.Itoa(ret.Label)))
			_, _ = logFile.Write([]byte("\t"))
			_, _ = logFile.Write([]byte(strconv.FormatFloat(float64(ret.Score), 'f', 6, 64)))
			_, _ = logFile.Write([]byte("\t"))
			_, _ = logFile.Write([]byte(strconv.FormatInt(int64(ret.Duration), 10)))
			_, _ = logFile.Write([]byte("\t"))
			_, _ = logFile.Write([]byte(strconv.Itoa(ret.Code)))
			_, _ = logFile.Write([]byte("\t"))
			_, _ = logFile.Write([]byte(ret.Message))
			_, _ = logFile.Write([]byte("\n"))
		}

	}
}

func (s *Service) postEval(ctx context.Context, uri interface{}) (interface{}, error) {
	var (
		xl   = xlog.FromContextSafe(ctx)
		resp PicResp
		req  model.EvalRequest
	)
	req.Data.URI = uri.(model.STRING)
	cli := &rpc.Client{Client: &http.Client{Timeout: time.Second * 10}}
	if err := cli.CallWithJson(ctx, &resp, "POST", "http://"+s.Host+"/v1/eval", req); err != nil {
		xl.Errorf("[PostEval] Eval failed, %s err: %v", s.Host, err)
		return nil, err
	}
	return resp, nil
}

func (s *Service) work(ctx context.Context, worker_id int) {
	var (
		xl = xlog.FromContextSafe(ctx)
	)
	for req0 := range s.queue {
		// default ret.Label = 0 => normal
		var ret PicResult
		atomic.AddUint64(&s.Done, 1)
		ret.Meta = req0.Meta
		start := time.Now()
		resp, err := s.postEval(ctx, req0.URI)
		ret.Duration = time.Since(start)
		if err != nil {
			ret.Code, ret.Message = httputil.DetectError(err)
			s.retCH <- ret
			continue
		}
		ret0 := resp.(PicResp)
		if ret0.Code == 0 || ret0.Code == 200 {
			xl.Debugf("worker:%d, %#v", worker_id, ret0)
			if ret0.Result.Result.March == 1 {
				ret.Score = ret0.Result.Result.MarchSocre
				ret.Label = 1
			}
			if ret0.Result.Result.Text == 1 {
				ret.Score = ret0.Result.Result.TextSocre
				ret.Label = 2
			}
			if ret0.Result.Result.Face == 1 {
				if ret0.Result.Result.Facenum == 0 {
					ret.Label = 0 // Normal
				}
				for _, face := range ret0.Result.Result.Faces {
					ok, name, group, score, err1 := s.Recognize.Recognize(ctx, nil, s.formatFeature(face.Features))
					xl.Debugf("%d %v %v %v %f %v", worker_id, ok, name, group, score, err1)
					if err1 == nil && ok {
						ret.Label = 3
						ret.Score = score
						break
					}
				}
				// ret.Label = 3
			}
			if ret0.Result.Result.BK == 1 {
				ret.Score = ret0.Result.Result.BKSocre
				ret.Label = 4
			}
			if ret0.Result.Result.Pulp == 1 {
				ret.Score = ret0.Result.Result.PulpSocre
				ret.Label = 5
			}
		}
		s.retCH <- ret
	}
}

func (s *Service) PostPic(
	ctx context.Context,
	req0 *Request,
	env *restrpc.Env,
) {
	atomic.AddUint64(&s.Total, 1)
	xl := xlog.FromContextSafe(ctx)

	ret := PicResult{
		Meta: req0.Meta,
	}
	select {
	case s.queue <- *req0:
		httputil.Reply(env.W, http.StatusOK, ret)
	default:
		xl.Warnf("fail to send request to queue")
		atomic.AddUint64(&s.Skip, 1)
		ret.Skip = true
		httputil.Reply(env.W, http.StatusOK, ret)
	}
}

func (s *Service) GetMetrics(
	ctx context.Context,
	req *struct{},
	env *restrpc.Env,
) {
	httputil.Reply(env.W, http.StatusOK, s.Metrics)
}

func (s Service) formatFeature(fs []float32) []byte {
	var sum float32 = 0
	for i := 0; i < len(fs); i = i + 1 {
		sum += fs[i] * fs[i]
	}
	sum = float32(math.Sqrt(float64(sum)))
	feature := make([]byte, len(fs)*4)
	for i := 0; i < len(fs); i = i + 1 {
		f := fs[i] / sum
		binary.LittleEndian.PutUint32(feature[i*4:i*4+4], math.Float32bits(f))
	}
	return feature
}

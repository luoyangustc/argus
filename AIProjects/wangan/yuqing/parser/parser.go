package parser

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"path"
	"strings"
	"sync"
	"time"

	mgo "gopkg.in/mgo.v2"
	"gopkg.in/mgo.v2/bson"
	"qiniu.com/argus/AIProjects/wangan/yuqing"
	"qiniu.com/argus/com/uri"
	"qiniu.com/argus/com/util"

	"qiniu.com/argus/AIProjects/wangan/yuqing/kmq"

	mgoutil "github.com/qiniu/db/mgoutil.v3"
	xlog "github.com/qiniu/xlog.v1"
	"qbox.us/api/kmqcli"
	rpc "qiniupkg.com/x/rpc.v7"
)

const (
	DEFAULT_TIMEOUT = 200000000000 // 200s
)

type ParserConfig struct {
	kmqcli.Config `json:"kmq"`
	Qiniu         uri.QiniuAdminHandlerConfig `json:"qiniu"`
	Mgo           mgoutil.Config              `json:"mgo"`
	UID           uint32                      `json:"uid"`
	VideoHost     string                      `json:"video_host"`
	ImageHost     string                      `json:"image_host"`
	Worker        int                         `json:"worker"`
	Ops           []string                    `json:"ops"`
}

type Parser interface {
	Fetch(context.Context)
	Run(context.Context) error
	Parse(context.Context, yuqing.Job) yuqing.Result
	GetResult(http.ResponseWriter, *http.Request)
	Proxy(w http.ResponseWriter, req *http.Request)
}

type parser struct {
	ParserConfig
	kmq.KMQ
	queue      chan yuqing.Job
	coll       mgoutil.Collection
	opHandlers map[string]OpHandler
	handler    uri.Handler
}

var _ Parser = &parser{}

func NewParser(conf ParserConfig) (Parser, error) {
	colls := struct {
		Results mgoutil.Collection `coll:"results"`
	}{}
	session, err := mgoutil.Open(&colls, &conf.Mgo)
	if err != nil {
		return &parser{}, err
	}
	session.SetPoolLimit(100)
	colls.Results.EnsureIndex(mgo.Index{Key: []string{"uri", "source"}})

	handlers := make(map[string]OpHandler, 0)
	for _, op := range conf.Ops {
		switch op {
		case "wangan_mix":
			handlers["wangan_mix"] = NewWanganHandler(conf.ImageHost)
		default:
			return &parser{}, errors.New("invalid op")
		}
	}

	return &parser{
		ParserConfig: conf,
		handler: uri.New(
			uri.WithAdminAkSkV2(conf.Qiniu,
				func() http.RoundTripper {
					return uri.StaticHeader{
						Header: map[string][]string{"X-From-Cdn": []string{"atlab"}},
						RT:     http.DefaultTransport,
					}
				}(),
			),
		),
		KMQ:        kmq.NewKMQ(conf.Config, conf.UID),
		queue:      make(chan yuqing.Job),
		coll:       colls.Results,
		opHandlers: handlers,
	}, nil
}

type OpRequest struct {
	Op     string `json:"op"`
	Params struct {
		Labels []struct {
			Label  string  `json:"label"`
			Select int     `json:"select"` // 0x01:INGORE; 0x02:ONLY
			Score  float32 `json:"score"`
		} `json:"labels"`
		Terminate struct {
			Mode   int            `json:"mode"` // 0x01:cut; 0x02:segment
			Labels map[string]int `json:"labels"`
		} `json:"terminate"`
		IgnoreEmptyLabels bool        `json:"ignore_empty_labels"`
		Other             interface{} `json:"other"`
	} `json:"params"`
}

type VideooRequest struct {
	Data struct {
		URI string `json:"uri"`
	} `json:"data"`
	Params struct {
		Vframe struct {
			Mode     int     `json:"mode"`
			Interval float32 `json:"interval"`
		} `json:"vframe"`
	} `json:"params"`
	OPs []OpRequest `json:"ops"`
}

type OpResponse struct {
	Labels   []yuqing.OpResultLable `json:"labels"`
	Segments []struct {
		OP   string `json:"op,omitempty"`
		Cuts []struct {
			Offset int64       `json:"offset"`
			URI    string      `json:"uri,omitempty"`
			Result interface{} `json:"result"`
		} `json:"cuts,omitempty"`
	} `json:"segments"`
}

type VideoResponse map[string]OpResponse

func qhash(ctx context.Context, uri string) string {
	var ret struct {
		Hash string `json:"hash"`
	}

	cli := rpc.Client{
		Client: &http.Client{
			Timeout: 200000000000,
		},
	}
	resp, err := cli.Get(uri + "?qhash/md5")
	if err != nil || resp.StatusCode/100 != 2 {
		return ""
	}
	err = rpc.CallRet(ctx, &ret, resp)
	if err != nil {
		return ""
	}
	return ret.Hash
}

func (p *parser) Fetch(ctx context.Context) {
	var (
		xl       = xlog.FromContextSafe(ctx)
		position = "@"
	)
	for {
		messages, next, err := p.KMQ.Consume(ctx, position, 100)
		if err != nil {
			xl.Errorf("failed to consume kmq, error: %s", err.Error())
			continue
		}
		position = next
		if len(messages) == 0 {
			time.Sleep(2 * time.Second)
			continue
		}
		for _, msg := range messages {
			p.queue <- msg
		}
	}
}

func (p *parser) Run(ctx context.Context) error {
	var (
		wg sync.WaitGroup
	)
	for i := 0; i < p.Worker; i++ {
		wg.Add(1)
		go func(ctx context.Context) {
			defer wg.Done()
			xl := xlog.FromContextSafe(ctx)
			coll := p.coll.CopySession()
			defer coll.CloseSession()

			for job := range p.queue {
				var job1 yuqing.Job
				if err := coll.Find(bson.M{"uri": job.URI, "source": job.Source}).One(&job1); err == nil {
					continue
				}
				result := p.Parse(ctx, job)
				result.ID = bson.NewObjectId()

				if len(result.Ops) > 0 && result.Error == "" {
					coll.Insert(result)
					xl.Infof("%v  %v", result.URI, result.Ops)
				}
				if result.Error != "" {
					xl.Warnf("req: %v, result: %v", job, result)
				}
			}
		}(util.SpawnContext2(ctx, i))
	}
	wg.Wait()
	return errors.New("parser worker done")
}

// 根据秒拍vid，获取视频cdn访问链接，带签名
func parseMiaopai(ctx context.Context, vid string) (rawUri string, videoUri string, message yuqing.MiaopaiMessage, err error) {
	url := fmt.Sprintf("http://gslb.miaopai.com/stream/%s.json?token=", vid)
	var resp1 struct {
		Status  int    `json:"status"`
		Message string `json:"msg"`
		Result  []struct {
			Host   string `json:"host"`
			Name   string `json:"name"`
			Path   string `json:"path"`
			Scheme string `json:"scheme"`
			Sign   string `json:"sign"`
		} `json:"result"`
	}

	client := &rpc.Client{
		Client: &http.Client{
			Timeout: 30 * time.Second,
		},
	}
	err = client.CallWithJson(ctx, &resp1, "GET", url, nil)
	if err != nil {
		return
	}
	if len(resp1.Result) == 0 {
		err = errors.New("no valid miaopai cdn address")
		return
	}
	result := resp1.Result[0]
	rawUri = result.Scheme + result.Host + result.Path
	videoUri = "http://n.miaopai.com/media/" + vid

	url = fmt.Sprintf("https://n.miaopai.com/api/aj_media/info.json?smid=%s&appid=530", vid)
	var resp2 struct {
		Code    int    `json:"code"`
		Message string `json:"msg"`
		Data    struct {
			SMID        string `json:"smid"`
			Description string `json:"description"`
			CreatedAt   int64  `json:"created_at"`
			User        struct {
				SUID     string `json:"suid"`
				Nick     string `json:"nick"`
				Desc     string `json:"desc"`
				Birthday string `json:"birthday"`
			} `json:"user"`
			Meta []struct {
				SVID       string `json:"svid"`
				ViewsCount int    `json:"views_count"`
				Pics       struct {
					Webp string `json:"webp"`
				} `json:"pics"`
			} `json:"meta_data"`
		} `json:"data"`
	}
	err = client.CallWithJson(ctx, &resp2, "GET", url, nil)
	if err != nil || resp2.Code != 200 || resp2.Message != "" {
		err = errors.New(fmt.Sprintf("miaopai errors: %v, msg: %s", err, resp2.Message))
		return
	}
	message.VID = vid
	message.SMID = resp2.Data.SMID
	message.Description = resp2.Data.Description
	message.CreatedAt = time.Unix(resp2.Data.CreatedAt, 0)
	if len(resp2.Data.Meta) > 0 {
		message.ViewsCount = resp2.Data.Meta[0].ViewsCount
	}
	message.Cover = resp2.Data.Meta[0].Pics.Webp
	message.UserNick = resp2.Data.User.Nick
	message.SUID = resp2.Data.User.SUID
	message.Birthday = resp2.Data.User.Birthday

	return
}

func (p *parser) Parse(ctx context.Context, job yuqing.Job) yuqing.Result {

	var (
		result = yuqing.Result{
			URI:       job.URI,
			Source:    job.Source,
			Message:   job.Message,
			Type:      job.Type,
			ParseTime: time.Now(),
			Ops:       make(map[string]yuqing.OpResult, 0),
		}
		start = time.Now()
	)

	if job.Source == yuqing.SourceTypeMiaopai {
		var (
			err error
			msg yuqing.MiaopaiMessage
		)
		vid := strings.TrimSuffix(path.Base(job.URI), path.Ext(job.URI))
		if job.URI, result.URI, msg, err = parseMiaopai(ctx, vid); err != nil {
			result.Error = err.Error()
			result.EvalDuration = time.Since(start).Seconds()
			return result
		}
		result.Message = msg
		job.Message = msg
	}

	for op, handler := range p.opHandlers {
		var (
			ret yuqing.Result
			err error
		)
		if job.Type == yuqing.MediaTypeVideo {
			ret, err = p.video(ctx, job)
		} else {
			ret, err = handler.ImageParse(ctx, job)
		}
		if err != nil {
			result.Error = err.Error()
			result.EvalDuration = time.Since(start).Seconds()
		}
		if _, ok := ret.Ops[op]; ok {
			result.Ops[op] = ret.Ops[op]
		}
		if ret.Score > result.Score {
			result.Score = ret.Score
		}
		result.Fsize = ret.Fsize
		result.MimeType = ret.MimeType
		result.MD5 = ret.MD5
	}
	return result
}

func (p *parser) video(ctx context.Context, job yuqing.Job) (yuqing.Result, error) {
	cli := rpc.Client{
		Client: &http.Client{
			Timeout: DEFAULT_TIMEOUT,
		},
	}

	var (
		xl   = xlog.FromContextSafe(ctx)
		resp VideoResponse
		req  VideooRequest
		ret  = yuqing.Result{
			Ops: make(map[string]yuqing.OpResult, 0),
		}
	)
	req.Data.URI = job.URI
	req.Params.Vframe.Interval = 2.0
	for _, handler := range p.opHandlers {
		req.OPs = append(req.OPs, handler.GetVideoRequest())
	}

	url := p.VideoHost + "/v1/video/" + xlog.GenReqId()
	err := cli.CallWithJson(ctx, &resp, "POST", url, req)
	if err != nil {
		xl.Warnf("call wange gate failed, req: %v, error: %s", req, err.Error())
		return yuqing.Result{}, err
	}

	for op, result := range resp {
		r := p.opHandlers[op].ParseVideoRespnse(result, job.URI)
		if len(r.Labels) > 0 {
			ret.Ops[op] = r
		}
		for _, label := range ret.Ops[op].Labels {
			if label.Score > ret.Score {
				ret.Score = label.Score
			}
		}
	}
	return ret, nil
}

// ---------------------------------------------------------------------- //
type OpHandler interface {
	GetVideoRequest() OpRequest
	ParseVideoRespnse(OpResponse, string) yuqing.OpResult
	ImageParse(context.Context, yuqing.Job) (yuqing.Result, error)
}

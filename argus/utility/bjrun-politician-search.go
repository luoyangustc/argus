package utility

import (
	"bytes"
	"context"
	"crypto/md5"
	"encoding/base64"
	"encoding/binary"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"time"

	"qiniu.com/argus/utility/evals"

	"github.com/qiniu/db/mgoutil.v3"
	"github.com/qiniu/errors"
	"github.com/qiniu/http/httputil.v1"
	"github.com/qiniu/xlog.v1"
	"labix.org/v2/mgo/bson"
	"qiniu.com/auth/authstub.v1"
)

// FaceSearchReq ...
type FaceSearchReq struct {
	Data struct {
		URI string `json:"uri"`
	} `json:"data"`
}

// FaceSearchResp ...
type FaceSearchResp struct {
	Code    int              `json:"code"`
	Message string           `json:"message"`
	Result  FaceSearchResult `json:"result"`
}

// FaceSearchResult ...
type FaceSearchResult struct {
	Review     bool               `json:"review"`
	Detections []FaceSearchDetail `json:"detections"`
}

// FaceSearchDetail ...
type FaceSearchDetail struct {
	BoundingBox FaceDetectBox `json:"boundingBox"`
	Value       struct {
		Name   string  `json:"name,omitempty"`
		Score  float32 `json:"score"`
		Review bool    `json:"review"`
	} `json:"value"`
	Sample *FaceSearchDetailSample `json:"sample,omitempty"`
}

type FaceSearchDetailSample struct {
	URL string   `json:"url"`
	Pts [][2]int `json:"pts"`
}

//----------------------------------------------------------------------------//

type _EvalFacexSearchReq struct {
	Data []struct {
		URI string `json:"uri"`
	} `json:"data"`
	Params struct {
		Threshold float32 `json:"threshold"`
	} `json:"params"`
}

type _EvalFacexSearchResp struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Result  struct {
		Index int     `json:"index"`
		Score float32 `json:"score"`
	} `json:"result"`
}

type iFacexSearch interface {
	Eval(context.Context, _EvalFacexSearchReq, _EvalEnv) (_EvalFacexSearchResp, error)
}

type _FacexSearch struct {
	url     string
	timeout time.Duration
}

func newFacexSearch(host string, timeout time.Duration) _FacexSearch {
	return _FacexSearch{url: host + "/v1/eval/facex-search", timeout: timeout}
}

func (fm _FacexSearch) Eval(
	ctx context.Context, req _EvalFacexSearchReq, env _EvalEnv,
) (_EvalFacexSearchResp, error) {
	var (
		client = newRPCClient(env, fm.timeout)

		resp _EvalFacexSearchResp
	)
	err := client.CallWithJson(ctx, &resp, "POST", fm.url, &req)
	return resp, err
}

type BjrunPoliticianSearchReq struct {
	Data struct {
		URI string `json:"uri"`
	} `json:"data"`
	Params struct {
		Filters []string `json:"filters,omitempty"`
	} `json:"params"`
}

//----------------------------------------------------------------------------//

type _EvalBjrunImageSearchReq struct {
	Data []struct {
		URI string `json:"uri"`
	} `json:"data"`
	Params struct {
		Limit *int `json:"limit,omitempty"`
	} `json:"params"`
}

type _EvalBjrunImageSearchResp struct {
	Code    int                           `json:"code"`
	Message string                        `json:"message"`
	Result  []_EvalBjrunImageSearchResult `json:"result"`
}
type _EvalBjrunImageSearchResult struct {
	Index int     `json:"index"`
	Score float32 `json:"score"`
}
type iBjrunImageSearch interface {
	Eval(context.Context, _EvalBjrunImageSearchReq, _EvalEnv) (_EvalBjrunImageSearchResp, error)
}

type _BjrunImageSearch struct {
	url     string
	timeout time.Duration
}

func newBjrunImageSearch(host string, timeout time.Duration) _BjrunImageSearch {
	return _BjrunImageSearch{url: host + "/v1/eval/image-search", timeout: timeout}
}

func (fm _BjrunImageSearch) Eval(
	ctx context.Context, req _EvalBjrunImageSearchReq, env _EvalEnv,
) (_EvalBjrunImageSearchResp, error) {
	var (
		client = newRPCClient(env, fm.timeout)

		resp _EvalBjrunImageSearchResp
	)
	err := client.CallWithJson(ctx, &resp, "POST", fm.url, &req)
	return resp, err
}

type BjrunImageSearchReq struct {
	Data struct {
		URI string `json:"uri"`
	} `json:"data"`
	Params struct {
		Limit   int      `json:"limit,omitempty"`
		Filters []string `json:"filters,omitempty"`
	} `json:"params,omitempty"`
}

// BjrunImageSearchResp ...
type BjrunImageSearchResp struct {
	Code    int                      `json:"code"`
	Message string                   `json:"message"`
	Result  []BjrunImageSearchResult `json:"result"`
}

// BjrunImageSearchResult ...
type BjrunImageSearchResult struct {
	URL   string  `json:"url"`
	Label string  `json:"label,omitempty"`
	Score float32 `json:"score"`
}

//----------------------------------------------------------------------------//

type BjrunCollections struct {
	Politicians mgoutil.Collection `coll:"Politicians"`
	Images      mgoutil.Collection `coll:"Images"`
}

var bjrcollections BjrunCollections

func initDB(cfg *mgoutil.Config) error {
	sess, err := mgoutil.Open(&bjrcollections, cfg)
	if err != nil {
		xlog.Errorf("", "init database failed:%v", err)
		return err
	}
	sess.SetPoolLimit(DefaultCollSessionPoolLimit)
	return err
}

type GeneralResp struct {
	Code    int         `json:"code"`
	Message string      `json:"message"`
	Result  interface{} `json:"result"`
}

func isBinaryData(dt string) bool {
	return strings.HasPrefix(dt, Base64Header)
}

func Base64Data2MD5(dt string) (string, error) {
	data64 := strings.TrimPrefix(dt, Base64Header)
	origin, err := base64.StdEncoding.DecodeString(data64)
	if err != nil {
		return "", err
	}
	md5Ctx := md5.New()
	md5Ctx.Write([]byte(origin))
	md5Sum := md5Ctx.Sum(nil)
	return base64.StdEncoding.EncodeToString(md5Sum[:]), nil
}

func (s *Service) PostBjrunPoliticianAdd(ctx context.Context,
	args *struct {
		Data []struct {
			URI       string `json:"uri"`
			Attribute struct {
				Name string `json:"name"`
			} `json:"attribute"`
		} `json:"data"`
	}, env *authstub.Env) {

	var (
		ctex, xl     = ctxAndLog(ctx, env.W, env.Req)
		ret          GeneralResp
		imgLocker    sync.Mutex
		politLocker  sync.Mutex
		waiter       sync.WaitGroup
		uid          = env.UserInfo.Uid
		utype        = env.UserInfo.Utype
		failedImages struct {
			Failed []string `json:"failed"`
		}
		newPolit []Politician
	)
	xl.Infof("PostBjrunPoliticianAdd args:%v", args)
	if len(args.Data) == 0 || len(args.Data) > 20 {
		httputil.ReplyErr(env.W, ErrArgs.Code, ErrArgs.Error())
		return
	}

	waiter.Add(len(args.Data))
	for _, item := range args.Data {
		go func(uri, name string, ctx context.Context) {
			defer waiter.Done()
			xl := xlog.FromContextSafe(ctx)
			if strings.TrimSpace(uri) == "" {
				return
			}
			if strings.TrimSpace(name) == "" {
				imgLocker.Lock()
				failedImages.Failed = append(failedImages.Failed, uri+": empty name")
				imgLocker.Unlock()
				return
			}

			var fdReq _EvalFaceDetectReq
			fdReq.Data.URI = uri
			fdResp, err := s.iFaceDetect.Eval(ctx, fdReq, _EvalEnv{Uid: uid, Utype: utype})
			if err != nil {
				xl.Errorf("PostBjrunPoliticianAdd query facex detect error:%v", err)
				imgLocker.Lock()
				failedImages.Failed = append(failedImages.Failed, uri)
				imgLocker.Unlock()
				return
			}
			if len(fdResp.Result.Detections) != 1 {
				xl.Errorf("PostBjrunPoliticianAdd %v face detected", len(fdResp.Result.Detections))
				imgLocker.Lock()
				failedImages.Failed = append(failedImages.Failed, uri+": invalid face number")
				imgLocker.Unlock()
				return
			}
			var ffReq _EvalFaceReq
			ffReq.Data.URI = uri
			ffReq.Data.Attribute.Pts = fdResp.Result.Detections[0].Pts
			ffResp, err := s.iFaceFeatureV2.Eval(ctx, ffReq, _EvalEnv{Uid: uid, Utype: utype})
			if err != nil {
				xl.Errorf("PostBjrunPoliticianAdd query facex detect error:%v", err)
				imgLocker.Lock()
				failedImages.Failed = append(failedImages.Failed, uri)
				imgLocker.Unlock()
				return
			}

			if !isBinaryData(uri) {
				politLocker.Lock()
				newPolit = append(newPolit, Politician{
					Name:    name,
					Url:     uri,
					Feature: ffResp,
				})
				politLocker.Unlock()
				return
			}

			MD5ID, err := Base64Data2MD5(uri)
			if err != nil {
				xl.Errorf("PostBjrunPoliticianAdd query Base64Data2MD5 error:%v", err)
				imgLocker.Lock()
				failedImages.Failed = append(failedImages.Failed, uri)
				imgLocker.Unlock()
				return
			}
			politLocker.Lock()
			newPolit = append(newPolit, Politician{
				Name:    name,
				ID:      MD5ID,
				Feature: ffResp,
			})
			politLocker.Unlock()
		}(item.URI, item.Attribute.Name, spawnContext(ctex))
	}
	waiter.Wait()
	if len(failedImages.Failed) == len(args.Data) {
		httputil.ReplyErr(env.W, http.StatusInternalServerError, "failed")
		return
	}
	if err := s._BjrunPoliticianManager.AddPolitician(ctex, newPolit...); err != nil {
		xl.Errorf("_BjrunPoliticianManager.AddPolitician error:%v", errors.Detail(err))
		httputil.ReplyErr(env.W, http.StatusInternalServerError, "add to politician database error")
		return
	}
	ret.Result = failedImages
	ret.Message = "success"
	httputil.Reply(env.W, http.StatusOK, ret)
}

func (b *Service) GetBjrunPoliticianList(ctx context.Context, env *authstub.Env) {
	var (
		ctex, xl = ctxAndLog(ctx, env.W, env.Req)
		ret      GeneralResp
		polit    struct {
			Politicians []string `json:"politicians"`
		}
	)
	dbret, err := b._BjrunPoliticianManager.Politicians(ctex)
	if err != nil {
		xl.Errorf("GetBjrunPoliticianList query database error:%v", err)
		httputil.ReplyErr(env.W, http.StatusInternalServerError, "query database error")
		return
	}
	ret.Message = "success"
	polit.Politicians = dbret
	ret.Result = polit
	httputil.Reply(env.W, http.StatusOK, ret)
}
func (s *Service) PostBjrunPoliticianImages(ctx context.Context, args *struct {
	Params *struct {
		Name string `json:"name"`
	} `json:"params"`
}, env *authstub.Env) {
	var (
		ctex, xl = ctxAndLog(ctx, env.W, env.Req)
		ret      GeneralResp
		imgList  struct {
			Images []string `json:"images"`
		}
	)
	dbret, err := s._BjrunPoliticianManager.PoliticianImages(ctex, args.Params.Name)
	if err != nil {
		xl.Errorf("PostBjrunPoliticianImages error:%v", errors.Detail(err))
		httputil.ReplyErr(env.W, http.StatusInternalServerError, "query database error")
		return
	}
	ret.Message = "success"
	imgList.Images = dbret
	ret.Result = imgList
	httputil.Reply(env.W, http.StatusOK, ret)
}
func (s *Service) PostBjrunPoliticianDel(ctx context.Context, args *struct {
	Data []struct {
		URI       string `json:"uri"`
		Attribute struct {
			Name string `json:"name"`
		} `json:"attribute"`
	} `json:"data"`
}, env *authstub.Env) {

	var (
		ctex, xl = ctxAndLog(ctx, env.W, env.Req)
		waiter   sync.WaitGroup
		err      error
	)

	xl.Infof("PostBjrunPoliticianAdd args:%v", args)
	if len(args.Data) == 0 || len(args.Data) > 20 {
		httputil.ReplyErr(env.W, ErrArgs.Code, ErrArgs.Error())
		return
	}

	waiter.Add(len(args.Data))
	for _, item := range args.Data {
		go func(uri, name string, ctx context.Context) {
			defer waiter.Done()
			if strings.TrimSpace(name) == "" {
				return
			}
			_err := s._BjrunPoliticianManager.Delete(ctx, name, uri)
			if _err != nil {
				err = _err
			}
		}(item.URI, item.Attribute.Name, spawnContext(ctex))
	}
	waiter.Wait()
	if err != nil {
		xl.Errorf("PostBjrunPoliticianDel query database error:%v", err)
		httputil.ReplyErr(env.W, http.StatusInternalServerError, "query database error")
		return
	}

	httputil.Reply(env.W, http.StatusOK, "success")
}

func (s *Service) PostBjrunPoliticianSearch(ctx context.Context, args *BjrunPoliticianSearchReq, env *authstub.Env) (ret *FaceSearchResp, err error) {
	var (
		ctex, xl = ctxAndLog(ctx, env.W, env.Req)
		uid      = env.UserInfo.Uid
		utype    = env.UserInfo.Utype

		lock   sync.Mutex
		waiter sync.WaitGroup
	)
	if strings.TrimSpace(args.Data.URI) == "" {
		return nil, ErrArgs
	}

	ret = &FaceSearchResp{}

	var fdResp _EvalFaceDetectResp
	var fdReq _EvalFaceDetectReq
	fdReq.Data.URI = args.Data.URI
	fdResp, err = s.iFaceDetect.Eval(ctx, fdReq, _EvalEnv{Uid: uid, Utype: utype})
	if err != nil {
		xl.Errorf("PostBjrunPoliticianSearch query facex detect error:%v", err)
		return nil, err
	}
	if len(fdResp.Result.Detections) == 0 {
		xl.Error("PostBjrunPoliticianSearch no face detected")
		return nil, httputil.NewError(http.StatusNotAcceptable, "no face detected")
	}

	polits, features, err := s._BjrunPoliticianManager.All(ctex)
	if err != nil {
		xl.Errorf("PostBjrunPoliticianSearch query database error:%v", err)
		return nil, err
	}
	if len(polits) == 0 {
		xl.Errorf("no valid politician data in dynamic database")
	}

	waiter.Add(len(fdResp.Result.Detections))

	for _, dt := range fdResp.Result.Detections {
		go func(det _EvalFaceDetection, ctx context.Context) {
			defer waiter.Done()

			var _inWaiter sync.WaitGroup
			var ffReq _EvalFaceReq
			ffReq.Data.URI = args.Data.URI
			ffReq.Data.Attribute.Pts = det.Pts
			ffResp, _err := s.iFaceFeatureV2.Eval(ctx, ffReq, _EvalEnv{Uid: uid, Utype: utype})
			if _err != nil {
				xl.Errorf("PostBjrunPoliticianSearch query facex detect error:%v", _err)
				return
			}
			var mReq = _EvalFacexSearchReq{
				Data: make([]struct {
					URI string `json:"uri"`
				}, 2),
			}
			var mResp _EvalFacexSearchResp
			var _errFs error
			if len(polits) != 0 {
				_inWaiter.Add(1)
				mReq.Params.Threshold = 0.525
				mReq.Data[0].URI = features
				mReq.Data[1].URI = "data:application/octet-stream;base64," +
					base64.StdEncoding.EncodeToString(ffResp)
				go func(ctx context.Context) {
					defer _inWaiter.Done()
					mResp, _errFs = s.iFacexSearch.Eval(ctx, mReq, _EvalEnv{Uid: uid, Utype: utype})
				}(spawnContext(ctx))
			}

			if s.Config.PoliticianFeatureVersion == "v3" {
				var pFreq evals.FaceReq
				pFreq.Data.URI = ffReq.Data.URI
				pFreq.Data.Attribute.Pts = ffReq.Data.Attribute.Pts
				ffResp, _err = s.facexFeatureV3.Eval(ctx, pFreq, uid, utype)
				if _err != nil {
					xl.Errorf("PostBjrunPoliticianSearch query facex feature-v3 error:%v", _err)
					return
				}
			}
			var (
				pReq = _EvalFaceSearchReq{
					Data: struct {
						URI string `json:"uri"`
					}{URI: "data:application/octet-stream;base64," +
						base64.StdEncoding.EncodeToString(ffResp)},
				}
				_errPs     error
				pRespName  string
				pRespScore float32
			)
			_inWaiter.Add(1)
			go func(ctx context.Context) { //statistic library
				defer _inWaiter.Done()
				var req evals.SimpleReq

				req.Data.URI = pReq.Data.URI
				pResp, _errPs := s.politician.Eval(ctx, req, uid, utype)

				if fpResp, ok := pResp.(evals.FaceSearchRespV2); ok {
					if _errPs == nil && len(fpResp.Result.Confidences) > 0 {
						pRespName = fpResp.Result.Confidences[0].Class
						pRespScore = fpResp.Result.Confidences[0].Score
					}
				} else if fpResp, ok := pResp.(evals.FaceSearchResp); ok && _errPs == nil {
					pRespName = fpResp.Result.Class
					pRespScore = fpResp.Result.Score
				}
			}(spawnContext(ctx))
			_inWaiter.Wait()

			if _errPs != nil || _errFs != nil {
				xl.Errorf("PostBjrunPoliticianSearch iFaceSearchPolitician error:%v,iFacexSearch error:%v", _errPs, _errFs)
			}
			if _errPs != nil && _errFs != nil {
				return
			}

			detail := FaceSearchDetail{
				BoundingBox: FaceDetectBox{
					Pts:   det.Pts,
					Score: det.Score,
				},
			}
			if mResp.Result.Score > pRespScore {
				if mResp.Result.Score > s.Config.PoliticianThreshold[1] {
					detail.Value.Name = polits[mResp.Result.Index].Name
				}
				detail.Value.Score = mResp.Result.Score
			} else {
				if pRespName != "" && pRespScore > s.Config.PoliticianThreshold[1] {
					detail.Value.Name = pRespName
				}
				detail.Value.Score = pRespScore
			}

			var isF bool
			if len(args.Params.Filters) == 0 {
				isF = true
			} else {
				for _, name := range args.Params.Filters {
					if detail.Value.Name == name {
						isF = true
						break
					}
				}
			}

			if isF && detail.Value.Score > s.Config.PoliticianThreshold[0] && detail.Value.Score < s.Config.PoliticianThreshold[2] {
				detail.Value.Review = true
				ret.Result.Review = true
			}
			if detail.Value.Score > 1.0 {
				detail.Value.Score = 1.0
			}

			lock.Lock()
			defer lock.Unlock()
			if isF {
				ret.Result.Detections = append(ret.Result.Detections, detail)
			}

		}(dt, spawnContext(ctex))
	}
	waiter.Wait()
	if len(ret.Result.Detections) == 0 {
		ret.Message = "no face detected"
	}
	return
}

//PoliticianManager
//-------------------------------------------------------------//

type _BjrunPoliticianManager interface {
	AddPolitician(context.Context, ...Politician) error
	Politicians(context.Context) ([]string, error)
	PoliticianImages(context.Context, ...string) ([]string, error)
	Delete(context.Context, string, string) error
	All(context.Context) ([]Politician, string, error)
}

type bjrunPoliticianDB struct {
	db *mgoutil.Collection

	polits   []Politician
	features string
	*sync.Mutex
	readMutex *sync.Mutex
}

func NewBjrunPoliticianManager() (_BjrunPoliticianManager, error) {
	return &bjrunPoliticianDB{
		db:        &bjrcollections.Politicians,
		Mutex:     new(sync.Mutex),
		readMutex: new(sync.Mutex),
	}, nil
}

type Politician struct {
	Name    string `bson:"name"`
	Url     string `bson:"url,omitempty"`
	ID      string `bson:"id,omitempty"`
	Feature []byte `bson:"feature"`
}

func (b bjrunPoliticianDB) AddPolitician(ctx context.Context, pts ...Politician) error {
	c := b.db.CopySession()
	defer c.CloseSession()
	var intermediate []interface{}
	for _, p := range pts {
		intermediate = append(intermediate, p)
	}
	return c.Insert(intermediate...)
}

func (b bjrunPoliticianDB) Politicians(ctx context.Context) ([]string, error) {
	c := b.db.CopySession()
	defer c.CloseSession()

	var (
		polits []string
	)
	err := c.Find(bson.M{}).Select(bson.M{"_id": 0, "url": 0, "id": 0, "feature": 0}).Distinct("name", &polits)

	return polits, err
}

func (b bjrunPoliticianDB) PoliticianImages(ctx context.Context, polits ...string) ([]string, error) {
	c := b.db.CopySession()
	defer c.CloseSession()

	type ImageList struct {
		Url string `bson:"url"`
		ID  string `bson:"id"`
	}
	var (
		uri []ImageList
		ret []string
	)
	err := c.Find(bson.M{"name": bson.M{"$in": polits}}).Select(bson.M{"_id": 0, "name": 0, "feature": 0}).All(&uri)
	if err == nil {
		for _, u := range uri {
			if u.Url == "" {
				ret = append(ret, u.ID)
			} else {
				ret = append(ret, u.Url)
			}
		}
	}
	return ret, err
}

func (b bjrunPoliticianDB) Delete(ctx context.Context, name string, uri string) error {
	c := b.db.CopySession()
	defer c.CloseSession()

	if uri == "" {
		_, err := c.RemoveAll(bson.M{"name": name})
		return err
	}
	_, err := c.RemoveAll(bson.M{"name": name, "$or": []bson.M{bson.M{"url": uri}, bson.M{"id": uri}}})
	return err
}

func (b *bjrunPoliticianDB) All(context.Context) ([]Politician, string, error) {
	c := b.db.CopySession()
	defer c.CloseSession()

	b.Lock()
	defer b.Unlock()

	if b.polits != nil {
		n, err := c.Find(bson.M{}).Select(bson.M{"_id": 0}).Count()
		if err != nil {
			return nil, "", err
		}
		if n == len(b.polits) {
			return b.polits, b.features, nil
		}
	}

	go func() {
		b.readMutex.Lock()
		defer b.readMutex.Unlock()

		c := b.db.CopySession()
		defer c.CloseSession()

		n, _ := c.Find(bson.M{}).Select(bson.M{"_id": 0}).Count()
		b.Lock()
		if b.polits != nil && len(b.polits) == n {
			b.Unlock()
			return
		}
		b.Unlock()

		var polits []Politician
		err := c.Find(bson.M{}).Select(bson.M{"_id": 0}).All(&polits)
		if err != nil {
			return
			// return nil, err
		}

		features, _ := func() (string, error) {
			{
				if len(polits) == 0 {
					return "", errors.New("no valid politician data in database")
				}
			}

			var (
				buf     = bytes.NewBuffer(nil)
				base64W = base64.NewEncoder(base64.StdEncoding, buf)
				b4      = make([]byte, 4)
			)

			binary.LittleEndian.PutUint32(b4, uint32(len(polits[0].Feature)))
			base64W.Write(b4)

			for _, item := range polits {
				if len(item.Feature) < 5 {
					base64W.Close()
					// xl.Errorf("PostBjrunPoliticianSearch feature size smaller than 5, politican:%v", item)
					return "", httputil.NewError(http.StatusInternalServerError, "terrible backend error")
				}
				base64W.Write(item.Feature)
			}
			base64W.Close()
			return "data:application/octet-stream;base64," + buf.String(), nil
		}()

		b.Lock()
		defer b.Unlock()
		b.polits = polits
		b.features = features
	}()
	return b.polits, b.features, nil
}

//ImageManager
//-------------------------------------------------------------//

type _BjrunImageManager interface {
	AddImage(context.Context, ...Image) error
	Labels(context.Context) ([]string, error)
	LabelImages(context.Context, ...string) ([]string, error)
	Delete(context.Context, string, string) error
	All(context.Context) ([]Image, string, error)
}
type Image struct {
	Label   string `bson:"label"`
	Url     string `bson:"url,omitempty"`
	ID      string `bson:"id,omitempty"`
	Feature []byte `bson:"feature"`
}

type bjrunImageDB struct {
	db        *mgoutil.Collection
	images    []Image
	features  string
	readMutex *sync.Mutex
	*sync.Mutex
}

func NewBjrunImageManager() (_BjrunImageManager, error) {
	return &bjrunImageDB{
		db:        &bjrcollections.Images,
		readMutex: new(sync.Mutex),
		Mutex:     new(sync.Mutex),
	}, nil
}

func (b *bjrunImageDB) All(context.Context) ([]Image, string, error) {
	c := b.db.CopySession()
	defer c.CloseSession()

	b.Lock()
	defer b.Unlock()

	if b.images != nil {
		n, err := c.Find(bson.M{}).Select(bson.M{"_id": 0}).Count()
		if err != nil {
			return nil, "", err
		}
		if n == len(b.images) {
			return b.images, b.features, nil
		}
	}

	go func() {
		b.readMutex.Lock()
		defer b.readMutex.Unlock()

		c := b.db.CopySession()
		defer c.CloseSession()

		n, _ := c.Find(bson.M{}).Select(bson.M{"_id": 0}).Count()
		b.Lock()
		if b.images != nil && len(b.images) == n {
			b.Unlock()
			return
		}
		b.Unlock()

		var images []Image
		err := c.Find(bson.M{}).Select(bson.M{"_id": 0}).All(&images)
		if err != nil {
			return
			// return nil, err
		}

		features, _ := func() (string, error) {
			{
				if len(images) == 0 {
					return "", errors.New("no valid politician data in database")
				}
			}

			var (
				buf     = bytes.NewBuffer(nil)
				base64W = base64.NewEncoder(base64.StdEncoding, buf)
				b4      = make([]byte, 4)
			)

			binary.LittleEndian.PutUint32(b4, uint32(len(images[0].Feature)))
			base64W.Write(b4)

			for _, item := range images {
				if len(item.Feature) < 5 {
					base64W.Close()
					// xl.Errorf("PostBjrunPoliticianSearch feature size smaller than 5, politican:%v", item)
					return "", httputil.NewError(http.StatusInternalServerError, "terrible backend error")
				}
				base64W.Write(item.Feature)
			}
			base64W.Close()
			return "data:application/octet-stream;base64," + buf.String(), nil
		}()

		b.Lock()
		defer b.Unlock()
		b.images = images
		b.features = features
	}()
	return b.images, b.features, nil
}

func (b bjrunImageDB) AddImage(ctx context.Context, pts ...Image) error {
	c := b.db.CopySession()
	defer c.CloseSession()
	var intermediate []interface{}
	for _, p := range pts {
		intermediate = append(intermediate, p)
	}
	return c.Insert(intermediate...)
}

func (b bjrunImageDB) Labels(ctx context.Context) ([]string, error) {
	c := b.db.CopySession()
	defer c.CloseSession()

	var (
		lb []string
	)
	err := c.Find(bson.M{}).Select(bson.M{"_id": 0, "url": 0, "id": 0, "feature": 0}).Distinct("label", &lb)
	return lb, err
}

func (b bjrunImageDB) LabelImages(ctx context.Context, labels ...string) ([]string, error) {
	c := b.db.CopySession()
	defer c.CloseSession()

	var (
		uri []struct {
			Url string `bson:"url"`
			ID  string `bson:"id"`
		}
		ret []string
	)
	err := c.Find(bson.M{"label": bson.M{"$in": labels}}).Select(bson.M{"_id": 0, "feature": 0, "label": 0}).All(&uri)
	if err == nil {
		for _, u := range uri {
			if u.Url == "" {
				ret = append(ret, u.ID)
			} else {
				ret = append(ret, u.Url)
			}
		}
	}
	return ret, err
}

func (b bjrunImageDB) Delete(ctx context.Context, label string, uri string) error {
	c := b.db.CopySession()
	defer c.CloseSession()

	fmt.Printf("label:%v,uri:%v", label, uri)
	if uri == "" {
		_, err := c.RemoveAll(bson.M{"label": label})
		return err
	}
	_, err := c.RemoveAll(bson.M{"label": label, "$or": []bson.M{bson.M{"url": uri}, bson.M{"id": uri}}})
	return err
}

func (s *Service) PostBjrunImageAdd(ctx context.Context,
	args *struct {
		Data []struct {
			URI       string `json:"uri"`
			Attribute struct {
				Label string `json:"label"`
			} `json:"attribute"`
		} `json:"data"`
	}, env *authstub.Env) {

	var (
		ctex, xl     = ctxAndLog(ctx, env.W, env.Req)
		ret          GeneralResp
		imgLocker    sync.Mutex
		politLocker  sync.Mutex
		waiter       sync.WaitGroup
		uid          = env.UserInfo.Uid
		utype        = env.UserInfo.Utype
		failedImages struct {
			Failed []string `json:"failed"`
		}
		newImages []Image
	)
	xl.Infof("PostBjrunPoliticianAdd args:%v", args)
	if len(args.Data) == 0 || len(args.Data) > 20 {
		httputil.ReplyErr(env.W, ErrArgs.Code, ErrArgs.Error())
		return
	}

	waiter.Add(len(args.Data))
	for _, item := range args.Data {
		go func(uri, label string, ctx context.Context) {
			defer waiter.Done()
			xl := xlog.FromContextSafe(ctx)
			if strings.TrimSpace(uri) == "" {
				return
			}
			if strings.TrimSpace(label) == "" {
				imgLocker.Lock()
				failedImages.Failed = append(failedImages.Failed, uri+": empty name")
				imgLocker.Unlock()
				return
			}

			var ffReq _EvalImageReq
			ffReq.Data.URI = uri //feature 无pts则返回整张图的feature
			ffResp, err := s.iFeature.Eval(ctx, ffReq, _EvalEnv{Uid: uid, Utype: utype})
			if err != nil {
				xl.Infof("PostBjrunPoliticianAdd query facex detect error:%v", err)
				imgLocker.Lock()
				failedImages.Failed = append(failedImages.Failed, uri)
				imgLocker.Unlock()
				return
			}
			if len(ffResp) < 5 {
				xl.Infof("PostBjrunPoliticianAdd query facex detect error:%v", err)
				imgLocker.Lock()
				failedImages.Failed = append(failedImages.Failed, uri+":feature length smaller than 5")
				imgLocker.Unlock()
				return
			}

			if !isBinaryData(uri) {
				politLocker.Lock()
				newImages = append(newImages, Image{
					Label:   label,
					Url:     uri,
					Feature: ffResp,
				})
				politLocker.Unlock()
				return
			}
			MD5ID, err := Base64Data2MD5(uri)
			if err != nil {
				imgLocker.Lock()
				failedImages.Failed = append(failedImages.Failed, uri)
				imgLocker.Unlock()
				return
			}
			politLocker.Lock()
			newImages = append(newImages, Image{
				Label:   label,
				ID:      MD5ID,
				Feature: ffResp,
			})
			politLocker.Unlock()
		}(item.URI, item.Attribute.Label, spawnContext(ctex))
	}
	waiter.Wait()
	if len(failedImages.Failed) == len(args.Data) {
		httputil.ReplyErr(env.W, http.StatusInternalServerError, "all failed")
		return
	}
	if err := s._BjrunImageManager.AddImage(ctex, newImages...); err != nil {
		xl.Errorf("PostBjrunImageAdd error:%v", errors.Detail(err))
		httputil.ReplyErr(env.W, http.StatusInternalServerError, "add to image database error")
		return
	}
	ret.Result = failedImages
	ret.Message = "success"
	httputil.Reply(env.W, http.StatusOK, ret)
}

func (b *Service) GetBjrunImageLabels(ctx context.Context, env *authstub.Env) {
	var (
		ctex, xl = ctxAndLog(ctx, env.W, env.Req)
		ret      GeneralResp
		lbs      struct {
			Labels []string `json:"labels"`
		}
	)
	dbret, err := b._BjrunImageManager.Labels(ctex)
	if err != nil {
		xl.Errorf("GetBjrunPoliticianList query database error:%v", err)
		httputil.ReplyErr(env.W, http.StatusInternalServerError, "query database error")
		return
	}
	ret.Message = "success"
	lbs.Labels = dbret
	ret.Result = lbs
	httputil.Reply(env.W, http.StatusOK, ret)
}
func (b *Service) GetBjrunImageList_(ctx context.Context, args *struct {
	CmdArgs []string
}, env *authstub.Env) {
	var (
		ctex, xl = ctxAndLog(ctx, env.W, env.Req)
		ret      GeneralResp
		imgList  struct {
			Images []string `json:"images"`
		}
	)
	if len(args.CmdArgs) == 0 {
		httputil.ReplyErr(env.W, ErrArgs.Code, ErrArgs.Error())
		return
	}
	dbret, err := b._BjrunImageManager.LabelImages(ctex, args.CmdArgs[0])
	if err != nil {
		xl.Errorf("PostBjrunPoliticianImages error:%v", errors.Detail(err))
		httputil.ReplyErr(env.W, http.StatusInternalServerError, "query database error")
		return
	}
	ret.Message = "success"
	imgList.Images = dbret
	ret.Result = imgList
	httputil.Reply(env.W, http.StatusOK, ret)
}
func (s *Service) PostBjrunImageDel(ctx context.Context, args *struct {
	Data []struct {
		URI       string `json:"uri"`
		Attribute struct {
			Label string `json:"label"`
		} `json:"attribute"`
	} `json:"data"`
}, env *authstub.Env) {

	var (
		ctex, xl = ctxAndLog(ctx, env.W, env.Req)
		waiter   sync.WaitGroup
		err      error
	)

	xl.Infof("PostBjrunImageDel args:%v", args)
	if len(args.Data) == 0 || len(args.Data) > 20 {
		httputil.ReplyErr(env.W, ErrArgs.Code, ErrArgs.Error())
		return
	}

	waiter.Add(len(args.Data))
	for _, item := range args.Data {
		go func(uri, label string, ctx context.Context) {
			defer waiter.Done()
			if strings.TrimSpace(label) == "" {
				return
			}
			_err := s._BjrunImageManager.Delete(ctx, label, uri)
			if _err != nil {
				err = _err
			}
		}(item.URI, item.Attribute.Label, spawnContext(ctex))
	}
	waiter.Wait()
	if err != nil {
		xl.Errorf("PostBjrunImageDel query database error:%v", err)
		httputil.ReplyErr(env.W, http.StatusInternalServerError, "query database error")
		return
	}

	httputil.Reply(env.W, http.StatusOK, "success")
}

func (s *Service) PostBjrunImageSearch(ctx context.Context, args *BjrunImageSearchReq, env *authstub.Env) (ret *BjrunImageSearchResp, err error) {
	var (
		ctex, xl = ctxAndLog(ctx, env.W, env.Req)
		uid      = env.UserInfo.Uid
		utype    = env.UserInfo.Utype

		waiter        sync.WaitGroup
		statisticResp _EvalImageSearchResp
		imgSearchErr  error
		mResp         _EvalBjrunImageSearchResp
	)
	if strings.TrimSpace(args.Data.URI) == "" {
		return nil, ErrArgs
	}

	InFilters := func(elem string) bool {
		var isFilterElem bool
		if len(args.Params.Filters) == 0 {
			isFilterElem = true
		} else {
			for _, e := range args.Params.Filters {
				if elem == e {
					isFilterElem = true
					break
				}
			}
		}
		return isFilterElem
	}

	ret = &BjrunImageSearchResp{}
	images, features, err := s._BjrunImageManager.All(ctex)
	if err != nil {
		xl.Errorf("PostBjrunImageSearch query database error:%v", err)
		return nil, err
	}
	if len(images) == 0 {
		xl.Error("no image data in dynamic database")
	}

	var ffReq _EvalImageReq
	ffReq.Data.URI = args.Data.URI
	ffResp, err := s.iFeature.Eval(ctx, ffReq, _EvalEnv{Uid: uid, Utype: utype})
	if err != nil {
		xl.Errorf("PostBjrunImageSearch query facex detect error:%v", err)
		return
	}
	if len(ffResp) < 5 {
		return nil, errors.New("get image feature error")
	}
	if args.Params.Limit == 0 {
		args.Params.Limit = 5
	}

	waiter.Add(1)
	go func(ctx context.Context) {
		defer waiter.Done()

		xl := xlog.FromContextSafe(ctx)
		imgReq := _EvalImageSearchReq{
			Data: struct {
				URI string `json:"uri"`
			}{URI: "data:application/octet-stream;base64," +
				base64.StdEncoding.EncodeToString(ffResp)},
			Params: &struct {
				Limit *int `json:"limit,omitempty"`
			}{Limit: &args.Params.Limit},
		}
		statisticResp, imgSearchErr = s.iImageSearch.Eval(ctx, "bjrun", imgReq, _EvalEnv{Uid: uid, Utype: utype})
		if imgSearchErr != nil {
			xl.Errorf("call PostImageSearch_ error:%v", imgSearchErr)
		}
	}(spawnContext(ctex))

	var _err error
	if len(images) != 0 {
		slimit := len(images)
		mReq := _EvalBjrunImageSearchReq{
			Data: make([]struct {
				URI string `json:"uri"`
			}, 2),
		}
		mReq.Data[0].URI = features
		mReq.Data[1].URI = "data:application/octet-stream;base64," +
			base64.StdEncoding.EncodeToString(ffResp)
		mReq.Params = struct {
			Limit *int `json:"limit,omitempty"`
		}{Limit: &slimit}

		mResp, _err = s.iBjrunImageSearch.Eval(ctx, mReq, _EvalEnv{Uid: uid, Utype: utype})
		if _err != nil {
			xl.Errorf("PostBjrunImageSearch call iBjrunImageSearch.Eval error:%v", _err)
		}
	}

	waiter.Wait()
	if imgSearchErr != nil && _err != nil {
		return nil, errors.New("call image search failed")
	}

	//merge result
	{
		var mindex int
		var sindex int

		for len(ret.Result) < args.Params.Limit && mindex < len(mResp.Result) && sindex < len(statisticResp.Result) {
			detail := BjrunImageSearchResult{}
			if mResp.Result[mindex].Score > statisticResp.Result[sindex].Score {
				detail.Label = images[mResp.Result[mindex].Index].Label
				detail.URL = images[mResp.Result[mindex].Index].ID
				if images[mResp.Result[mindex].Index].ID == "" {
					detail.URL = images[mResp.Result[mindex].Index].Url
				}
				detail.Score = mResp.Result[mindex].Score
				mindex++
			} else {
				detail.Label = statisticResp.Result[sindex].Label
				detail.URL = statisticResp.Result[sindex].Class
				detail.Score = statisticResp.Result[sindex].Score
				sindex++
			}
			if InFilters(detail.Label) {
				ret.Result = append(ret.Result, detail)
			}
		}

		for len(ret.Result) < args.Params.Limit && mindex < len(mResp.Result) {
			detail := BjrunImageSearchResult{}
			detail.Label = images[mResp.Result[mindex].Index].Label
			detail.URL = images[mResp.Result[mindex].Index].ID
			if images[mResp.Result[mindex].Index].ID == "" {
				detail.URL = images[mResp.Result[mindex].Index].Url
			}
			detail.Score = mResp.Result[mindex].Score
			mindex++
			if InFilters(detail.Label) {
				ret.Result = append(ret.Result, detail)
			}
		}
		for len(ret.Result) < args.Params.Limit && sindex < len(statisticResp.Result) {
			detail := BjrunImageSearchResult{}
			detail.Label = statisticResp.Result[sindex].Label
			detail.URL = statisticResp.Result[sindex].Class
			detail.Score = statisticResp.Result[sindex].Score
			sindex++
			if InFilters(detail.Label) {
				ret.Result = append(ret.Result, detail)
			}
		}
	}

	ret.Message = "success"
	if len(ret.Result) == 0 {
		ret.Message = "no valid results"
	}
	return ret, nil
}

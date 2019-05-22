package utility

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/binary"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/qiniu/db/mgoutil.v3"
	"github.com/qiniu/errors"
	"github.com/qiniu/http/httputil.v1"
	"github.com/qiniu/xlog.v1"
	"labix.org/v2/mgo/bson"
	"qiniu.com/auth/authstub.v1"
)

type CelebritySearchReq BjrunPoliticianSearchReq

type iFaceSearchCelebrity interface {
	Eval(context.Context, _EvalFaceSearchReq, _EvalEnv) (_EvalFaceSearchResp, error)
}

type _FaceSearchCelebrity struct {
	host    string
	timeout time.Duration
}

func newFaceSearchCelebrity(host string, timeout time.Duration) iFaceSearchCelebrity {
	return _FaceSearchCelebrity{host: host, timeout: timeout}
}

func (fm _FaceSearchCelebrity) Eval(
	ctx context.Context, req _EvalFaceSearchReq, env _EvalEnv,
) (_EvalFaceSearchResp, error) {
	var (
		url    = fm.host + "/v1/eval/celebrity"
		client = newRPCClient(env, fm.timeout)

		resp _EvalFaceSearchResp
	)
	err := callRetry(ctx,
		func(ctx context.Context) error {
			return client.CallWithJson(ctx, &resp, "POST", url, &req)
		})
	return resp, err
}

//----------------------------------------------------------------------------//

const CelebFacexDtThreshold = 0.976

type CelebReq struct {
	Data struct {
		URI string `json:"uri"`
	} `json:"data"`
	Params struct {
		Filters []string `json:"filters,omitempty"`
	} `json:"params"`
}

type CelebCollections struct {
	Celebritys mgoutil.Collection `coll:"Celebritys"`
}

var celebcollections CelebCollections

func initCelebDB(cfg *mgoutil.Config) error {
	sess, err := mgoutil.Open(&celebcollections, cfg)
	if err != nil {
		xlog.Errorf("", "init celebrity database failed:%v", err)
		return err
	}
	sess.SetPoolLimit(DefaultCollSessionPoolLimit)
	return err
}

func (s *Service) PostCelebrityAdd(ctx context.Context,
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
		newPolit []Celebrity
	)
	xl.Infof("PostCelebrityAdd args:%v", args)
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
				xl.Errorf("PostCelebrityAdd query facex detect error:%v", err)
				imgLocker.Lock()
				failedImages.Failed = append(failedImages.Failed, uri)
				imgLocker.Unlock()
				return
			}
			if len(fdResp.Result.Detections) != 1 {
				xl.Errorf("PostCelebrityAdd %v face detected", len(fdResp.Result.Detections))
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
				xl.Errorf("PostCelebrityAdd query facex detect error:%v", err)
				imgLocker.Lock()
				failedImages.Failed = append(failedImages.Failed, uri)
				imgLocker.Unlock()
				return
			}

			if !isBinaryData(uri) {
				politLocker.Lock()
				newPolit = append(newPolit, Celebrity{
					Name:    name,
					Url:     uri,
					Feature: ffResp,
				})
				politLocker.Unlock()
				return
			}

			MD5ID, err := Base64Data2MD5(uri)
			if err != nil {
				xl.Errorf("PostCelebrityAdd query Base64Data2MD5 error:%v", err)
				imgLocker.Lock()
				failedImages.Failed = append(failedImages.Failed, uri)
				imgLocker.Unlock()
				return
			}
			politLocker.Lock()
			newPolit = append(newPolit, Celebrity{
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
	if err := s._CelebrityManager.AddCelebrity(ctex, newPolit...); err != nil {
		xl.Errorf("_CelebrityManager.AddCelebrity error:%v", errors.Detail(err))
		httputil.ReplyErr(env.W, http.StatusInternalServerError, "add to Celebrity database error")
		return
	}
	ret.Result = failedImages
	ret.Message = "success"
	httputil.Reply(env.W, http.StatusOK, ret)
}

func (b *Service) GetCelebrityList(ctx context.Context, env *authstub.Env) {
	var (
		ctex, xl = ctxAndLog(ctx, env.W, env.Req)
		ret      GeneralResp
		polit    struct {
			Celebritys []string `json:"celebritys"`
		}
	)
	dbret, err := b._CelebrityManager.Celebritys(ctex)
	if err != nil {
		xl.Errorf("GetCelebrityList query database error:%v", err)
		httputil.ReplyErr(env.W, http.StatusInternalServerError, "query database error")
		return
	}
	ret.Message = "success"
	polit.Celebritys = dbret
	ret.Result = polit
	httputil.Reply(env.W, http.StatusOK, ret)
}

func (s *Service) PostCelebrityImages(ctx context.Context, args *struct {
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
	dbret, err := s._CelebrityManager.CelebrityImages(ctex, args.Params.Name)
	if err != nil {
		xl.Errorf("PostCelebrityImages error:%v", errors.Detail(err))
		httputil.ReplyErr(env.W, http.StatusInternalServerError, "query database error")
		return
	}
	ret.Message = "success"
	imgList.Images = dbret
	ret.Result = imgList
	httputil.Reply(env.W, http.StatusOK, ret)
}

func (s *Service) PostCelebrityDel(ctx context.Context, args *struct {
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

	xl.Infof("PostCelebrityAdd args:%v", args)
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
			_err := s._CelebrityManager.Delete(ctx, name, uri)
			if _err != nil {
				err = _err
			}
		}(item.URI, item.Attribute.Name, spawnContext(ctex))
	}
	waiter.Wait()
	if err != nil {
		xl.Errorf("PostCelebrityDel query database error:%v", err)
		httputil.ReplyErr(env.W, http.StatusInternalServerError, "query database error")
		return
	}

	httputil.Reply(env.W, http.StatusOK, "success")
}

func (s *Service) PostCelebritySearch(ctx context.Context, args *CelebritySearchReq, env *authstub.Env) (ret *FaceSearchResp, err error) {
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
		xl.Errorf("PostCelebritySearch query facex detect error:%v", err)
		return nil, err
	}
	if len(fdResp.Result.Detections) == 0 {
		xl.Error("PostCelebritySearch no face detected")
		return nil, httputil.NewError(http.StatusNotAcceptable, "no face detected")
	}

	polits, features, err := s._CelebrityManager.All(ctex)
	if err != nil {
		xl.Errorf("PostCelebrity Search query database error:%v", err)
		return nil, err
	}
	if len(polits) == 0 {
		xl.Errorf("no valid celebrity data in dynamic database")
	}

	waiter.Add(len(fdResp.Result.Detections))

	for _, dt := range fdResp.Result.Detections {
		go func(det _EvalFaceDetection, ctx context.Context) {
			defer waiter.Done()

			if det.Score < CelebFacexDtThreshold {
				return
			}
			var _inWaiter sync.WaitGroup
			var ffReq _EvalFaceReq
			ffReq.Data.URI = args.Data.URI
			ffReq.Data.Attribute.Pts = det.Pts
			ffResp, _err := s.iFaceFeatureV2.Eval(ctx, ffReq, _EvalEnv{Uid: uid, Utype: utype})
			if _err != nil {
				xl.Errorf("PostCelebritySearch query facex detect error:%v", _err)
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

			var pReq = _EvalFaceSearchReq{
				Data: struct {
					URI string `json:"uri"`
				}{URI: "data:application/octet-stream;base64," +
					base64.StdEncoding.EncodeToString(ffResp)},
			}
			var pResp _EvalFaceSearchResp
			var _errPs error
			_inWaiter.Add(1)
			go func(ctx context.Context) { //statistic library
				defer _inWaiter.Done()
				pResp, _errPs = s.iFaceSearchCelebrity.Eval(ctx, pReq, _EvalEnv{Uid: uid, Utype: utype})
			}(spawnContext(ctx))
			_inWaiter.Wait()

			if _errPs != nil || _errFs != nil {
				xl.Errorf("PostCelebritySearch iFaceSearchCelebrity error:%v,iFacexSearch error:%v", _errPs, _errFs)
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
			if mResp.Result.Score > pResp.Result.Score {
				if mResp.Result.Score > s.Config.PoliticianThreshold[1] {
					detail.Value.Name = polits[mResp.Result.Index].Name
				}
				detail.Value.Score = mResp.Result.Score
			} else {
				if pResp.Result.Class != "" && pResp.Result.Score > s.Config.PoliticianThreshold[1] {
					detail.Value.Name = pResp.Result.Class
				}
				detail.Value.Score = pResp.Result.Score
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

//CelebrityManager
//-------------------------------------------------------------//

type _CelebrityManager interface {
	AddCelebrity(context.Context, ...Celebrity) error
	Celebritys(context.Context) ([]string, error)
	CelebrityImages(context.Context, ...string) ([]string, error)
	Delete(context.Context, string, string) error
	All(context.Context) ([]Celebrity, string, error)
}

type celebrityDB struct {
	db *mgoutil.Collection

	polits   []Celebrity
	features string
	*sync.Mutex
	readMutex *sync.Mutex
}

func NewCelebrityManager() (_CelebrityManager, error) {
	return &celebrityDB{
		db:        &celebcollections.Celebritys,
		Mutex:     new(sync.Mutex),
		readMutex: new(sync.Mutex),
	}, nil
}

type Celebrity struct {
	Name    string `bson:"name"`
	Url     string `bson:"url,omitempty"`
	ID      string `bson:"id,omitempty"`
	Feature []byte `bson:"feature"`
}

func (b celebrityDB) AddCelebrity(ctx context.Context, pts ...Celebrity) error {
	c := b.db.CopySession()
	defer c.CloseSession()
	var intermediate []interface{}
	for _, p := range pts {
		intermediate = append(intermediate, p)
	}
	return c.Insert(intermediate...)
}

func (b celebrityDB) Celebritys(ctx context.Context) ([]string, error) {
	c := b.db.CopySession()
	defer c.CloseSession()

	var (
		polits []string
	)
	err := c.Find(bson.M{}).Select(bson.M{"_id": 0, "url": 0, "id": 0, "feature": 0}).Distinct("name", &polits)

	return polits, err
}

func (b celebrityDB) CelebrityImages(ctx context.Context, polits ...string) ([]string, error) {
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

func (b celebrityDB) Delete(ctx context.Context, name string, uri string) error {
	c := b.db.CopySession()
	defer c.CloseSession()

	if uri == "" {
		_, err := c.RemoveAll(bson.M{"name": name})
		return err
	}
	_, err := c.RemoveAll(bson.M{"name": name, "$or": []bson.M{bson.M{"url": uri}, bson.M{"id": uri}}})
	return err
}

func (b *celebrityDB) All(context.Context) ([]Celebrity, string, error) {
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

		var polits []Celebrity
		err := c.Find(bson.M{}).Select(bson.M{"_id": 0}).All(&polits)
		if err != nil {
			return
			// return nil, err
		}

		features, _ := func() (string, error) {
			{
				if len(polits) == 0 {
					return "", errors.New("no valid celebrity data in database")
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
					// xl.Errorf("PostCelebritySearch feature size smaller than 5, politican:%v", item)
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

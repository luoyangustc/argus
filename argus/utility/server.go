package utility

import (
	"bytes"
	"context"
	"fmt"
	"image"
	"image/jpeg"
	"image/png"
	"net/http"
	"time"

	"github.com/nfnt/resize"
	"github.com/qiniu/db/mgoutil.v3"
	"github.com/qiniu/xlog.v1"
	"golang.org/x/image/bmp"
	"qbox.us/errors"

	"qiniu.com/argus/utility/evals"
)

type Config struct {
	UseMock                  bool            `json:"use_mock"`
	ServingHost              string          `json:"serving_host"`
	Timeout                  TimeLimitation  `json:"time_out"`
	TerrorThreshold          float32         `json:"terror_threshold"`
	FaceSimThreshold         float32         `json:"face_sim_threshold"`
	BjRTerrorThreshold       []float32       `json:"bjrun_terror_threshold"`
	PoliticianFeatureVersion string          `json:"politician_feature_version"`
	PoliticianUpdate         bool            `json:"politician_update"`
	PulpReviewThreshold      float32         `json:"pulp_review_threshold"`
	PulpFusionThreshold      []int           `json:"pulp_fusion_threshold"`
	PoliticianThreshold      []float32       `json:"politician_threshold"`
	BluedDetectThreshold     float32         `json:"blued_detect_threshold"`
	MongConf                 *mgoutil.Config `json:"mgo_config"`
}

type TimeLimitation struct {
	FaceDetect  time.Duration `json:"face_detect"`
	FaceCluster time.Duration `json:"face_cluster"`
	Terror      time.Duration `json:"terror"`
}

type ImageSize struct {
	Width  int `json:"width"`
	Height int `json:"height"`
}

type Service struct {
	Config

	iFaceDetect
	iFaceAge
	iFaceGender
	iFaceFeature
	iFaceFeatureV2
	iFaceCluster
	iFaceSearchCelebrity
	iFaceGroupSearch
	iFacexSearch
	iImageSearch
	iBjrunImageSearch
	iFeature
	iDetection
	iObjectClassify
	iScene
	iBluedD
	iBluedClassify
	iOcrText
	iOcrScene
	iOcrSariIdcard
	iOcrSariVat

	iFeatureV2 iFeature

	facexDetect      evals.IFaceDetect
	facexFeatureV1   evals.IFaceFeature
	facexFeatureV2   evals.IFaceFeature
	facexFeatureV3   evals.IFaceFeature
	imageInfo        evals.IImageInfo
	politician       evals.IPolitician
	pulp             evals.IPulp
	terrorPreDetect  evals.ITerrorPreDetect
	terrorDetect     evals.ITerrorDetect
	terrorPostDetect evals.ITerrorPostDetect
	terrorClassify   evals.ITerrorClassify

	_BjrunPoliticianManager
	_BjrunImageManager
	_CelebrityManager
}

func New(c Config) (*Service, error) {

	var (
		xl  = xlog.FromContextSafe(context.Background())
		srv = &Service{
			Config: c,
		}
	)

	err := initDB(c.MongConf)
	if err != nil {
		xl.Errorf("New service initDB error:%v", err)
	}
	errc := initCelebDB(c.MongConf)
	if errc != nil {
		xl.Errorf("New service initCelebDB error:%v", errc)
	}

	srv._BjrunImageManager, _ = NewBjrunImageManager()
	srv._BjrunPoliticianManager, _ = NewBjrunPoliticianManager()
	srv._CelebrityManager, _ = NewCelebrityManager()

	srv.iFaceDetect = newFaceDetect(c.ServingHost, time.Second*120)   // TODO
	srv.iFaceAge = newFaceAge(c.ServingHost, time.Second*120)         // TODO
	srv.iFaceGender = newFaceGender(c.ServingHost, time.Second*120)   // TODO
	srv.iFaceFeature = newFaceFeature(c.ServingHost, time.Second*400) // TODO
	srv.iFaceFeatureV2 = newFaceFeatureV2(c.ServingHost, time.Second*400)
	srv.iFaceCluster = newFaceCluster(c.ServingHost, time.Second*600) // TODO
	srv.iFaceSearchCelebrity = newFaceSearchCelebrity(c.ServingHost, time.Second*60)
	srv.iFaceGroupSearch = newFaceGroupSearch(c.ServingHost, time.Second*60) // TODO
	srv.iFeature = newFeature(c.ServingHost, time.Second*120)                // TODO
	srv.iFeature = newFeature(c.ServingHost, time.Second*120)                // TODO
	srv.iFeatureV2 = newFeatureV2(c.ServingHost, time.Second*120)            // TODO
	srv.iImageSearch = newImageSearch(c.ServingHost, time.Second*60)
	srv.iBjrunImageSearch = newBjrunImageSearch(c.ServingHost, time.Second*60)
	srv.iFacexSearch = newFacexSearch(c.ServingHost, time.Second*60) // TODO
	srv.iDetection = newObjectDetection(c.ServingHost, time.Second*10)
	srv.iObjectClassify = newObjectClassify(c.ServingHost, time.Second*10)
	srv.iScene = newScene(c.ServingHost, time.Second*10)
	srv.iBluedD = newBluedD(c.ServingHost, time.Second*60)
	srv.iBluedClassify = newBluedClassify(c.ServingHost, time.Second*60)

	srv.facexDetect = evals.NewFaceDetect(c.ServingHost, time.Second*60)
	srv.facexFeatureV1 = evals.NewFaceFeature(c.ServingHost, time.Second*60, "")
	srv.facexFeatureV2 = evals.NewFaceFeature(c.ServingHost, time.Second*60, "-v2")
	srv.facexFeatureV3 = evals.NewFaceFeature(c.ServingHost, time.Second*60, "-v3")
	srv.imageInfo = evals.NewImageInfo(c.ServingHost, time.Second*60)
	srv.pulp = evals.NewPulp(c.ServingHost, time.Second*60)
	srv.terrorPreDetect = evals.NewTerrorPreDetect(c.ServingHost, time.Second*60)
	srv.terrorDetect = evals.NewTerrorDetect(c.ServingHost, time.Second*60)
	srv.terrorPostDetect = evals.NewTerrorPostDetect(c.ServingHost, time.Second*60)
	srv.terrorClassify = evals.NewTerrorClassify(c.ServingHost, time.Second*60)
	srv.iOcrText = newOcrText(c.ServingHost, time.Second*60)
	srv.iOcrScene = newOcrScene(c.ServingHost, time.Second*60)
	srv.iOcrSariIdcard = newOcrSariIdcard(c.ServingHost, time.Second*60)
	srv.iOcrSariVat = newOcrSariVat(c.ServingHost, time.Second*60)

	srv.politician = evals.NewPolitician(c.ServingHost, time.Second*60, "")
	if srv.Config.PoliticianUpdate {
		srv.politician = evals.NewPolitician(c.ServingHost, time.Second*60, "-u")
	}

	return srv, nil
}

func ctxAndLog(
	ctx context.Context, w http.ResponseWriter, req *http.Request,
) (context.Context, *xlog.Logger) {

	xl, ok := xlog.FromContext(ctx)
	if !ok {
		xl = xlog.New(w, req)
		ctx = xlog.NewContext(ctx, xl)
	}
	return ctx, xl
}

func spawnContext(ctx context.Context) context.Context {
	return xlog.NewContext(ctx, xlog.FromContextSafe(ctx).Spawn())
}

func LocalResize(width, height int, img []byte) (iData []byte, err error) {

	var (
		i     image.Image
		iType string
	)

	if len(img) < 0 || width < 0 || height < 0 {
		return nil, errors.New("invalid params")
	}
	i, iType, err = image.Decode(bytes.NewReader(img))
	if err != nil {
		return
	}
	//如果size在范围之内则返回原始数据
	if i.Bounds().Dx() <= width && i.Bounds().Dy() <= height {
		return img, nil
	}

	dtBuf := bytes.NewBuffer(make([]byte, 0))
	ret := resize.Resize(uint(width), uint(height), i, resize.NearestNeighbor)
	switch iType {
	case "jpeg":
		err = jpeg.Encode(dtBuf, ret, nil)
		iData = dtBuf.Bytes()
	case "png":
		err = png.Encode(dtBuf, ret)
		iData = dtBuf.Bytes()
	case "bmp":
		err = bmp.Encode(dtBuf, ret)
		iData = dtBuf.Bytes()
	default:
		err = fmt.Errorf("unknown image type:%v", iType)
	}

	return
}

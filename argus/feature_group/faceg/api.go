package faceg

import (
	"context"
	"strings"
	"time"

	"github.com/qiniu/http/httputil.v1"

	FG "qiniu.com/argus/feature_group"
	"qiniu.com/argus/utility/evals"
)

const FACE_GROUP_WORKER_NODE_PREFIX = "/ava/argus/face_group/worker"

type FaceGFeatureAPIConfig struct {
	DetectHost       string       `json:"detect_host"`
	DetectVersion    string       `json:"detect_version"`
	FeatureHost      string       `json:"feature_host"`
	FeatureVersion   string       `json:"feature_version"`
	FeatureLength    int          `json:"feature_length"`    // feature length in float32
	FeatureByteOrder FG.ByteOrder `json:"feature_byteorder"` // 0: little-endian, 1: big-endian
	ReserveByteOrder bool         `json:"reserve_byteorder"` // if fasle, we'll always use little-endian.
	Threshold        float32      `json:"threshold"`
}

type FaceGFeatureAPI struct {
	FaceGFeatureAPIConfig
	evals.IFaceDetect
	evals.IFaceFeature
}

func NewFaceGFeatureAPI(config FaceGFeatureAPIConfig) FaceGFeatureAPI {
	featureV := config.FeatureVersion
	if len(featureV) > 0 && (!strings.HasPrefix(featureV, "-")) { // format: <cmd>-<version>
		featureV = "-" + featureV
	}

	return FaceGFeatureAPI{
		FaceGFeatureAPIConfig: config,
		IFaceDetect:           evals.NewFaceDetect(config.DetectHost, time.Second*30), // TODO: detect version not used yet.
		IFaceFeature: NewFaceFeature(
			evals.NewFaceFeature(config.FeatureHost, time.Second*30, featureV),
			featureV, config.FeatureByteOrder, config.ReserveByteOrder),
	}
}

/////////////////////////////////////////////////////////////////////////////

func NewFaceFeature(iFaceFeature evals.IFaceFeature,
	featureV string, byteorder FG.ByteOrder, reserve bool) *faceFeature {
	return &faceFeature{
		IFaceFeature:     iFaceFeature,
		FeatureV:         featureV,
		ByteOrder:        byteorder,
		ReserveByteOrder: reserve,
	}
}

type faceFeature struct {
	FeatureV         string
	IFaceFeature     evals.IFaceFeature
	ByteOrder        FG.ByteOrder
	ReserveByteOrder bool
}

func (f *faceFeature) Eval(ctx context.Context, req evals.FaceReq, uid, utype uint32) ([]byte, error) {
	t1 := time.Now()
	bs, err := f.IFaceFeature.Eval(ctx, req, uid, utype)
	if err != nil {
		_ClientTimeHistogram(f.FeatureV, httputil.DetectCode(err)).Observe(float64(time.Since(t1) / time.Second))
		return nil, err
	}
	_ClientTimeHistogram(f.FeatureV, 200).Observe(float64(time.Since(t1) / time.Second))

	if f.ReserveByteOrder != true && f.ByteOrder == FG.BigEndian {
		bs = FG.BigEndianToLittleEndian(bs)
	}

	return bs, nil
}

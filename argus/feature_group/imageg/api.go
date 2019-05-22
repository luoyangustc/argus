package imageg

import (
	"context"
	"encoding/binary"
	"math"
	"time"

	FG "qiniu.com/argus/feature_group"
	"qiniu.com/argus/utility/evals"
	httputil "qiniupkg.com/http/httputil.v2"
)

const IMAGE_GROUP_WORKER_NODE_PREFIX = "/ava/argus/image_group/worker"

type ImageGFeatureAPIConfig struct {
	FeatureHost      string       `json:"feature_host"`
	FeatureVersion   string       `json:"feature_version"`
	FeatureLength    int          `json:"feature_length"`    // feature length in float32
	FeatureByteOrder FG.ByteOrder `json:"feature_byteorder"` // 0: little-endian, 1: big-endian
	ReserveByteOrder bool         `json:"reserve_byteorder"` // if fasle, we'll always use little-endian.
	Threshold        float32      `json:"threshold"`
}

type ImageGFeatureAPI struct {
	ImageGFeatureAPIConfig
	evals.IImageFeature
}

func NewImageGFeatureAPI(config ImageGFeatureAPIConfig) ImageGFeatureAPI {
	return ImageGFeatureAPI{
		ImageGFeatureAPIConfig: config,
		IImageFeature:          NewImageFeature(evals.NewImageFeature(config.FeatureHost, time.Second*30, ""), config.FeatureVersion, config.FeatureByteOrder, config.ReserveByteOrder), // TODO: version not used yet.
	}
}

/////////////////////////////////////////////////////////////////////////////

func NewImageFeature(iImageFeature evals.IImageFeature, featureV string, byteorder FG.ByteOrder, reserve bool) *imageFeature {
	return &imageFeature{
		IImageFeature:    iImageFeature,
		FeatureV:         featureV,
		ByteOrder:        byteorder,
		ReserveByteOrder: reserve,
	}
}

type imageFeature struct {
	FeatureV string
	evals.IImageFeature
	ByteOrder        FG.ByteOrder
	ReserveByteOrder bool
}

func (f imageFeature) Eval(
	ctx context.Context, req evals.SimpleReq, uid, utype uint32,
) ([]byte, error) {
	t1 := time.Now()
	bs, err := f.IImageFeature.Eval(ctx, req, uid, utype)
	if err != nil || len(bs) == 0 {
		_ClientTimeHistogram(f.FeatureV, httputil.DetectCode(err)).Observe(float64(time.Since(t1) / time.Second))
		return bs, err
	}
	_ClientTimeHistogram(f.FeatureV, 200).Observe(float64(time.Since(t1) / time.Second))

	bo := f.ByteOrder
	if !f.ReserveByteOrder && f.ByteOrder == FG.BigEndian {
		bs = FG.BigEndianToLittleEndian(bs)
		bo = FG.LittleEndian
	}

	var (
		n  = len(bs) / 4
		fs = make([]float32, n)
	)
	if bo == FG.BigEndian {
		FG.ParseFloat32Buf(binary.BigEndian, bs, fs)
	} else {
		FG.ParseFloat32Buf(binary.LittleEndian, bs, fs)
	}

	var sum float64
	for _, f := range fs {
		sum += float64(f * f)
	}
	sum = math.Sqrt(sum)
	for i, f := range fs {
		fs[i] = f / float32(sum)
	}

	var bs2 = make([]byte, len(bs))
	if bo == FG.BigEndian {
		FG.FormatFloat32s(binary.BigEndian, fs, bs2)
	} else {
		FG.FormatFloat32s(binary.LittleEndian, fs, bs2)
	}
	return bs2, nil
}

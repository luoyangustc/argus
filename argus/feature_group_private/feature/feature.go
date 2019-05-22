package feature

import (
	"context"
	"encoding/binary"
	"fmt"
	"io/ioutil"
	"net/http"
	"time"

	"github.com/qiniu/http/httputil.v1"

	"github.com/pkg/errors"
	"github.com/qiniu/rpc.v1"
	xlog "github.com/qiniu/xlog.v1"
	"qiniu.com/argus/argus/facec/client"
	"qiniu.com/argus/feature_group_private"
	"qiniu.com/argus/feature_group_private/proto"
	"qiniu.com/argus/tuso/search"
)

var _ ImageFeature = new(imageFeature)
var _ FaceFeature = new(faceFeature)
var _ BaseFeature = new(baseFeature)

type imageFeature struct {
	host        string
	timeout     time.Duration
	featureSize int
}

func NewImageFeature(host string, timeout time.Duration, featureSize int) imageFeature {
	return imageFeature{
		host:        host,
		timeout:     timeout,
		featureSize: featureSize,
	}
}

func (img imageFeature) Image(ctx context.Context, uri proto.ImageURI) (fv proto.FeatureValue, err error) {
	var (
		cli = client.NewRPCClient(client.EvalEnv{Uid: 1, Utype: 0}, img.timeout)
		xl  = xlog.FromContextSafe(ctx)
	)

	if len(uri) == 0 {
		return nil, httputil.NewError(http.StatusBadRequest, "invalid image data")
	}

	req := map[string]interface{}{"data": map[string]string{"uri": string(uri)}}
	resp, err := cli.DoRequestWithJson(ctx, "POST", img.host+"/v1/eval/image-feature", req)
	if err != nil {
		return
	}
	defer resp.Body.Close()
	if resp.StatusCode/100 != 2 {
		return nil, errors.Wrap(rpc.ResponseError(resp), "PostEvalFeature")
	}
	fv, err = ioutil.ReadAll(resp.Body)
	if err != nil {
		xl.Errorf("Parse image feature failed: %s", err.Error())
		return nil, err
	}
	if len(fv) != img.featureSize {
		return nil, errors.New("imageFeature api return bad length")
	}
	search.NormFeatures(fv, img.featureSize)
	return
}

type faceFeature struct {
	host        string
	timeout     time.Duration
	featureSize int
}

func NewFaceFeature(host string, timeout time.Duration, featureSize int) faceFeature {
	return faceFeature{
		host:        host,
		timeout:     timeout,
		featureSize: featureSize,
	}
}

func (ff faceFeature) Face(ctx context.Context, uri proto.ImageURI, pts [][2]int) (fv proto.FeatureValue, err error) {
	var (
		cli = client.NewRPCClient(client.EvalEnv{Uid: 1, Utype: 0}, ff.timeout)
		xl  = xlog.FromContextSafe(ctx)
	)

	if len(uri) == 0 {
		return nil, httputil.NewError(http.StatusBadRequest, "invalid image data")
	}

	req := map[string]interface{}{"data": struct {
		URI       string `json:"uri"`
		Attribute struct {
			PTS [][2]int `json:"pts"`
		} `json:"attribute"`
	}{
		URI: string(uri),
		Attribute: struct {
			PTS [][2]int `json:"pts"`
		}{
			PTS: pts,
		},
	}}
	resp, err := cli.DoRequestWithJson(ctx, "POST", ff.host+"/v1/eval/facex-feature-v4", req)

	if err != nil {
		return
	}
	defer resp.Body.Close()
	if resp.StatusCode/100 != 2 {
		return nil, errors.Wrap(rpc.ResponseError(resp), "faceFeature.Face")
	}
	fv, err = ioutil.ReadAll(resp.Body)
	if err != nil {
		xl.Errorf("Parse face feature failed: %s", err.Error())
		return nil, err
	}
	if len(fv) != ff.featureSize {
		return nil, errors.New("faceFeature api return bad length")
	}
	fv = bigEndianToLittleEndian(fv)
	return
}

func (ff faceFeature) FaceBoxesQuality(ctx context.Context, uri proto.ImageURI) (
	boxes []proto.FaceDetectBox,
	err error) {
	var (
		cli   = client.NewRPCClient(client.EvalEnv{Uid: 1, Utype: 0}, ff.timeout)
		fdRet struct {
			Code    int    `json:"code"`
			Message string `json:"message"`
			Result  struct {
				Detections []struct {
					Index        int                                `json:"index"`
					Class        string                             `json:"class"`
					Score        float32                            `json:"score"`
					Pts          [][2]int                           `json:"pts"`
					Quality      proto.FaceQualityClass             `json:"quality"`
					Orientation  proto.FaceOrientation              `json:"orientation"`
					QualityScore map[proto.FaceQualityClass]float32 `json:"q_score"`
				} `json:"detections"`
			} `json:"result"`
		}
	)
	boxes = make([]proto.FaceDetectBox, 0)

	if len(uri) == 0 {
		return nil, httputil.NewError(http.StatusBadRequest, "invalid image data")
	}

	req := map[string]interface{}{"data": map[string]string{"uri": string(uri)}, "params": map[string]interface{}{"use_quality": 1}}
	err = cli.CallWithJson(ctx, &fdRet, "POST", ff.host+"/v1/eval/facex-detect", req)
	if err != nil {
		return
	}

	for _, detection := range fdRet.Result.Detections {
		boxes = append(boxes, proto.FaceDetectBox{
			BoundingBox: proto.BoundingBox{
				Pts:   detection.Pts,
				Score: proto.BoundingBoxScore(detection.Score),
			},
			Quality: proto.FaceQuality{
				Quality:      detection.Quality,
				Orientation:  detection.Orientation,
				QualityScore: detection.QualityScore,
			},
		})
	}
	return
}

func (ff faceFeature) FaceBoxes(ctx context.Context, uri proto.ImageURI) (
	boxes []proto.BoundingBox,
	err error) {
	var (
		cli   = client.NewRPCClient(client.EvalEnv{Uid: 1, Utype: 0}, ff.timeout)
		fdRet struct {
			Code    int    `json:"code"`
			Message string `json:"message"`
			Result  struct {
				Detections []struct {
					Index int      `json:"index"`
					Class string   `json:"class"`
					Score float32  `json:"score"`
					Pts   [][2]int `json:"pts"`
				} `json:"detections"`
			} `json:"result"`
		}
	)
	boxes = make([]proto.BoundingBox, 0)

	if len(uri) == 0 {
		return nil, httputil.NewError(http.StatusBadRequest, "invalid image data")
	}

	req := map[string]interface{}{"data": map[string]string{"uri": string(uri)}, "params": map[string]interface{}{"use_quality": 0}}
	err = cli.CallWithJson(ctx, &fdRet, "POST", ff.host+"/v1/eval/facex-detect", req)
	if err != nil {
		return
	}

	for _, detection := range fdRet.Result.Detections {
		boxes = append(boxes, proto.BoundingBox{
			Pts:   detection.Pts,
			Score: proto.BoundingBoxScore(detection.Score),
		})
	}
	return
}

func bigEndianToLittleEndian(a []byte) []byte {
	b := make([]byte, len(a))
	for i := 0; i < len(a); i += 4 {
		r := binary.BigEndian.Uint32(a[i : i+4])
		binary.LittleEndian.PutUint32(b[i:], r)
	}
	return b
}

type baseFeature struct {
	timeout time.Duration
	prefix  string
}

func NewBaseFeature(timeout time.Duration, prefix string) baseFeature {
	return baseFeature{
		timeout: timeout,
		prefix:  prefix,
	}
}

func (bf baseFeature) CreateGroup(ctx context.Context, address proto.NodeAddress, group proto.GroupName, cfg proto.GroupConfig) (err error) {
	var (
		cli = client.NewRPCClient(client.EvalEnv{Uid: 1, Utype: 0}, bf.timeout)
	)

	uri := fmt.Sprintf("http://%s/%s/groups/%s", address, bf.prefix, group)
	req := struct {
		Config          proto.GroupConfig `json:"config"`
		ClusterInternal bool              `json:"cluster_internal"`
	}{
		Config:          cfg,
		ClusterInternal: true,
	}
	return cli.CallWithJson(ctx, nil, "POST", uri, req)
}

func (bf baseFeature) RemoveGroup(ctx context.Context, address proto.NodeAddress, group proto.GroupName) (err error) {
	var (
		cli = client.NewRPCClient(client.EvalEnv{Uid: 1, Utype: 0}, bf.timeout)
	)
	req := struct {
		ClusterInternal bool `json:"cluster_internal"`
	}{
		ClusterInternal: true,
	}

	uri := fmt.Sprintf("http://%s/%s/groups/%s/remove", address, bf.prefix, group)
	return cli.CallWithJson(ctx, nil, "POST", uri, req)
}

func (bf baseFeature) AddFeature(ctx context.Context, address proto.NodeAddress, group proto.GroupName, features ...proto.Feature) (err error) {
	var (
		cli = client.NewRPCClient(client.EvalEnv{Uid: 1, Utype: 0}, bf.timeout)
	)

	uri := fmt.Sprintf("http://%s/%s/groups/%s/feature/add", address, bf.prefix, group)

	req := struct {
		Features        []proto.FeatureJson `json:"features"`
		ClusterInternal bool                `json:"cluster_internal"`
	}{
		ClusterInternal: true,
	}
	for _, feature := range features {
		req.Features = append(req.Features, feature.ToFeatureJson())
	}
	return cli.CallWithJson(ctx, nil, "POST", uri, req)
}

func (bf baseFeature) DeleteFeature(ctx context.Context, address proto.NodeAddress, group proto.GroupName, ids ...proto.FeatureID) (err error) {
	var (
		cli = client.NewRPCClient(client.EvalEnv{Uid: 1, Utype: 0}, bf.timeout)
	)

	uri := fmt.Sprintf("http://%s/%s/groups/%s/delete", address, bf.prefix, group)
	req := struct {
		IDs             []proto.FeatureID `json:"ids"`
		ClusterInternal bool              `json:"cluster_internal"`
	}{
		IDs:             ids,
		ClusterInternal: true,
	}
	return cli.CallWithJson(ctx, nil, "POST", uri, req)
}

func (bf baseFeature) UpdateFeature(ctx context.Context, address proto.NodeAddress, group proto.GroupName, features ...proto.Feature) (err error) {
	var (
		cli = client.NewRPCClient(client.EvalEnv{Uid: 1, Utype: 0}, bf.timeout)
	)

	uri := fmt.Sprintf("http://%s/%s/groups/%s/feature/update", address, bf.prefix, group)

	req := struct {
		Features        []proto.FeatureJson `json:"features"`
		ClusterInternal bool                `json:"cluster_internal"`
	}{
		ClusterInternal: true,
	}
	for _, feature := range features {
		req.Features = append(req.Features, feature.ToFeatureJson())
	}
	return cli.CallWithJson(ctx, nil, "POST", uri, req)
}

func (bf baseFeature) SearchFeature(ctx context.Context, address proto.NodeAddress, group proto.GroupName, threshold float32, limit int, features ...proto.FeatureValue) (ret [][]feature_group.FeatureSearchItem, err error) {
	var (
		cli    = client.NewRPCClient(client.EvalEnv{Uid: 1, Utype: 0}, bf.timeout)
		result [][]feature_group.FeatureSearchRespItem
	)

	req := struct {
		ClusterInternal bool                     `json:"cluster_internal"`
		Features        []proto.FeatureValueJson `json:"features"`
		Threshold       float32                  `json:"threshold"`
		Limit           int                      `json:"limit"`
	}{
		Threshold:       threshold,
		Limit:           limit,
		ClusterInternal: true,
	}
	for _, feature := range features {
		req.Features = append(req.Features, feature.ToFeatureValueJson())
	}

	uri := fmt.Sprintf("http://%s/%s/groups/%s/feature/search", address, bf.prefix, group)

	if err = cli.CallWithJson(ctx, &result, "POST", uri, req); err != nil {
		return
	}
	for r, row := range result {
		ret = append(ret, make([]feature_group.FeatureSearchItem, 0))
		for _, item := range row {
			ret[r] = append(ret[r], feature_group.FeatureSearchItem{
				ID:    item.Value.ID,
				Score: item.Score,
			})
		}
	}

	return
}

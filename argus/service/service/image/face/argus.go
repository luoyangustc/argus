package face

import (
	"context"
	"encoding/binary"
	"math"

	"github.com/go-kit/kit/endpoint"
	"golang.org/x/sync/errgroup"

	"github.com/qiniu/xlog.v1"

	"qiniu.com/argus/com/util"
	. "qiniu.com/argus/service/service"
	pimage "qiniu.com/argus/service/service/image"
	"qiniu.com/argus/utility/evals"
)

type FaceSimReq struct {
	Data []struct {
		IMG pimage.Image
		// URI string `json:"uri"`
	} `json:"data"`
}

type FaceSimResp struct {
	Code    int           `json:"code"`
	Message string        `json:"message"`
	Result  FaceSimResult `json:"result"`
}

type FaceSimResult struct {
	Faces      []FaceBox `json:"faces"`
	Similarity float32   `json:"similarity"`
	Same       bool      `json:"same"`
}

type FaceBox struct {
	Pts   [][2]int `json:"pts"`
	Score float32  `json:"score"`
}

type FaceDetectReq struct {
	Data struct {
		IMG pimage.Image
	} `json:"data"`
}

type FaceDetectResp struct {
	Code    int              `json:"code"`
	Message string           `json:"message"`
	Result  FaceDetectResult `json:"result"`
}

type FaceDetectResult struct {
	Detections []FaceDetectDetail `json:"detections"`
}

type FaceDetectDetail struct {
	BoundingBox FaceBox `json:"bounding_box"`
}

type FaceAttributeReq struct {
	Data struct {
		IMG       pimage.Image
		Attribute struct {
			Faces []struct {
				Orientation float32  `json:"orientation"`
				Pts         [][2]int `json:"pts,omitempty"`
				Landmarks   [][2]int `json:"landmarks,omitempty"`
			} `json:"faces"`
		} `json:"attribute"`
	} `json:"data"`
	Params struct {
		RoiScale float32 `json:"roi_scale"`
	} `json:"params"`
}

type FaceAttributeResult struct {
	Faces []struct {
		Orientation float32  `json:"orientation,omitempty"`
		Pts         [][2]int `json:"pts,omitempty"`
		Landmarks   [][2]int `json:"landmarks,omitempty"`
		Age         float32  `json:"age"`
		Gender      struct {
			Type  string  `json:"type"`
			Score float32 `json:"score"`
		} `json:"gender"`
	} `json:"faces"`
}

type FaceAttributeResp struct {
	Code    int                 `json:"code"`
	Message string              `json:"message"`
	Result  FaceAttributeResult `json:"result"`
}

func LargestFace(Detections []evals.FaceDetection) int {
	var (
		lgFaceArea  int
		lgFaceIndex int
	)
	for i, dt := range Detections {
		area := (dt.Pts[1][0] - dt.Pts[0][0]) * (dt.Pts[2][1] - dt.Pts[1][1])
		if area > lgFaceArea {
			lgFaceArea = area
			lgFaceIndex = i
		}
	}
	return lgFaceIndex
}

type FaceService interface {
	Sim(ctx context.Context, req FaceSimReq) (FaceSimResp, error)
	Detect(ctx context.Context, req FaceDetectReq) (FaceDetectResp, error)
	Attribute(ctx context.Context, req FaceAttributeReq) (FaceAttributeResp, error)
}

var _ FaceService = FaceEndpoints{}

type FaceEndpoints struct {
	SimEP       endpoint.Endpoint
	DetectEP    endpoint.Endpoint
	AttributeEP endpoint.Endpoint
}

func (ends FaceEndpoints) Sim(ctx context.Context, req FaceSimReq) (FaceSimResp, error) {
	response, err := ends.SimEP(ctx, req)
	if err != nil {
		return FaceSimResp{}, err
	}
	resp := response.(FaceSimResp)
	return resp, nil
}

func (ends FaceEndpoints) Detect(ctx context.Context, req FaceDetectReq) (FaceDetectResp, error) {
	response, err := ends.DetectEP(ctx, req)
	if err != nil {
		return FaceDetectResp{}, err
	}
	resp := response.(FaceDetectResp)
	return resp, nil
}

func (ends FaceEndpoints) Attribute(ctx context.Context, req FaceAttributeReq) (FaceAttributeResp, error) {
	response, err := ends.AttributeEP(ctx, req)
	if err != nil {
		return FaceAttributeResp{}, err
	}
	return response.(FaceAttributeResp), nil
}

var _ FaceService = faceService{}

type Config struct {
	SimThreshold float32 `json:"sim_threshold"`
}

var (
	DEFAULT = Config{
		SimThreshold: 0.45,
	}
)

type faceService struct {
	Config
	eFD EvalFaceDetectService
	eFF EvalFaceFeatureService
	eFA EvalFaceAttributeService
}

func NewFaceService(
	conf Config, fd EvalFaceDetectService, ff EvalFaceFeatureService, fa EvalFaceAttributeService,
) (FaceService, error) {
	// if conf.SimThreshold == 0.0 {
	// 	conf.SimThreshold = 0.5
	// }
	return faceService{Config: conf, eFD: fd, eFF: ff, eFA: fa}, nil
}

func (s faceService) Sim(
	ctx context.Context, args FaceSimReq,
) (ret FaceSimResp, err error) {

	var xl = xlog.FromContextSafe(ctx)

	if len(args.Data) != 2 {
		xl.Warnf("invalid number of uris. %d", len(args.Data))
		return FaceSimResp{}, ErrArgs("invalid number of uris")
	}

	faces := make([]FaceBox, len(args.Data))
	features := make([][]byte, len(args.Data))

	waiter := errgroup.Group{}
	for i := 0; i < len(args.Data); i++ {
		ctx2 := util.SpawnContext2(ctx, i)
		ord := i
		waiter.Go(func() error {
			ctx := ctx2
			img := args.Data[ord].IMG

			xl := xlog.FromContextSafe(ctx)
			var (
				fdReq FaceDetecReq
				ftReq FaceReq
			)
			fdReq.Data.IMG = img
			fdReq.Params.UseQuality = 0
			fdResp, err := s.eFD.EvalFaceDetect(ctx, fdReq)
			if err != nil {
				xl.Errorf("call facex-detect error: %v，", err)
				return ErrInternal(err.Error())
			}
			if fdResp.Code != 0 && fdResp.Code/100 != 2 {
				xl.Errorf("call facex-detect error:%v", fdResp)
				return ErrInternal(fdResp.Message)
			}
			if len(fdResp.Result.Detections) == 0 {
				xl.Errorf("call facex-detect error:%v", fdResp)
				return ErrFaceNotFound("found no face")
			}

			ftReq.Data.IMG = img
			LgFaceIndex := LargestFace(fdResp.Result.Detections)
			ftReq.Data.Attribute.Pts = fdResp.Result.Detections[LgFaceIndex].Pts
			ftResp, err := s.eFF.EvalFaceFeature(ctx, ftReq)
			if err != nil {
				xl.Errorf("call facex-feature error:%v", err)
				return ErrInternal(err.Error())
			}
			if len(ftResp) < 5 {
				xl.Errorf("feature length error:%v", ftResp)
				return ErrInternal("feature length error")
			}

			if fdResp.Result.Detections[LgFaceIndex].Score > 1.0 {
				fdResp.Result.Detections[LgFaceIndex].Score = 1.0
			}
			features[ord] = ftResp
			faces[ord] = FaceBox{
				Pts:   fdResp.Result.Detections[LgFaceIndex].Pts,
				Score: fdResp.Result.Detections[LgFaceIndex].Score,
			}

			return nil
		})
	}
	err = waiter.Wait()
	if err != nil {
		return FaceSimResp{}, err
	}

	// for _, ctx2 := range ctxs {
	// 	xl.Xput(xlog.FromContextSafe(ctx2).Xget())
	// }

	for _, ft := range features {
		if len(ft) == 0 {
			return FaceSimResp{}, ErrInternal("No valid features were obtained")
		}
	}

	var score float32
	for i, n := 0, len(features[0]); i < n; i += 4 {
		score += math.Float32frombits(binary.BigEndian.Uint32(features[0][i:i+4])) *
			math.Float32frombits(binary.BigEndian.Uint32(features[1][i:i+4]))
	}

	ret.Message = "success"
	ret.Result.Faces = []FaceBox{faces[0], faces[1]}
	ret.Result.Similarity = score
	if ret.Result.Similarity > 1.0 {
		ret.Result.Similarity = 1.0
	} else if ret.Result.Similarity < 0 {
		ret.Result.Similarity = 0
	}
	ret.Result.Same = ret.Result.Similarity >= s.SimThreshold

	return
}

func (s faceService) Detect(
	ctx context.Context, args FaceDetectReq,
) (ret FaceDetectResp, err error) {
	var (
		xl    = xlog.FromContextSafe(ctx)
		fdReq FaceDetecReq
	)

	fdReq.Data.IMG = args.Data.IMG
	fdReq.Params.UseQuality = 0
	fdResp, err := s.eFD.EvalFaceDetect(ctx, fdReq)
	if err != nil {
		xl.Errorf("call facex-detect error: %v，", err)
		return FaceDetectResp{}, ErrInternal(err.Error())
	}
	if fdResp.Code != 0 && fdResp.Code/100 != 2 {
		xl.Errorf("call facex-detect error:%v", fdResp)
		return FaceDetectResp{}, ErrInternal(fdResp.Message)
	}

	for _, dt := range fdResp.Result.Detections {
		ret.Result.Detections = append(
			ret.Result.Detections,
			FaceDetectDetail{BoundingBox: FaceBox{Pts: dt.Pts, Score: dt.Score}},
		)
	}

	if len(ret.Result.Detections) == 0 {
		ret.Message = "No valid face info detected"
	}

	return
}

func (s faceService) Attribute(
	ctx context.Context, args FaceAttributeReq,
) (FaceAttributeResp, error) {
	var (
		xl = xlog.FromContextSafe(ctx)
	)
	faResp, err := s.eFA.EvalFaceAttribute(ctx, args)
	if err != nil {
		xl.Errorf("call facex-attribute: %v", err)
		return FaceAttributeResp{}, ErrInternal(err.Error())
	}
	if faResp.Code != 0 && faResp.Code/100 != 2 {
		xl.Errorf("call facex-attribute error:%v", faResp)
		return FaceAttributeResp{}, ErrInternal(faResp.Message)
	}
	return faResp, nil
}

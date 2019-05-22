package faceg

import (
	"net/http"

	"github.com/qiniu/http/httputil.v1"

	xlog "github.com/qiniu/xlog.v1"
	"qiniu.com/argus/utility/evals"
)

const modeSingle = "SINGLE"
const modeLargest = "LARGEST"
const faceThresholdSize = 50 // 小于50的脸忽略

func checkFaceMode(req *FaceGroupAddReq) error {
	for _, item := range req.Data {
		switch item.Attribute.Mode {
		case "", modeSingle:
		case modeLargest:
		default:
			return httputil.NewError(http.StatusBadRequest, `invalid mode`)
		}
	}
	return nil
}

func checkFaceDetectionResp(xl *xlog.Logger, dResp evals.FaceDetectResp, mode string, uri string) (one evals.FaceDetection, err error) {
	if len(dResp.Result.Detections) == 0 {
		xl.Warnf("not one face: %v %v", uri, len(dResp.Result.Detections))
		return evals.FaceDetection{}, httputil.NewError(http.StatusBadRequest, `not face detected`)
	}
	switch mode {
	case "", modeSingle:
		if len(dResp.Result.Detections) != 1 {
			xl.Warnf("multiple face detected: %v %v", uri, len(dResp.Result.Detections))
			return evals.FaceDetection{}, httputil.NewError(http.StatusBadRequest, `multiple face detected`)
		}
		one = dResp.Result.Detections[0]
		if len(one.Pts) == 4 && one.Pts[0][0]+faceThresholdSize > one.Pts[2][0] && one.Pts[0][1]+faceThresholdSize > one.Pts[2][1] {
			xl.Warnf("face size < 50x50: %v %v", uri, one)
			return evals.FaceDetection{}, httputil.NewError(http.StatusBadRequest, `face size < 50x50`)
		}
		return one, nil
	case modeLargest:
		index := -1
		maxFace := -1
		for i, fb := range dResp.Result.Detections {
			if maxFace < (fb.Pts[1][0]-fb.Pts[0][0])*(fb.Pts[2][1]-fb.Pts[1][1]) {
				maxFace = (fb.Pts[1][0] - fb.Pts[0][0]) * (fb.Pts[2][1] - fb.Pts[1][1])
				index = i
			}
		}
		one = dResp.Result.Detections[index]
		if len(one.Pts) == 4 && one.Pts[0][0]+faceThresholdSize > one.Pts[2][0] && one.Pts[0][1]+faceThresholdSize > one.Pts[2][1] {
			xl.Warnf("face size < 50x50: %v %v", uri, one)
			return evals.FaceDetection{}, httputil.NewError(http.StatusBadRequest, `face size < 50x50`)
		}
		return one, nil
	}
	panic("should no reach " + mode)
}

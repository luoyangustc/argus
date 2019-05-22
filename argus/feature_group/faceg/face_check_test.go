package faceg

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/qiniu/xlog.v1"
	"qiniu.com/argus/utility/evals"
)

func Test_checkFaceDetectionResp(t *testing.T) {
	t.Run("SINGLE mode", func(t *testing.T) {
		t.Run("empty", func(t *testing.T) {
			a := assert.New(t)
			xl := xlog.NewWith("TEST")
			mode := ""
			uri := ""
			dResp := evals.FaceDetectResp{}
			_, err := checkFaceDetectionResp(xl, dResp, mode, uri)
			a.NotNil(err)
			if err != nil {
				a.Contains(err.Error(), "not face detected")
			}
		})
		t.Run("normal", func(t *testing.T) {
			a := assert.New(t)
			xl := xlog.NewWith("TEST")
			body := `{"code":200,"message":"tron_status_success","result":{"detections":[{"class":"face","index":1,"pts":[[328,2],[408,2],[408,96],[328,96]],"score":0.9990037083625793}]}}`
			dResp := evals.FaceDetectResp{}
			a.Nil(json.Unmarshal([]byte(body), &dResp))
			mode := ""
			uri := ""
			_, err := checkFaceDetectionResp(xl, dResp, mode, uri)
			a.Nil(err)
		})
		t.Run("small", func(t *testing.T) {
			a := assert.New(t)
			xl := xlog.NewWith("TEST")
			body := `{"code":200,"message":"tron_status_success","result":{"detections":[{"class":"face","index":1,"pts":[[74,1],[93,1],[93,20],[74,20]],"score":0.9776273965835571}]}}`
			dResp := evals.FaceDetectResp{}
			a.Nil(json.Unmarshal([]byte(body), &dResp))
			mode := ""
			uri := ""
			_, err := checkFaceDetectionResp(xl, dResp, mode, uri)
			a.NotNil(err)
			if err != nil {
				a.Contains(err.Error(), "face size < 50x50")
			}
		})
		t.Run("small, mode SINGLE", func(t *testing.T) {
			a := assert.New(t)
			xl := xlog.NewWith("TEST")
			body := `{"code":200,"message":"tron_status_success","result":{"detections":[{"class":"face","index":1,"pts":[[74,1],[93,1],[93,20],[74,20]],"score":0.9776273965835571}]}}`
			dResp := evals.FaceDetectResp{}
			a.Nil(json.Unmarshal([]byte(body), &dResp))
			mode := "SINGLE"
			uri := ""
			_, err := checkFaceDetectionResp(xl, dResp, mode, uri)
			a.NotNil(err)
			if err != nil {
				a.Contains(err.Error(), "face size < 50x50")
			}
		})
		t.Run("multiple", func(t *testing.T) {
			a := assert.New(t)
			xl := xlog.NewWith("TEST")
			body := `{"code":200,"message":"tron_status_success","result":{"detections":[{"class":"face","index":1,"pts":[[313,77],[358,77],[358,142],[313,142]],"score":0.9998549222946167},{"class":"face","index":1,"pts":[[100,69],[150,69],[150,136],[100,136]],"score":0.9998458623886108}]}}`
			dResp := evals.FaceDetectResp{}
			a.Nil(json.Unmarshal([]byte(body), &dResp))
			uri := ""
			mode := ""
			_, err := checkFaceDetectionResp(xl, dResp, mode, uri)
			a.NotNil(err)
			if err != nil {
				a.Contains(err.Error(), "multiple")
			}
		})
	})

	t.Run("LARGEST mode", func(t *testing.T) {
		mode := "LARGEST"
		t.Run("empty", func(t *testing.T) {
			a := assert.New(t)
			xl := xlog.NewWith("TEST")
			uri := ""
			dResp := evals.FaceDetectResp{}
			_, err := checkFaceDetectionResp(xl, dResp, mode, uri)
			a.NotNil(err)
			if err != nil {
				a.Contains(err.Error(), "not face detected")
			}
		})
		t.Run("normal", func(t *testing.T) {
			a := assert.New(t)
			xl := xlog.NewWith("TEST")
			body := `{"code":200,"message":"tron_status_success","result":{"detections":[{"class":"face","index":1,"pts":[[328,2],[408,2],[408,96],[328,96]],"score":0.9990037083625793}]}}`
			dResp := evals.FaceDetectResp{}
			a.Nil(json.Unmarshal([]byte(body), &dResp))
			uri := ""
			_, err := checkFaceDetectionResp(xl, dResp, mode, uri)
			a.Nil(err)
		})
		t.Run("small", func(t *testing.T) {
			a := assert.New(t)
			xl := xlog.NewWith("TEST")
			body := `{"code":200,"message":"tron_status_success","result":{"detections":[{"class":"face","index":1,"pts":[[74,1],[93,1],[93,20],[74,20]],"score":0.9776273965835571}]}}`
			dResp := evals.FaceDetectResp{}
			a.Nil(json.Unmarshal([]byte(body), &dResp))
			uri := ""
			_, err := checkFaceDetectionResp(xl, dResp, mode, uri)
			a.NotNil(err)
			if err != nil {
				a.Contains(err.Error(), "face size < 50x50")
			}
		})
		t.Run("small, mode SINGLE", func(t *testing.T) {
			a := assert.New(t)
			xl := xlog.NewWith("TEST")
			body := `{"code":200,"message":"tron_status_success","result":{"detections":[{"class":"face","index":1,"pts":[[74,1],[93,1],[93,20],[74,20]],"score":0.9776273965835571}]}}`
			dResp := evals.FaceDetectResp{}
			a.Nil(json.Unmarshal([]byte(body), &dResp))
			uri := ""
			_, err := checkFaceDetectionResp(xl, dResp, mode, uri)
			a.NotNil(err)
			if err != nil {
				a.Contains(err.Error(), "face size < 50x50")
			}
		})
		t.Run("multiple", func(t *testing.T) {
			a := assert.New(t)
			xl := xlog.NewWith("TEST")
			body := `{"code":200,"message":"tron_status_success","result":{"detections":[{"class":"face","index":1,"pts":[[313,77],[358,77],[358,142],[313,142]],"score":0.9998549222946167},{"class":"face","index":1,"pts":[[100,69],[150,69],[150,136],[100,136]],"score":0.9998458623886108}]}}`
			dResp := evals.FaceDetectResp{}
			a.Nil(json.Unmarshal([]byte(body), &dResp))
			uri := ""
			r, err := checkFaceDetectionResp(xl, dResp, mode, uri)
			a.Nil(err)
			a.Equal(r, dResp.Result.Detections[1])
		})
	})
}

func Test_checkFaceMode(t *testing.T) {
	req := &FaceGroupAddReq{
		Data: []FaceGroupAddData{
			FaceGroupAddData{
				Attribute: struct {
					ID   string          `json:"id"`
					Name string          `json:"name"`
					Mode string          `json:"mode"`
					Desc json.RawMessage `json:"desc,omitempty"`
				}{
					Mode: "SINGLE",
				},
			},
			FaceGroupAddData{
				Attribute: struct {
					ID   string          `json:"id"`
					Name string          `json:"name"`
					Mode string          `json:"mode"`
					Desc json.RawMessage `json:"desc,omitempty"`
				}{
					Mode: "LARGEST",
				},
			},
		},
	}
	err := checkFaceMode(req)
	assert.Nil(t, err)

	req = &FaceGroupAddReq{
		Data: []FaceGroupAddData{
			FaceGroupAddData{
				Attribute: struct {
					ID   string          `json:"id"`
					Name string          `json:"name"`
					Mode string          `json:"mode"`
					Desc json.RawMessage `json:"desc,omitempty"`
				}{
					Mode: "Test",
				},
			},
		},
	}
	err = checkFaceMode(req)
	assert.Equal(t, "invalid mode", err.Error())
}

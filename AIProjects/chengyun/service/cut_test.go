package service

import (
	"context"
	"encoding/json"
	"io/ioutil"
	"os"
	"path"
	"testing"
	"time"

	"github.com/qiniu/xlog.v1"
	"github.com/stretchr/testify/assert"
)

type zhatuMock struct {
	r []jsonLine
}

func (zt zhatuMock) Eval(
	ctx context.Context, req EvalZhatuReq,
) (resp EvalZhatuResp, err error) {
	uri := req.Data.URI
	for _, r := range zt.r {
		if r.Img == uri {
			return EvalZhatuResp{
				Code: 200,
				Result: struct {
					Detections []EvalZhatuDetection `json:"detections"`
				}{
					Detections: r.Resp,
				},
			}, nil
		}
	}
	panic("")
}

type jsonLine struct {
	Img   string               `json:"img"`
	Index int                  `json:"index"`
	Resp  []EvalZhatuDetection `json:"resp"`
}

func TestCutVideo(t *testing.T) {
	a := assert.New(t)
	ctx := context.Background()
	m := new(Manager)

	captureTime := time.Date(2018, 3, 1, 10, 20, 0, 0, time.Local).Add(time.Second * 600) // 20180301月 10:20 + 600s

	buf, err := ioutil.ReadFile("testdata/19.json")
	a.Nil(err)
	var r []jsonLine
	a.Nil(json.Unmarshal(buf, &r))

	m.Zhatu = &zhatuMock{r: r}

	images := make([]string, 0)
	for _, v := range r {
		images = append(images, v.Img)
	}

	video := checkVideoResult{
		video: &Video{
			Path:  "testdata/19.mp4",
			Start: captureTime.Add(time.Second * -600),
			End:   captureTime.Add(time.Second * -81),
		},
		images: images,
	}
	parts, err := m.searchVideo(ctx, captureTime, "testdata", 1, video)
	a.Nil(err)
	a.Equal(12, len(parts))
	a.Equal(parts[0].startIndex, 640)
}

func TestConsumeCap(t *testing.T) {
	xlog.SetOutputLevel(0)
	m := Manager{}
	m.Workspace = "/workspace/disk"
	m.FileServer = "http://100.100.57.179:8000"
	m.Zhatu = NewZhatu(EvalConfig{Host: "http://100.100.62.237:10203"})
	m.InitVideo()
	cap := Capture{
		ID:   "500",
		Time: "20180402134352",
	}

	dir := path.Join(m.Workspace, tmpPrefix, cap.ID)
	os.MkdirAll(dir, 0755)

	resource := path.Join(m.Workspace, resourcePrefix, cap.Time[:8], cap.Time[8:10], cap.ID)
	os.MkdirAll(resource, 0755)

	err := m.ConsumeIllegalCapture(context.Background(), &cap, dir, resource)
	if err == nil {
		os.RemoveAll(dir)
	}
}

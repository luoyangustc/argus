package monitor

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"time"

	"qiniu.com/argus/censor"
)

func MonitorCensor(ctx context.Context) {

	bs, _ := json.Marshal(censor.ImageRequest{
		Datas: []struct {
			DataID string `json:"data_id,omitempty"`
			URI    string `json:"uri"`
		}{{URI: Image1}},
	})

	buf := bytes.NewBuffer(bs)

	req, _ := http.NewRequest("POST", "http://ai-censor.qiniuapi.com/v1/censor/image/recognition", buf)
	req.ContentLength = 49220
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Qiniu yvkBAhylLK6HTrhU644UcFiVeFhRMR4geKGB1Prt:PaFVy6qlGimHeOTslzBryxI2-VI=")

	MonitorHTTP(ctx, req, "ai-censor.qiniuapi.com", "image/recognition", time.Second*5)
}

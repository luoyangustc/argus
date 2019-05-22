package monitor

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"time"
)

func MonitorPulp(ctx context.Context) {

	bs, _ := json.Marshal(struct {
		Data struct {
			URI string `json:"uri"`
		} `json:"data"`
	}{
		Data: struct {
			URI string `json:"uri"`
		}{URI: Image1},
	})

	buf := bytes.NewBuffer(bs)

	req, _ := http.NewRequest("POST", "http://argus.atlab.ai/v1/pulp", buf)
	req.ContentLength = 49192
	req.Header.Set("Authorization", "Qiniu yvkBAhylLK6HTrhU644UcFiVeFhRMR4geKGB1Prt:DULsqeUHqSUxA0MkVVujXMSK3NY=")
	req.Header.Set("Content-Type", "application/json")

	MonitorHTTP(ctx, req, "argus.atlab.ai", "pulp", time.Second*3)
}

package concerns

import (
	"context"
	"fmt"
	"time"

	"qbox.us/qconf/qconfapi"
	"qiniu.com/argus/argus/com/auth"
	ahttp "qiniu.com/argus/argus/com/http"
	"qiniupkg.com/api.v7/kodo"
	log "qiniupkg.com/x/log.v7"
)

type KodoClient struct {
	DomainApiHost string
	Qconf         *qconfapi.Config
	Kodo          *kodo.Config
}

func NewKodoClient(
	domainApiHost string,
	qconf *qconfapi.Config,
	kodo *kodo.Config,
) *KodoClient {
	return &KodoClient{
		DomainApiHost: domainApiHost,
		Qconf:         qconf,
		Kodo:          kodo,
	}
}

func (s *KodoClient) GetBucketInfo(uid uint32, bucket string) (
	string, string, []string) {
	ak, sk, err := auth.AkSk(qconfapi.New(s.Qconf), uid)
	if err != nil {
		log.Errorf("auth.AkSk err, %+v", err)
		return "", "", nil
	}

	cli := ahttp.NewQiniuAuthRPCClient(ak, sk, time.Second*10)
	var domains = []struct {
		Domain string `json:"domain"`
		Tbl    string `json:"tbl"`
		Global bool   `json:"global"`
	}{}
	err = cli.Call(context.Background(), &domains,
		"GET", fmt.Sprintf("%s/v7/domain/list?tbl=%s", s.DomainApiHost, bucket),
	)
	if err != nil {
		log.Errorf("Get domains err, %+v", err)
		return ak, sk, nil
	}

	if domains == nil || len(domains) == 0 {
		log.Error("Get domains err, empty")
		return ak, sk, nil
	}

	dms := []string{}
	for _, dm := range domains {
		dms = append(dms, dm.Domain)
	}

	return ak, sk, dms
}

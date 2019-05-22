package client

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"time"

	xlog "github.com/qiniu/xlog.v1"
	"qbox.us/qconf/qconfapi"
	"qiniu.com/argus/argus/com/auth"
	BUCKET "qiniu.com/argus/argus/com/bucket"
	ahttp "qiniu.com/argus/argus/com/http"
	"qiniu.com/argus/ccp/manager/proto"
	PKodo "qiniu.com/argus/ccp/manager/proto/kodo"
	"qiniupkg.com/api.v7/kodo"
	log "qiniupkg.com/x/log.v7"
)

type SaveBack interface {
	Save(ctx context.Context, rule *proto.Rule, keys []string) error
	GetInnerBucketInfo(ctx context.Context) (uid uint32, ak, sk, bucket, domain string)
	GetKodoInfo(ctx context.Context) *kodo.Config
}

type _SaveBack struct {
	InnerConfig
	DomainApiHost string
	Qconf         *qconfapi.Config
	Kodo          *kodo.Config
}

func NewSaveBack(innerCfg InnerConfig,
	domainApiHost string,
	qconf *qconfapi.Config,
	kodo *kodo.Config,
) SaveBack {
	return &_SaveBack{
		InnerConfig:   innerCfg,
		DomainApiHost: domainApiHost,
		Qconf:         qconf,
		Kodo:          kodo,
	}
}

func (s *_SaveBack) Save(ctx context.Context, rule *proto.Rule, keys []string) error {
	xl := xlog.FromContextSafe(ctx)

	innerSaver := s.InnerConfig.GetInnerSaver(ctx, "")
	inAk, inSk, inDms := s.GetBucketInfo(innerSaver.UID, innerSaver.Bucket)
	xl.Infof("Get InnerDomains %s", inDms)
	if len(inDms) <= 0 {
		err := errors.New("InnerDomains empty")
		xl.Error(err)
		return err
	}

	innerBucket := BUCKET.Bucket{
		Config: BUCKET.Config{
			Config: *s.Kodo,
			Bucket: innerSaver.Bucket,
			Domain: inDms[0],
		}.New(inAk, inSk, 0, innerSaver.Bucket, ""),
	}

	var userBucket *BUCKET.Bucket
	if rule.Saver.IsOn {

		// 必须同一个UID
		uAk, uSk, uDms := s.GetBucketInfo(rule.UID, rule.Saver.Bucket)
		xl.Infof("Get UserDomains %s", uDms)
		if len(uDms) <= 0 {
			err := errors.New("UserDomains empty")
			xl.Error(err)
			return err
		}

		prefix := ""
		if rule.Saver.Prefix != nil {
			prefix = *rule.Saver.Prefix
		}
		userBucket = &BUCKET.Bucket{
			Config: BUCKET.Config{
				Config: *s.Kodo,
				Bucket: rule.Saver.Bucket,
				Domain: uDms[0],
			}.New(uAk, uSk, 0, rule.Saver.Bucket, prefix),
		}

	} else {
		kodoSrc, _, err := PKodo.UnmarshalRule(ctx, rule)
		if err != nil {
			xl.Errorf("kodo.UnmarshalRule err, %+v", err)
			return err
		}

		if len(kodoSrc.Buckets) <= 0 {
			err := fmt.Errorf("kodoSrc.Buckets empty, %+v", kodoSrc)
			xl.Errorf("%+v", err)
			return err
		}

		// 必须同一个UID
		uAk, uSk, uDms := s.GetBucketInfo(rule.UID, kodoSrc.Buckets[0].Bucket)
		xl.Infof("Get UserDomains %s", uDms)
		if len(uDms) <= 0 {
			err := errors.New("UserDomains empty")
			xl.Error(err)
			return err
		}

		prefix := ""
		if kodoSrc.Buckets[0].Prefix != nil {
			prefix = *kodoSrc.Buckets[0].Prefix
		}
		userBucket = &BUCKET.Bucket{
			Config: BUCKET.Config{
				Config: *s.Kodo,
				Bucket: kodoSrc.Buckets[0].Bucket,
				Domain: uDms[0],
			}.New(uAk, uSk, 0, kodoSrc.Buckets[0].Bucket, prefix),
		}
	}

	for _, key := range keys {
		rder, err := innerBucket.ReadByDomain(ctx, key)
		if err != nil {
			xl.Errorf("read file err: %s, %v", key, err)
			return err
		}
		defer rder.Close()

		buf := bytes.NewBuffer(make([]byte, 0, 1024*1024*256))
		_, err = buf.ReadFrom(rder)
		if err != nil && err != io.EOF {
			xl.Errorf("Read Buf err: %s, %v", key, err)
			return err
		}

		saveKey, err := userBucket.Save(ctx, key, buf, int64(buf.Len()))
		if err != nil && err != io.EOF {
			xl.Errorf("Save Buf err: %s, %v", key, err)
			return err
		}

		xl.Infof("SaveKey: %s", saveKey)
	}

	return nil
}

//====

func (s *_SaveBack) GetBucketInfo(uid uint32, bucket string) (
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

func (s *_SaveBack) GetInnerBucketInfo(ctx context.Context) (uid uint32, ak, sk, bucket, domain string) {
	innerSaver := s.InnerConfig.GetInnerSaver(ctx, "")
	uid = innerSaver.UID
	bucket = innerSaver.Bucket
	var dms []string
	ak, sk, dms = s.GetBucketInfo(uid, bucket)
	if len(dms) > 0 {
		domain = dms[0]
	}
	return
}

func (s *_SaveBack) GetKodoInfo(ctx context.Context) *kodo.Config {
	return s.Kodo
}

//====

var _ SaveBack = MockSaveBack{}

type MockSaveBack struct {
}

func (msb MockSaveBack) Save(ctx context.Context, rule *proto.Rule, keys []string) error {
	return nil
}

func (msb MockSaveBack) GetInnerBucketInfo(ctx context.Context) (uid uint32, ak, sk, bucket, domain string) {
	return 0, "", "", "", ""
}

func (msb MockSaveBack) GetKodoInfo(ctx context.Context) *kodo.Config {
	return nil
}

package saver

import (
	"bytes"
	"compress/gzip"
	"context"
	"fmt"
	"time"

	"qbox.us/qconf/qconfapi"
	"qiniupkg.com/api.v7/kodo"

	xlog "github.com/qiniu/xlog.v1"
	BUCKET "qiniu.com/argus/argus/com/bucket"
	"qiniu.com/argus/ccp/manual/dao"
)

type IBucketSaver interface {
	SaveResult(context.Context, string, string, *bytes.Buffer, int) error
}

func NewBucketSaver(domainApiHost string,
	qconf *qconfapi.Config,
	kodo *kodo.Config,
	setDao *dao.ISetDAO,
) IBucketSaver {
	return _BucketSaver{
		kClient: &KodoClient{
			DomainApiHost: domainApiHost,
			Qconf:         qconf,
			Kodo:          kodo,
		},
		SetDao: *setDao,
	}
}

var _ IBucketSaver = _BucketSaver{}

type _BucketSaver struct {
	kClient *KodoClient

	SetDao dao.ISetDAO
}

func (bs _BucketSaver) SaveResult(ctx context.Context, id, mimetype string, buf *bytes.Buffer, lineno int) error {
	var (
		xl = xlog.FromContextSafe(ctx)
	)

	setInMgo, err := bs.SetDao.QueryByID(ctx, id)
	if err != nil {
		xl.Errorf("notify.SetDao.QueryByID error: %#v", err.Error())
		return err
	}

	ak, sk, domain := bs.kClient.GetBucketInfo(setInMgo.Saver.UID, setInMgo.Saver.Bucket)
	bucketCli := BUCKET.Bucket{
		Config: BUCKET.Config{
			Config: *bs.kClient.Kodo,
			Bucket: setInMgo.Saver.Bucket,
			Domain: domain,
		}.New(ak, sk, 0, setInMgo.Saver.Bucket, ""),
	}

	//人审结果->bucket
	resultFile := fmt.Sprintf("%s/%s%s_%s_%d_cap_result_%d", *setInMgo.Saver.Prefix, id, mimetype, time.Now().Format("20060102150405"), len(setInMgo.ResultFiles), lineno)
	buf2 := genZip(buf)
	_, err = bucketCli.Save(ctx, resultFile, buf2, int64(buf2.Len()))
	if err != nil {
		xl.Errorf("Save cap result err: %s, %#v", id, err.Error())
		return err
	}
	xl.Infof("save job %s ccp_manual result to file: %s", id, resultFile)

	setInMgo.ResultFiles = append(setInMgo.ResultFiles, resultFile)
	err = bs.SetDao.Update(ctx, setInMgo)
	if err != nil {
		xl.Errorf("notify.SetDao.Update error: %#v", err.Error())
		return err
	}

	return nil
}

func genZip(buf *bytes.Buffer) *bytes.Buffer {
	buf2 := bytes.NewBuffer(nil)
	w := gzip.NewWriter(buf2)
	defer w.Close()
	_, _ = w.Write(buf.Bytes())
	w.Flush()
	return buf2
}

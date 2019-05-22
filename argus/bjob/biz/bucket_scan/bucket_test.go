package bucket_scan

import (
	"testing"
)

func TestBucketList(t *testing.T) {

	// qCli := func() *qconfapi.Client {
	// 	var conf qconfapi.Config
	// 	_ = json.Unmarshal(
	// 		[]byte(``),
	// 		&conf,
	// 	)
	// 	return qconfapi.New(&conf)
	// }()

	// var rs records.Records
	// {
	// 	ak, sk, _ := auth.AkSk(qCli, 1381102897)
	// 	stg := bucket.Bucket{
	// 		Config: bucket.Config{Config: kodo.Config{}}.
	// 			New(ak, sk, 0, "argus-bcp", "/5b3a0db77eb0de0006b4f613/"),
	// 	}
	// 	rs = records.NewRecords(context.Background(),
	// 		records.NewFile(stg, time.Hour, 1000000),
	// 		1024*1024*64)
	// }

	// ak, sk, _ := auth.AkSk(qCli, 1369082408)
	// scan := bucket.Scanner{
	// 	Config: bucket.Config{Config: kodo.Config{}}.
	// 		New(ak, sk, 0, "lianaibiji", ""),
	// }
	// iter, _ := scan.Scan(context.Background(), 1000)
	// ctx := context.Background()

	// xl := xlog.FromContextSafe(ctx)

	// var count int64
	// for {
	// 	item, _, ok := iter.Next(ctx)
	// 	if !ok {
	// 		if err := iter.Error(); err != nil {
	// 			xlog.FromContextSafe(ctx).Warnf("Next Failed. %v", err)
	// 		}
	// 		t.Fatalf("xx")
	// 	}

	// 	itemURI := func() string {
	// 		return fmt.Sprintf("qiniu://z%d/%s/%s", 0, "lianaibiji", item.Key)
	// 	}()

	// 	count++
	// 	if count%100000 == 0 {
	// 		xl.Infof(">>>>>> %d", count)
	// 	}

	// 	has, _ := rs.HasKey(ctx, records.RecordKey(itemURI))
	// 	if !has {
	// 		xl.Infof("............ %s", itemURI)
	// 	}
	// }

}

func TestBucketList2(t *testing.T) {

	// var conf qconfapi.Config
	// _ = json.Unmarshal(
	// 	[]byte(``),
	// 	&conf,
	// )
	// var uriConfig URI.QiniuAdminHandlerConfig
	// _ = json.Unmarshal(
	// 	[]byte(``),
	// 	&uriConfig,
	// )

	// qCli := func() *qconfapi.Client { return qconfapi.New(&conf) }()

	// handler := URI.New(URI.WithAdminAkSkV2(uriConfig, nil))
	// _ = handler

	// ak, sk, _ := auth.AkSk(qCli, 1369082408)
	// scan := bucket.Scanner{
	// 	Config: bucket.Config{Config: kodo.Config{}}.
	// 		New(ak, sk, 0, "lianaibiji", ""),
	// }
	// iter, _ := scan.Scan(context.Background(), 1000)
	// ctx := context.Background()

	// xl := xlog.FromContextSafe(ctx)

	// var count, images, videos int64
	// for {
	// 	item, _, ok := iter.Next(ctx)
	// 	if !ok {
	// 		if err := iter.Error(); err != nil {
	// 			xlog.FromContextSafe(ctx).Warnf("Next Failed. %v", err)
	// 		}
	// 		t.Fatalf("xx")
	// 	}

	// 	itemURI := func() string {
	// 		return fmt.Sprintf("qiniu://%d@z%d/%s/%s", 1369082408, 0, "lianaibiji", item.Key)
	// 	}()
	// 	count++

	// 	// buf, err := func() ([]byte, error) {
	// 	// 	resp, err := handler.Get(ctx, URI.Request{URI: itemURI}, URI.WithRange(0, 260))
	// 	// 	if err != nil {
	// 	// 		return nil, err
	// 	// 	}
	// 	// 	defer resp.Body.Close()
	// 	// 	return ioutil.ReadAll(resp.Body)
	// 	// }()
	// 	// if err != nil {
	// 	// 	xl.Warnf("get failed. %s %v", itemURI, err)
	// 	// 	continue
	// 	// }

	// 	// if filetype.IsImage(buf) {
	// 	// 	images++
	// 	// } else if filetype.IsVideo(buf) {
	// 	// 	videos++
	// 	// }

	// 	if strings.HasPrefix(item.MimeType, "image/") {
	// 		images++
	// 	} else if strings.HasPrefix(item.MimeType, "video/") {
	// 		videos++
	// 	} else if strings.HasPrefix(item.MimeType, "audio/") {
	// 	} else {
	// 		xl.Infof("%s %s", itemURI, item.MimeType)
	// 	}

	// 	if count%100000 == 0 {
	// 		xl.Infof(">>>>>> %d %d %d", count, images, videos)
	// 	}

	// }

}

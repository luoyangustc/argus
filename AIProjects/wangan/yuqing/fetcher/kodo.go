package fetcher

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/lionsoul2014/ip2region/binding/golang/ip2region"

	"github.com/qiniu/rpc.v3"
	xlog "github.com/qiniu/xlog.v1"
	"qbox.us/api/kmqcli"
	tblmgr "qbox.us/api/tblmgr.v2"
	"qbox.us/api/uc.v3"
	"qiniu.com/argus/AIProjects/wangan/yuqing"
	"qiniu.com/argus/com/uri"
	"qiniu.com/argus/com/util"
	qboxmac "qiniu.com/auth/qboxmac.v1"
)

const (
	OP_INPUT         = "i"
	USER_CENTER_HOST = "http://uc.qbox.me"
)

type KodoConfig struct {
	kmqcli.Config `json:"kmq"`
	TblMgrProxy   string `json:"tblmgr_proxy"`
	UID           uint32 `json:"uid"`
	QueueName     string `json:"queue_name"`
	IPRegion      string `json:"ipregion_db"`
}

type Kodo interface {
	Run(context.Context)
	AddItbl(context.Context, int, []string) error
	AddBlackList(context.Context, uint32, []string) error
	Message() chan yuqing.Message
	Buckets() []yuqing.BucketEntry
	BlackList() []yuqing.BucketEntry
	GetMetrics() yuqing.Metrics

	SetMaxVideoSize(size int)
}

type kodo struct {
	KodoConfig
	yuqing.Metrics
	//Nets       []net.IPNet
	MsgChannel chan yuqing.Message

	partitions   []int32
	users        []int
	buckets      sync.Map // key:itbl
	black_list   sync.Map
	maxVideoSize int
	ipregion     *ip2region.Ip2Region
}

var _ Kodo = &kodo{}

func NewKodo(conf KodoConfig) (Kodo, error) {
	ipregion, err := ip2region.New(conf.IPRegion)
	if err != nil {
		return nil, errors.New("create ipregion err:" + err.Error())
	}

	return &kodo{
		KodoConfig:   conf,
		MsgChannel:   make(chan yuqing.Message, 100),
		maxVideoSize: 10 * 1024 * 1024,
		ipregion:     ipregion,
	}, nil
}

func (k *kodo) Run(ctx context.Context) {
	var (
		xl  = xlog.FromContextSafe(ctx)
		cli = kmqcli.New(&k.KodoConfig.Config)
	)

	code, partitions, err := cli.GetQueuePartitions(k.UID, k.QueueName, xl)
	if err != nil || code != 200 {
		xl.Info("fail to get partitions, err: ", err)
		return
	}
	if len(partitions) == 0 {
		xl.Errorf("kmq partitions is empty")
		return
	}
	k.partitions = partitions

	for _, partition := range k.partitions {
		go func(ctx context.Context, partition int32) {
			xl := xlog.FromContextSafe(ctx)
			client := kmqcli.New(&k.KodoConfig.Config)
			index := 0
			position := "@"
			for {
				code, messages, next, err := client.ConsumeMessagesByPartitonAdmin(k.UID, k.QueueName, position, 1000, partition, xl)
				if code == 612 || len(messages) == 0 {
					xl.Info("all message is consumed...")
					time.Sleep(1 * time.Second)
					break
				}
				if err != nil || code != 200 {
					xl.Info("fail to ConsumeMessagesAdmin, err: ", err)
					return
				}
				for _, m := range messages {
					var message yuqing.KmqMsg
					err := json.Unmarshal([]byte(m), &message)
					if err != nil {
						xl.Warnf("Json unmarshal err: %s, message: %s", err.Error(), m)
					}
					ip, port := ParseIpPortFromBase64(message.Object.IP)
					if message.Op == OP_INPUT {
						atomic.AddUint64(&k.Total, 1)

						//if strings.HasPrefix(message.Object.MimeType, "video/") && message.Object.FSize < 10*1024*1024 {
						if message.Object.MimeType != "video/MP2T" &&
							(message.Object.MimeType == "application/x-mpegurl" || strings.HasPrefix(message.Object.MimeType, "video/")) &&
							message.Object.FSize < int64(k.maxVideoSize) {
							//if message.Object.MimeType == "application/x-mpegurl" || strings.HasPrefix(message.Object.MimeType, "video/") {
							atomic.AddUint64(&k.TargetVideo, 1)
							info, err := k.ipregion.MemorySearch(ip.String())
							if err != nil {
								xl.Warnf("fail to query ipregion (%s), error: %s", ip.String(), err.Error())
								continue
							}
							if info.CityId == 995 || !uri.IsPublicIP(ip.String()) {
								atomic.AddUint64(&k.TargetRegion, 1)
								// target user
								var (
									value  interface{}
									bucket yuqing.BucketEntry
									ok     bool
									e0     error
									itbl   = strings.Split(message.Object.ID, ":")[0]
								)
								if _, ok = k.black_list.Load(itbl); ok {
									atomic.AddUint64(&k.Unavailable, 1)
									continue
								}
								value, ok = k.buckets.Load(itbl)
								if !ok {
									if len(k.users) > 0 {
										continue
									}
									itbl_num, _ := strconv.ParseUint(itbl, 36, 32)
									if bucket, e0 = k.getBucketByItbl(ctx, uint32(itbl_num)); e0 != nil {
										xl.Errorf("fail to get bucket info by itbl %d, err: %s", itbl_num, e0.Error())
										break
									}
									info, e1 := k.getBucketInfo(ctx, bucket.Uid, bucket.Name)
									if e1 != nil {
										xl.Errorf("fail to get bucket %s info through uc, uid %d, error: %s", bucket.Name, bucket.Uid, err.Error())
										break
									} else {
										if info.Private == 1 || info.Protected == 1 {
											xl.Warnf("cache private bucket %s, uid %d, itbl %s", bucket.Name, bucket.Uid, itbl)
											atomic.AddUint64(&k.Unavailable, 1)
											k.black_list.LoadOrStore(itbl, bucket)
											break
										}
									}

									if bucket.Domains, e0 = k.getDomainsByTbl(ctx, bucket.Name, bucket.Uid); e0 != nil {
										xl.Errorf("failed to get domain for bucket %s, error: %s", bucket.Name, e0.Error())
										break
									}
									k.buckets.Store(itbl, bucket)
								} else {
									bucket = value.(yuqing.BucketEntry)
								}

								atomic.AddUint64(&k.TargetUser, 1)
								ids := strings.Split(message.Object.ID, ":")
								var uri string
								if len(bucket.Domains) > 0 {
									uri = "http://" + bucket.Domains[0] + "/" + ids[1]
								}
								msg := yuqing.Message{
									Bucket:   bucket.Name,
									UID:      bucket.Uid,
									Itbl:     ids[0],
									Key:      ids[1],
									MD5:      ParseMD5FromBase64(message.Object.MD5),
									Fsize:    message.Object.FSize,
									IP:       ip.String(),
									Port:     port,
									URI:      uri,
									FH:       message.Object.FH,
									MimeType: message.Object.MimeType,
									PutTime:  time.Unix(message.Object.PutTime/10000000, message.Object.PutTime%10000000),
								}
								k.MsgChannel <- msg
							}
							if !uri.IsPublicIP(ip.String()) {
								atomic.AddUint64(&k.Inner, 1)
							}
						}

					}
				}
				index++
				position = next
			}
		}(util.SpawnContext2(ctx, int(partition)), partition)
	}
}

func (k *kodo) Message() chan yuqing.Message { return k.MsgChannel }
func (k *kodo) GetMetrics() yuqing.Metrics   { return k.Metrics }
func (k *kodo) SetMaxVideoSize(size int)     { k.maxVideoSize = size }

func (k *kodo) AddBlackList(ctx context.Context, uid uint32, bl_buckets []string) error {
	var (
		xl = xlog.FromContextSafe(ctx)
	)

	suInfo := fmt.Sprintf("%d/0", uid)
	transportMac := qboxmac.Mac{
		AccessKey: k.AccessKey,
		SecretKey: []byte(k.SecretKey),
	}
	transport := qboxmac.NewAdminTransport(&transportMac, suInfo, nil)
	tbl := tblmgr.New(USER_CENTER_HOST, transport)

	entries, err := tbl.Buckets(xl, "z0")
	if err != nil {
		xl.Errorf("fail to call get buckets, error: %s", err.Error())
		return err
	}
	for _, bucket := range entries {
		itbl := strconv.FormatUint(uint64(bucket.Itbl), 36)
		var found bool
		if len(bl_buckets) > 0 {
			for _, bl := range bl_buckets {
				if bl == bucket.Tbl {
					found = true
				}
			}
		} else {
			found = true
		}
		if found {
			k.black_list.Store(itbl, yuqing.BucketEntry{
				Itbl: itbl,
				Name: bucket.Tbl,
				Uid:  bucket.Uid,
				Zone: bucket.Zone,
			})
		}
	}
	return nil
}

func (k *kodo) AddItbl(ctx context.Context, uid int, buckets []string) error {
	var (
		xl = xlog.FromContextSafe(ctx)
	)

	suInfo := fmt.Sprintf("%d/0", uid)
	transportMac := qboxmac.Mac{
		AccessKey: k.AccessKey,
		SecretKey: []byte(k.SecretKey),
	}
	transport := qboxmac.NewAdminTransport(&transportMac, suInfo, nil)
	tbl := tblmgr.New(USER_CENTER_HOST, transport)

	entries, err := tbl.Buckets(xl, "z0")
	if err != nil {
		xl.Errorf("fail to call get buckets, error: %s", err.Error())
		return err
	}
	fmt.Println("Entries:", entries)
	for _, bucket := range entries {
		var (
			found bool
		)
		if len(buckets) == 0 {
			found = true
		} else {
			for _, name := range buckets {
				if name == bucket.Tbl {
					found = true
				}
			}
		}

		if found {
			_, err := k.getBucketInfo(ctx, bucket.Uid, bucket.Tbl)
			//if err != nil || info.Private == 1 || info.Protected == 1 {
			if err != nil {
				continue
			}

			itbl := strconv.FormatUint(uint64(bucket.Itbl), 36)
			bucketInfo := yuqing.BucketEntry{
				Itbl: itbl,
				Name: bucket.Tbl,
				Uid:  bucket.Uid,
				Zone: bucket.Zone,
			}
			if bucketInfo.Domains, err = k.getDomainsByTbl(ctx, bucket.Tbl, uint32(uid)); err != nil {
				xl.Errorf("failed to get domain for bucket %s, error: %s", bucket.Tbl, err.Error())
				return err
			}
			k.buckets.Store(itbl, bucketInfo)
		}

	}

	k.users = append(k.users, uid)
	return nil
}

func (k *kodo) Buckets() []yuqing.BucketEntry {
	var ret []yuqing.BucketEntry
	k.buckets.Range(func(key, value interface{}) bool {
		ret = append(ret, value.(yuqing.BucketEntry))
		return true
	})
	return ret
}
func (k *kodo) BlackList() []yuqing.BucketEntry {
	var ret []yuqing.BucketEntry
	k.black_list.Range(func(key, value interface{}) bool {
		ret = append(ret, value.(yuqing.BucketEntry))
		return true
	})
	return ret
}

func (k *kodo) getDomainsByTbl(ctx context.Context, tbl string, uid uint32) ([]string, error) {
	suInfo := fmt.Sprintf("%d/0", uid)
	transportMac := qboxmac.Mac{
		AccessKey: k.AccessKey,
		SecretKey: []byte(k.SecretKey),
	}
	transport := qboxmac.NewAdminTransport(&transportMac, suInfo, nil)
	var domains = []struct {
		Domain string `json:"domain"`
		Tbl    string `json:"tbl"`
		Global bool   `json:"global"`
	}{}
	cli := rpc.Client{Client: &http.Client{Transport: transport}}
	err := cli.CallWithJson(ctx, &domains, "GET", fmt.Sprintf("http://api.qiniu.com/v7/domain/list?tbl=%s", tbl), nil)
	if err != nil {
		return nil, err
	}
	var ret []string
	for _, domain := range domains {
		if !strings.HasPrefix(domain.Domain, ".") {
			ret = append(ret, domain.Domain)
		}
	}
	return ret, nil
}

func (k *kodo) getBucketByItbl(ctx context.Context, itbl uint32) (bucket yuqing.BucketEntry, err error) {
	cli := rpc.Client{Client: &http.Client{}}

	var ret tblmgr.BucketEntry
	err = cli.CallWithJson(ctx, &ret, "GET", k.TblMgrProxy+"/itblbucket/"+strconv.Itoa(int(itbl)), nil)
	if err != nil {
		return yuqing.BucketEntry{}, err
	}
	bucket = yuqing.BucketEntry{
		Itbl: strconv.FormatUint(uint64(ret.Itbl), 36),
		Name: ret.Tbl,
		Uid:  ret.Uid,
		Zone: ret.Zone,
	}
	return
}

func (k *kodo) getBucketInfo(ctx context.Context, uid uint32, bucket string) (uc.BucketInfo, error) {
	suInfo := fmt.Sprintf("%d/0", uid)
	transportMac := qboxmac.Mac{
		AccessKey: k.AccessKey,
		SecretKey: []byte(k.SecretKey),
	}
	transport := qboxmac.NewAdminTransport(&transportMac, suInfo, nil)
	cli := uc.New([]string{"http://uc.qbox.me"}, transport)
	return cli.BucketInfo(xlog.FromContextSafe(ctx), bucket)
}

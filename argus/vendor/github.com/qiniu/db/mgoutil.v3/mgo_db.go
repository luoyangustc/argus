package mgoutil

import (
	"reflect"
	"strings"
	"syscall"
	"time"

	"gopkg.in/mgo.v2"
	"qbox.us/lbsocketproxy"
	"qiniupkg.com/x/log.v7"
)

// ------------------------------------------------------------------------
func Dail(host string, mode string, syncTimeoutInS int64) (session *mgo.Session, err error) {

	session, err = mgo.Dial(host)
	if err != nil {
		log.Error("Connect MongoDB failed:", err, "- host:", host)
		return
	}

	if mode != "" {
		SetMode(session, mode, true)
	}
	if syncTimeoutInS != 0 {
		session.SetSyncTimeout(time.Duration(int64(time.Second) * syncTimeoutInS))
	}
	return
}

func DialWithProxy(host, mode string, syncTimeoutInS int64, proxyConf *lbsocketproxy.Config) (session *mgo.Session, err error) {
	return dialWithProxy(host, mode, syncTimeoutInS, false, proxyConf)
}

func dialWithProxy(host, mode string, syncTimeoutInS int64, direct bool, proxyConf *lbsocketproxy.Config) (session *mgo.Session, err error) {
	addrs := getMongoHosts(host)
	timeout := time.Second * 10
	info := mgo.DialInfo{
		Addrs:   addrs,
		Direct:  direct,
		Timeout: timeout,
	}
	if proxyConf != nil {
		proxy, err := lbsocketproxy.NewLbSocketProxy(proxyConf)
		if err != nil {
			return session, err
		}
		info.Dial = proxy.Dial
	}
	session, err = mgo.DialWithInfo(&info)
	if err != nil {
		return
	}
	session.SetSyncTimeout(1 * time.Minute)
	session.SetSocketTimeout(1 * time.Minute)

	if mode != "" {
		SetMode(session, mode, true)
	}
	if syncTimeoutInS != 0 {
		session.SetSyncTimeout(time.Duration(int64(time.Second) * syncTimeoutInS))
	}
	return
}

// ------------------------------------------------------------------------

type Safe struct {
	W        int    `json:"w"`
	WMode    string `json:"wmode"`
	WTimeout int    `json:"wtimeoutms"`
	FSync    bool   `json:"fsync"`
	J        bool   `json:"j"`
}

type Config struct {
	Host           string `json:"host"`
	DB             string `json:"db"`
	Mode           string `json:"mode"`
	SyncTimeoutInS int64  `json:"timeout"` // 以秒为单位
	Direct         bool   `json:"direct"`

	Safe    *Safe                 `json:"safe"`
	Proxies *lbsocketproxy.Config `json:"proxies"`
}

func Open(ret interface{}, cfg *Config) (session *mgo.Session, err error) {

	session, err = dialWithProxy(cfg.Host, cfg.Mode, cfg.SyncTimeoutInS, cfg.Direct, cfg.Proxies)
	if err != nil {
		return
	}

	EnsureSafe(session, cfg.Safe)

	if ret != nil {
		db := session.DB(cfg.DB)
		err = InitCollections(ret, db)
		if err != nil {
			session.Close()
			session = nil
		}
	}
	return
}

func InitCollections(ret interface{}, db *mgo.Database) (err error) {

	v := reflect.ValueOf(ret)
	if v.Kind() != reflect.Ptr {
		log.Error("mgoutil.Open: ret must be a pointer")
		return syscall.EINVAL
	}

	v = v.Elem()
	if v.Kind() != reflect.Struct {
		log.Error("mgoutil.Open: ret must be a struct pointer")
		return syscall.EINVAL
	}

	t := v.Type()
	for i := 0; i < t.NumField(); i++ {
		sf := t.Field(i)
		if sf.Tag == "" {
			continue
		}
		coll := sf.Tag.Get("coll")
		if coll == "" {
			continue
		}
		switch elem := v.Field(i).Addr().Interface().(type) {
		case *Collection:
			elem.Collection = db.C(coll)
		case **mgo.Collection:
			*elem = db.C(coll)
		default:
			log.Error("mgoutil.Open: coll must be *mgo.Collection or mgoutil.Collection")
			return syscall.EINVAL
		}
	}
	return
}

// ------------------------------------------------------------------------

// W 和 WMode 只在 replset 模式下生效，非replset不能配置，否则会出错
// WMode只在2.0版本以上才生效
func EnsureSafe(session *mgo.Session, safe *Safe) {
	if safe == nil {
		return
	}
	session.EnsureSafe(&mgo.Safe{
		W:        safe.W,
		WMode:    safe.WMode,
		WTimeout: safe.WTimeout,
		FSync:    safe.FSync,
		J:        safe.J,
	})
}

// [mongodb://][user:pass@]host1[:port1][,host2[:port2],...][/database][?options]
func getMongoHosts(raw string) []string {
	if strings.HasPrefix(raw, "mongodb://") {
		raw = raw[len("mongodb://"):]
	}
	if idx := strings.Index(raw, "@"); idx != -1 {
		raw = raw[idx+1:]
	}
	if idx := strings.Index(raw, "/"); idx != -1 {
		raw = raw[:idx]
	}
	if idx := strings.Index(raw, "?"); idx != -1 {
		raw = raw[:idx]
	}
	return strings.Split(raw, ",")
}

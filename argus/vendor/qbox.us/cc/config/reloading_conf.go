package config

import (
	"bytes"
	"crypto/md5"
	"encoding/base64"
	"fmt"
	"io/ioutil"
	"os"
	"os/signal"
	"sync"
	"syscall"
	"time"

	"github.com/qiniu/errors"
	"github.com/qiniu/rpc.v1"
	"github.com/qiniu/xlog.v1"
)

const (
	DefaultRemoteLock = "remote.lock"
)

func calcMd5sum(b []byte) []byte {
	h := md5.New()
	h.Write(b)
	return h.Sum(nil)
}

type ReloadingConfig struct {
	ConfName   string `json:"conf_name"`
	RemoteLock string `json:"remote_lock"`
	ReloadMs   int    `json:"reloading_ms"` // 如果ReloadMs为0,则不会自动reload
	RemoteURL  string `json:"remote_url"`

	md5sum []byte
	mutex  sync.Mutex
}

func (self *ReloadingConfig) reload(xl *xlog.Logger, onReload func(l rpc.Logger, data []byte) error) error {

	self.mutex.Lock()
	defer self.mutex.Unlock()

	errRemote := self.remoteReload(xl, onReload)
	if errRemote != nil {
		xl.Warn("remoteReload failed", errors.Detail(errRemote))
	} else {
		return nil
	}
	errLocal := self.localReload(xl, onReload)
	if errLocal != nil {
		err := errors.Info(errLocal, "localReload").Detail(errRemote)
		return err
	}
	return nil
}

func (self *ReloadingConfig) remoteReload(xl *xlog.Logger, onReload func(l rpc.Logger, data []byte) error) (err error) {

	if _, err1 := os.Stat(self.RemoteLock); !os.IsNotExist(err1) {
		if err1 != nil {
			err = errors.Info(err1, "os.Stat").Detail(err1)
			return err
		}
		err = errors.New("remote is locked")
		return
	}

	data, md5sum2, err := fetchRemote(xl, self.RemoteURL)
	if err != nil {
		err = errors.Info(err, "fetchRemote").Detail(err)
		return
	}

	if bytes.Equal(md5sum2, self.md5sum) {
		xl.Info("remoteReload: do nothing(md5sum is equal)")
	} else {
		confName := fmt.Sprintf("%v_%v", self.ConfName, base64.URLEncoding.EncodeToString(md5sum2))
		err = ioutil.WriteFile(confName, data, 0666)
		if err != nil {
			err = errors.Info(err, "ioutil.WriteFile")
			return
		}
		xl.Infof("remote file is changed, confName: %v, oldmd5: %v, newmd5: %v", confName, self.md5sum, md5sum2)

		err = onReload(xl, data)
		if err != nil {
			os.Remove(confName)
			err = errors.Info(err, "onReload", confName).Detail(err)
			return
		}

		err = os.Rename(confName, self.ConfName)
		if err != nil {
			os.Remove(confName)
			err = errors.Info(err, "os.Rename")
			return
		}
		self.md5sum = md5sum2
	}
	return
}

func (self *ReloadingConfig) localReload(xl *xlog.Logger, onReload func(l rpc.Logger, data []byte) error) (err error) {

	data, err := ioutil.ReadFile(self.ConfName)
	if err != nil {
		err = errors.Info(err, "ioutil.ReadFile").Detail(err)
		return
	}
	md5sum2 := calcMd5sum(data)

	if bytes.Equal(md5sum2, self.md5sum) {
		xl.Info("localReload: do nothing(md5sum is equal)")
	} else {
		xl.Infof("local file is changed, confName: %v, oldmd5: %v, newmd5: %v", self.ConfName, self.md5sum, md5sum2)

		err = onReload(xl, data)
		if err != nil {
			err = errors.Info(err, "onReload").Detail(err)
			return
		}
		self.md5sum = md5sum2
	}
	return
}

func fetchRemote(xl *xlog.Logger, URL string) (data, md5sum []byte, err error) {

	resp, err := rpc.DefaultClient.Get(xl, URL)
	if err != nil {
		return
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		err = rpc.ResponseError(resp)
		return
	}

	data, err = ioutil.ReadAll(resp.Body)
	if err != nil {
		err = errors.Info(err, "ioutil.ReadAll")
		return
	}
	md5sum = calcMd5sum(data)

	return
}

func StartReloading(cfg *ReloadingConfig, onReload func(l rpc.Logger, data []byte) error) (err error) {

	xl := xlog.NewWith("StartReloading")

	if cfg.RemoteLock == "" {
		cfg.RemoteLock = DefaultRemoteLock
	}

	err = cfg.reload(xl, onReload)
	if err != nil {
		xl.Error("cfg.reload:", errors.Detail(err))
		return
	}

	go func() {
		c := make(chan os.Signal, 1)
		signal.Notify(c, syscall.SIGHUP)
		for s := range c {
			xl := xlog.NewWith(fmt.Sprintf("Reloading/%v/%v", cfg.ConfName, s.String()))

			err := cfg.reload(xl, onReload)
			if err != nil {
				xl.Error("cfg.reload:", errors.Detail(err))
			}
		}
	}()

	if cfg.ReloadMs == 0 {
		return
	}
	go func() {
		dur := time.Duration(cfg.ReloadMs) * time.Millisecond
		for t := range time.Tick(dur) {

			xl := xlog.NewWith(fmt.Sprintf("Reloading/%v/%v", cfg.ConfName, t.Unix()))

			err := cfg.reload(xl, onReload)
			if err != nil {
				xl.Error("cfg.reload:", errors.Detail(err))
			}
		}
	}()
	return
}

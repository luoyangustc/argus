package db

import (
	"context"
	"testing"
	"time"

	"github.com/qiniu/db/mgoutil.v3"
)

func init() {
	Init(&mgoutil.Config{
		Host: "mongodb://127.0.0.1:27017",
		DB:   "argus_test",
	})
}

func TestDataVersionDao(t *testing.T) {
	d, _ := NewDataVersionDao()
	dao := d.(*dataVersionDao)

	uid, euid := "uid", time.Now().Format("20060102150405")
	_, err := dao.FindGroupVersion(context.Background(), uid, euid)
	if err == nil {
		t.Fatal(err)
	}

	status, version := STATUS_TODO, "123"

	t.Run("updateStatus", func(t *testing.T) {
		v, err := dao.UpdateStatus(context.Background(), uid, euid, status, STATUS_DOING)
		if err != nil || v == nil {
			t.Error("update error", err)
			return
		}

		g, err := dao.FindGroupVersion(context.Background(), uid, euid)
		if err != nil {
			t.Error("find group version error", err)
			return
		}

		if g.Status != STATUS_DOING {
			t.Error("status update failed", g.Status)
			return
		}

		dao.UpdateStatus(context.Background(), uid, euid, STATUS_DOING, status)
	})

	t.Run("UpdateStatusAndVersion", func(t *testing.T) {
		suc, err := dao.UpdateStatusAndVersion(
			context.Background(),
			uid, euid,
			"",
			version,
			status,
			STATUS_DOING)

		if err != nil || !suc {
			t.Error("update error", err, suc)
			return
		}

		g, err := dao.FindGroupVersion(context.Background(), uid, euid)
		if err != nil {
			t.Error("find group version error", err)
			return
		}

		if g.Status != STATUS_DOING {
			t.Error("status not updated", g.Status)
			return
		}

		if g.Version != version {
			t.Error("version not updated", g.Version)
		}
	})

}

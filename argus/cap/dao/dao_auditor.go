package dao

import (
	"context"
	"time"

	"gopkg.in/mgo.v2"
	"gopkg.in/mgo.v2/bson"

	"github.com/qiniu/db/mgoutil.v3"
	"github.com/qiniu/xlog.v1"
)

//
const (
	Valid   string = "valid"
	Invalid string = "invalid"
)

//
const (
	Online  string = "online"
	Offline string = "offline"
	Timeout string = "timeout"
)

// AuditorDAO
type IAuditorDAO interface {
	QueryByAID(ctx context.Context, auditorID string) (*AuditorInMgo, error)
	QueryAll(ctx context.Context) ([]AuditorInMgo, error)
	Insert(ctx context.Context, auditors ...AuditorInMgo) error
	Update(ctx context.Context, auditor *AuditorInMgo) error
	Remove(ctx context.Context, auditorID string) error
}

////////////////////////////////////////////////////////////////////////////////

var _ IAuditorDAO = _AuditorDAO{}

type _AuditorDAO struct {
	coll *mgoutil.Collection
}

// NewAuditorInMgo New
func NewAuditorInMgo(coll *mgoutil.Collection) IAuditorDAO {

	coll.EnsureIndex(mgo.Index{Key: []string{"auditor_id"}, Unique: true})

	return _AuditorDAO{coll: coll}
}

func (m _AuditorDAO) QueryByAID(ctx context.Context, auditorID string) (*AuditorInMgo, error) {
	coll := m.coll.CopySession()
	defer coll.CloseSession()

	var (
		xl      = xlog.FromContextSafe(ctx)
		auditor AuditorInMgo
	)

	err := coll.Find(bson.M{"auditor_id": auditorID}).One(&auditor)
	if err != nil {
		xl.Warnf("find auditor failed. %s %v", auditorID, err)
		return nil, err
	}

	return &auditor, nil
}

func (m _AuditorDAO) QueryAll(ctx context.Context) ([]AuditorInMgo, error) {
	coll := m.coll.CopySession()
	defer coll.CloseSession()

	var (
		xl       = xlog.FromContextSafe(ctx)
		auditors []AuditorInMgo
	)

	err := coll.Find(bson.M{}).All(&auditors)
	if err != nil {
		xl.Warnf("find auditors failed. %v", err)
	}

	return auditors, err
}

func (m _AuditorDAO) Insert(ctx context.Context, auditors ...AuditorInMgo) error {
	coll := m.coll.CopySession()
	defer coll.CloseSession()

	for _, auditor := range auditors {

		auditor.ID = bson.NewObjectId()
		auditor.CreatedAt = time.Now()
		auditor.UpdatedAt = auditor.CreatedAt
		auditor.Version = 1

		err := coll.Insert(auditor)
		if err != nil {
			return err
		}
	}

	return nil
}

func (m _AuditorDAO) Update(ctx context.Context, auditor *AuditorInMgo) error {
	coll := m.coll.CopySession()
	defer coll.CloseSession()

	auditor.UpdatedAt = time.Now()
	version := auditor.Version
	auditor.Version++

	return coll.Update(bson.M{"_id": auditor.ID, "version": version}, auditor)
}

func (m _AuditorDAO) Remove(ctx context.Context, auditorID string) error {
	coll := m.coll.CopySession()
	defer coll.CloseSession()

	return coll.Remove(bson.M{"auditor_id": auditorID})
}

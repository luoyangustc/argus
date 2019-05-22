package facec

import (
	"context"
	"errors"
	"time"

	"qiniu.com/argus/argus/facec/db"
)

type GroupMutex struct {
	dao db.DataVersionDao
}

func NewGroupMutex(dao db.DataVersionDao) GroupMutex {
	return GroupMutex{dao: dao}
}

type GroupProcedure struct {
	Uid        string
	Euid       string
	Version    string
	OldVersion string

	dao db.DataVersionDao
}

func (m *GroupMutex) NewProcedure(
	ctx context.Context,
	uid, euid string,
	timeout time.Duration,
) (*GroupProcedure, error) {

	var (
		deadline = time.Now().Add(timeout)
		sleep    = time.Millisecond
	)

	for {
		if time.Now().After(deadline) {
			return nil, errors.New("timeout")
		}

		v, err := m.dao.UpdateStatus(context.Background(), uid, euid, db.STATUS_TODO, db.STATUS_DOING)
		if err != nil {
			return nil, err
		}
		if v != nil {
			return &GroupProcedure{
				Uid:        uid,
				Euid:       euid,
				Version:    NewVersion(),
				OldVersion: v.Version,
				dao:        m.dao,
			}, nil
		}

		time.Sleep(sleep)
		sleep = sleep * 2
	}
}

func (p *GroupProcedure) Commit() error {
	_, err := p.dao.UpdateStatusAndVersion(
		context.Background(),
		p.Uid, p.Euid,
		p.OldVersion, p.Version,
		db.STATUS_DOING, db.STATUS_TODO,
	)
	return err
}

func (p *GroupProcedure) Revert() error {
	_, err := p.dao.UpdateStatusAndVersion(
		context.Background(),
		p.Uid, p.Euid,
		p.OldVersion, p.OldVersion,
		db.STATUS_DOING, db.STATUS_TODO,
	)
	return err
}

func (p *GroupProcedure) GetVersion() string    { return p.Version }
func (p *GroupProcedure) GetOldVersion() string { return p.OldVersion }

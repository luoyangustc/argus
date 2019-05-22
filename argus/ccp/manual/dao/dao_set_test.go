package dao

import (
	"context"
	"testing"

	mgoutil "github.com/qiniu/db/mgoutil.v3"
	"github.com/stretchr/testify/assert"
	ENUMS "qiniu.com/argus/cap/enums"
)

func TestSetDAO(t *testing.T) {
	var (
		ctx  = context.Background()
		conf = CcpCapMgoConfig{
			IdleTimeout:  5,
			MgoPoolLimit: 100,
			Mgo: mgoutil.Config{
				Host: "127.0.0.1:27017",
				DB:   "ccp_manual_test",
				Mode: "Strong",
			},
		}
	)

	setDao, err := NewSetInMgo(conf)
	assert.NoError(t, err)

	var (
		setInMgo = SetInMgo{
			SetID:      "testSetDaoDB",
			SourceType: "Kodo",
			Type:       string(ENUMS.REALTIME),
			NotifyURL:  "http://test.com",
		}
	)
	{
		err = setDao.Insert(ctx, &setInMgo)
		assert.NoError(t, err)

		v, err := setDao.QueryByID(ctx, setInMgo.SetID)
		assert.NoError(t, err)
		assert.Equal(t, v.SetID, setInMgo.SetID)
		assert.Equal(t, v.SourceType, setInMgo.SourceType)
		assert.Equal(t, string(ENUMS.REALTIME), v.Type)
		assert.Equal(t, v.NotifyURL, setInMgo.NotifyURL)

		err = setDao.Remove(ctx, setInMgo.SetID)
		assert.NoError(t, err)
	}
}

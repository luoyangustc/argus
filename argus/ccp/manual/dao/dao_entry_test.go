package dao

import (
	"context"
	"testing"

	mgoutil "github.com/qiniu/db/mgoutil.v3"
	"github.com/stretchr/testify/assert"
)

func TestEntryDAO(t *testing.T) {
	var (
		ctx     = context.Background()
		setConf = CcpCapMgoConfig{
			IdleTimeout:  5,
			MgoPoolLimit: 100,
			Mgo: mgoutil.Config{
				Host: "127.0.0.1:27017",
				DB:   "ccp_cap_set",
				Mode: "Strong",
			},
		}
		entryConf = CcpCapMgoConfig{
			IdleTimeout:  5,
			MgoPoolLimit: 100,
			Mgo: mgoutil.Config{
				Host: "127.0.0.1:27017",
				DB:   "ccp_cap_set",
				Mode: "Strong",
			},
		}
	)

	entryDao, err := NewEntryInMgo(entryConf)
	assert.NoError(t, err)
	setDao, err := NewSetInMgo(setConf)
	assert.NoError(t, err)

	var (
		setInMgo = SetInMgo{
			SetID:      "testSetDbId",
			SourceType: "Kodo",
			Type:       "Stream",
			NotifyURL:  "http://test.com",
		}
		entryInMgo = EntryInMgo{
			EntryID:  "testEntryDbId",
			SetID:    setInMgo.SetID,
			URIGet:   "http://testUriGet",
			MimeType: "IMAGE", // IMAGE / VIDEO / LIVE
		}
	)
	{
		err = setDao.Insert(ctx, &setInMgo)
		assert.NoError(t, err)
	}
	{
		err = entryDao.Insert(ctx, &entryInMgo)
		assert.NoError(t, err)

		v, err := entryDao.QueryByID(ctx, entryInMgo.SetID, entryInMgo.EntryID)
		assert.NoError(t, err)
		assert.NotNil(t, v)
		assert.Equal(t, v.SetID, setInMgo.SetID)
		assert.Equal(t, v.EntryID, entryInMgo.EntryID)
		assert.Equal(t, v.URIGet, entryInMgo.URIGet)
		assert.Equal(t, v.MimeType, entryInMgo.MimeType)

		paginator := NewPaginator(0, 2)
		entrys, err := entryDao.QueryBySetId(ctx, entryInMgo.SetID, paginator)
		assert.NoError(t, err)
		assert.NotEmpty(t, entrys)
		assert.Equal(t, entrys[0].SetID, setInMgo.SetID)
		assert.Equal(t, entrys[0].EntryID, entryInMgo.EntryID)
		assert.Equal(t, entrys[0].URIGet, entryInMgo.URIGet)
		assert.Equal(t, entrys[0].MimeType, entryInMgo.MimeType)

		entryInMgo.MimeType = "VIDEO"
		err = entryDao.Update(ctx, &entryInMgo)
		assert.NoError(t, err)
		v, err = entryDao.QueryByID(ctx, entryInMgo.SetID, entryInMgo.EntryID)
		assert.NoError(t, err)
		assert.Equal(t, v.MimeType, entryInMgo.MimeType)

		err = entryDao.Remove(ctx, entryInMgo.EntryID)
		assert.NoError(t, err)

		err = setDao.Remove(ctx, setInMgo.SetID)
		assert.NoError(t, err)
	}
}

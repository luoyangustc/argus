package cap

import (
	"context"
	"testing"

	mgoutil "github.com/qiniu/db/mgoutil.v3"
	"github.com/stretchr/testify/assert"
	ENUMS "qiniu.com/argus/cap/enums"
	"qiniu.com/argus/ccp/manual/client"
	"qiniu.com/argus/ccp/manual/dao"
	"qiniu.com/argus/ccp/manual/enums"
	"qiniu.com/argus/ccp/manual/model"
)

func TestService(t *testing.T) {
	var (
		ctx  = context.Background()
		conf = dao.CcpCapMgoConfig{
			IdleTimeout:  5,
			MgoPoolLimit: 100,
			Mgo: mgoutil.Config{
				Host: "127.0.0.1:27017",
				DB:   "ccp_manual_test",
				Mode: "Strong",
			},
		}
		entryConf = dao.CcpCapMgoConfig{
			IdleTimeout:  5,
			MgoPoolLimit: 100,
			Mgo: mgoutil.Config{
				Host: "127.0.0.1:27017",
				DB:   "ccp_manual_test",
				Mode: "Strong",
			},
		}
		batchConf = dao.CcpCapMgoConfig{
			IdleTimeout:  5,
			MgoPoolLimit: 100,
			Mgo: mgoutil.Config{
				Host: "127.0.0.1:27017",
				DB:   "ccp_cap_batch_entry",
				Mode: "Strong",
			},
		}
	)

	batchEntryDao, err := dao.NewBatchEntryInMgo(batchConf)
	assert.NoError(t, err)
	entryDao, err := dao.NewEntryInMgo(entryConf)
	assert.NoError(t, err)
	setDao, err := dao.NewSetInMgo(conf)
	assert.NoError(t, err)

	c := client.NewCAPClient(&model.CAPConfig{})
	handler := NewMaunalHandler(ctx, setDao, entryDao, batchEntryDao, c)
	iService := NewService(ctx, handler)

	var (
		setId    = "test_set_db_id"
		setModel = model.SetModel{
			SetId:      setId,
			UID:        12345,
			SourceType: "Kodo",
			Type:       enums.TYPE_BATCH,
			NotifyURL:  "http://test.com",
		}

		entryId    = setId + "entry_id"
		entryInMgo = dao.EntryInMgo{
			EntryID:  entryId,
			SetID:    setModel.SetId,
			URIGet:   "http://testUriGet",
			MimeType: "IMAGE", // IMAGE / VIDEO / LIVE
		}
	)

	{
		err = handler.InsertSet(ctx, &setModel)
		assert.NoError(t, err)

		err = entryDao.Insert(ctx, &entryInMgo)
		assert.NoError(t, err)
	}
	{
		resp, err := iService.GetSets(ctx, nil)
		assert.NoError(t, err)
		assert.NotNil(t, resp)
	}

	{
		req := struct {
			CmdArgs []string // setID
		}{CmdArgs: []string{setModel.SetId}}
		resp, err := iService.GetSets_(ctx, &req, nil)
		assert.NoError(t, err)
		assert.NotNil(t, resp)
		assert.Equal(t, resp.SetId, setModel.SetId)
		assert.Equal(t, resp.SourceType, setModel.SourceType)
		assert.Equal(t, string(ENUMS.BATCH), string(resp.Type))
		assert.Equal(t, resp.NotifyURL, setModel.NotifyURL)
	}

	{
		req := struct {
			CmdArgs []string // setID
			Offset  int      `json:"offset"`
			Limit   int      `json:"limit"`
		}{CmdArgs: []string{setModel.SetId}}
		resp, err := iService.GetSets_Entries(ctx, &req, nil)
		assert.NoError(t, err)
		assert.NotNil(t, resp)
	}

	{
		req := struct {
			CmdArgs []string // setID
		}{CmdArgs: []string{setModel.SetId, entryInMgo.EntryID}}
		resp, err := iService.GetSets_Entries_(ctx, &req, nil)
		assert.NoError(t, err)
		assert.NotNil(t, resp)
		assert.Equal(t, resp.EntryID, entryInMgo.EntryID)
	}

	//InsertEntries
	// uid := uint32(12345)
	// bucket := "test_bucket"
	// keys := []string{"key1", "key2"}
	// err = handler.InsertEntries(ctx, uid, bucket, setId, keys)
	// assert.NoError(t, err)

	//QueryEntries

	entries, err := handler.QueryEntries(ctx, setId, 0, 2)
	assert.NoError(t, err)
	assert.Len(t, entries, 1)
	assert.Equal(t, entryId, entries[0].EntryID)

	err = entryDao.Remove(ctx, entryInMgo.EntryID)
	assert.NoError(t, err)

	err = setDao.Remove(ctx, setModel.SetId)
	assert.NoError(t, err)
}

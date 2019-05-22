package cap

import (
	"context"

	xlog "github.com/qiniu/xlog.v1"

	ENUMS "qiniu.com/argus/cap/enums"
	"qiniu.com/argus/ccp/manual/client"
	"qiniu.com/argus/ccp/manual/dao"
	"qiniu.com/argus/ccp/manual/enums"
	"qiniu.com/argus/ccp/manual/model"
)

type IManualHandler interface {
	InsertSet(context.Context, *model.SetModel) error
	QuerySetById(context.Context, string) (*model.SetModel, error)
	QuerySets(context.Context) (*model.QuerySetsResp, error)

	//存量任务
	InsertEntries(context.Context, uint32, string, string, []string) error
	QueryEntries(context.Context, string, int, int) ([]model.EntryModel, error)

	//增量任务
	InsertEntry(context.Context, string, *model.EntryModel) error
	QueryEntry(context.Context, string, string) (*model.EntryModel, error)
}

var _ IManualHandler = _MaunalHanlder{}

type _MaunalHanlder struct {
	SetDao        dao.ISetDAO
	EntryDao      dao.IEntryDAO
	BatchEntryDao dao.IBatchEntryDAO
	Client        client.ICAPClient
	//CallbackUrl   string
}

func NewMaunalHandler(ctx context.Context,
	setDao dao.ISetDAO,
	entryDao dao.IEntryDAO,
	batchEntryDao dao.IBatchEntryDAO,
	client client.ICAPClient,
	//callbackUrl string,
) IManualHandler {
	return _MaunalHanlder{
		SetDao:        setDao,
		EntryDao:      entryDao,
		BatchEntryDao: batchEntryDao,
		Client:        client,
		//CallbackUrl:   callbackUrl,
	}
}

func (c _MaunalHanlder) InsertSet(ctx context.Context, setModel *model.SetModel) (err error) {
	var (
		xl = xlog.FromContextSafe(ctx)
	)

	//将ccp-manager发过来的大写形式做一个转换
	typeModel := setModel.Type
	switch typeModel {
	case enums.TYPE_STREAM:
		setModel.Type = ENUMS.REALTIME
	case enums.TYPE_BATCH:
		setModel.Type = ENUMS.BATCH
	}

	setInMgo := model.ToSetInMgo(setModel)
	xl.Infof("setInMgo: %#v", setInMgo)
	//插入数据库
	err = c.SetDao.Insert(ctx, setInMgo)
	if err != nil {
		xl.Warnf("insertSet to DB error: %#v", err.Error())
		return err
	}
	return nil
}

func (c _MaunalHanlder) QuerySetById(ctx context.Context, setId string) (*model.SetModel, error) {
	var (
		xl = xlog.FromContextSafe(ctx)
	)

	dbResp, err := c.SetDao.QueryByID(ctx, setId)
	if err != nil {
		xl.Warnf("querybyId error for setId: %#v, %#v", setId, err.Error())
		return nil, err
	}

	return model.FromSetInMgo(dbResp), nil
}

func (c _MaunalHanlder) QuerySets(ctx context.Context) (*model.QuerySetsResp, error) {
	var (
		xl   = xlog.FromContextSafe(ctx)
		resp = model.QuerySetsResp{}
	)
	dbResp, err := c.SetDao.QueryAll(ctx)
	if err != nil {
		xl.Errorf("set queryAll error: %#v", err.Error())
		return nil, err
	}
	for _, v := range dbResp {
		resp.Result = append(resp.Result, *model.FromSetInMgo(&v))
	}
	return &resp, nil
}

//=========================================================================================
//存量请求
func (c _MaunalHanlder) InsertEntries(ctx context.Context, uid uint32, bucket, setId string, keys []string) error {
	var (
		xl                  = xlog.FromContextSafe(ctx)
		ManmalDefaultPrefix = "CCP_MANUAL"
	)

	//得到对应的set信息
	setInMgo, err := c.SetDao.QueryByID(ctx, setId)
	if err != nil {
		xl.Errorf("c.SetDao.QueryByID error: %#v", err.Error())
		return err
	}

	//如果没有设置人审的默认存放bucket，就和机审的结果放到同一个bucket
	if setInMgo.Saver == nil {
		setInMgo.Saver = &struct {
			UID    uint32  `bson:"uid"`
			Bucket string  `bson:"bucket"`
			Prefix *string `bson:"prefix"`
		}{
			UID:    uid,
			Bucket: bucket,
			Prefix: &ManmalDefaultPrefix,
		}
		err := c.SetDao.Update(ctx, setInMgo)
		if err != nil {
			xl.Errorf("c.SetDao.update() error: %v", err.Error())
			return err
		}
	}

	setModel := model.FromSetInMgo(setInMgo)

	batchInMgo := &dao.BatchEntryInMgo{
		Uid:    uid,
		Bucket: bucket,
		Keys:   keys,
		SetId:  setId,
		Status: enums.BatchEntryJobStatusNew,
	}
	//创建图片的job
	if setInMgo.Image.IsOn {
		iJobId, err := c.Client.NewJob(ctx, ENUMS.MimeTypeImage, setModel)
		if err != nil {
			xl.Errorf("%s newImageJob error: %#v", setModel.SetId, err.Error())
			return err
		}
		batchInMgo.ImageSetID = iJobId
	}

	//创建视频的job
	if setInMgo.Video.IsOn {
		vJobId, err := c.Client.NewJob(ctx, ENUMS.MimeTypeVideo, setModel)
		if err != nil {
			xl.Errorf("%s  newVideoJob error: %#v", setModel.SetId, err.Error())
			return err
		}
		batchInMgo.VideoSetID = vJobId
	}

	//要人审的文件 -> batchEntry DB
	err = c.BatchEntryDao.BatchInsert(ctx, batchInMgo)
	if err != nil {
		xl.Warnf("insertEntries error: %#v", err.Error())
		return err
	}
	return nil
}

//增量请求
func (c _MaunalHanlder) InsertEntry(ctx context.Context, setId string, entryModel *model.EntryModel) error {
	// var (
	// 	xl = xlog.FromContextSafe(ctx)
	// )
	// entryInMgo := model.ToEntryInMgo(entryModel)
	// entryInMgo.SetID = setId
	// err := c.EntryDao.Insert(ctx, entryInMgo)
	// if err != nil {
	// 	xl.Warnf("insertSet to DB error: %#v", err.Error())
	// 	return err
	// }

	// setModel, err := c.SetDao.QueryByID(ctx, setId)
	// if err != nil {
	// 	xl.Warnf("queryByID %s error: %#v", setId, err.Error())
	// 	return err
	// }

	// err = c.Client.PushStreamTask(ctx, model.FromSetInMgo(setModel), entryModel)
	// if err != nil {
	// 	xl.Warnf("pushStreamTask error: %#v", err.Error())
	// 	return err
	// }

	return nil
}

func (c _MaunalHanlder) QueryEntries(ctx context.Context, id string, offset, limit int) ([]model.EntryModel, error) {
	var (
		xl   = xlog.FromContextSafe(ctx)
		resp = make([]model.EntryModel, 0)
	)
	p := dao.NewPaginator(offset, limit)
	dbResp, err := c.EntryDao.QueryBySetId(ctx, id, p)
	if err != nil {
		xl.Warnf("set queryAll error: %#v", err.Error())
		return nil, err
	}

	for _, v := range dbResp {
		resp = append(resp, *model.FromEntryInMgo(v))
	}

	return resp, nil
}

func (c _MaunalHanlder) QueryEntry(ctx context.Context, setId, entryId string) (*model.EntryModel, error) {
	var (
		xl = xlog.FromContextSafe(ctx)
	)
	dbResp, err := c.EntryDao.QueryByID(ctx, setId, entryId)
	if err != nil {
		xl.Warnf("set queryAll error: %#v", err.Error())
		return nil, err
	}

	return model.FromEntryInMgo(dbResp), nil
}

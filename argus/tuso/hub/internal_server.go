package hub

import (
	"context"

	"github.com/qiniu/http/httputil.v1"
	"gopkg.in/mgo.v2"

	"gopkg.in/mgo.v2/bson"

	"qiniu.com/argus/tuso/proto"
	authstub "qiniu.com/auth/authstub.v1"
)

var ErrFileNotFound = httputil.NewError(404, "file not found")

var _ proto.InternalApi = new(internalServer)

type internalServer struct {
	db *db
}

func (s *internalServer) GetHubInfo(ctx context.Context, req *proto.GetHubInfoReq, env *authstub.Env) (resp *proto.GetHubInfoResp, err error) {
	var hub dHub
	err = s.db.Hub.Find(bson.M{"hub_name": req.HubName}).One(&hub)
	if err != nil {
		if mgo.ErrNotFound == err {
			return nil, ErrHubNotFound
		}
		return nil, err
	}

	var hubMeta dHubMeta
	err = s.db.HubMeta.Find(bson.M{"hub_name": req.HubName, "version": req.Version}).One(&hubMeta)
	if err != nil {
		if mgo.ErrNotFound == err {
			return nil, ErrHubNotFound
		}
		return nil, err
	}
	resp = &proto.GetHubInfoResp{
		Uid:              hub.UID,
		Bucket:           hub.Bucket,
		Prefix:           hub.Prefix,
		FeatureFileIndex: hubMeta.FeatureFileIndex,
	}
	return resp, nil
}

func (s *internalServer) GetFilemetaInfo(ctx context.Context, req *proto.GetFileMetaInfoReq, env *authstub.Env) (resp *proto.GetFileMetaInfoResp, err error) {
	var fm dFileMeta
	err = s.db.FileMeta.Find(bson.M{"hub_name": req.HubName, "index": req.FeatureFileIndex, "offset": req.FeatureFileOffset}).One(&fm)
	if err != nil {
		if mgo.ErrNotFound == err {
			return nil, ErrFileNotFound
		}
		return nil, err
	}
	resp = &proto.GetFileMetaInfoResp{
		Key:        fm.Key,
		UpdateTime: fm.UpdateTime,
		Status:     string(fm.Status),
	}
	return resp, nil
}

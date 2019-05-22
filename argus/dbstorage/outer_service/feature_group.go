package outer_service

import (
	"context"
	"fmt"
	"time"

	"github.com/qiniu/http/httputil.v1"
	ahttp "qiniu.com/argus/argus/com/http"
	"qiniu.com/argus/dbstorage/proto"
)

var _ IFaceGroup = new(FaceGroup)

type Map map[string]interface{}

type FaceGroup struct {
	host      string
	timeout   time.Duration
	uid       uint32
	utype     uint32
	isPrivate bool
}

type FaceGroupResp struct {
	Ids    []string `json:"faces"`
	Errors []*struct {
		Code    int    `json:"code"`
		Message string `json:"message"`
	} `json:"errors"`
}

type FaceGroupPrivateResp struct {
	Id string `json:"id"`
}

func NewFaceGroup(host string, timeout time.Duration, isPrivate bool, uid, utype uint32) *FaceGroup {
	return &FaceGroup{
		host:      host,
		timeout:   timeout,
		uid:       uid,
		utype:     utype,
		isPrivate: isPrivate,
	}
}

func (fg *FaceGroup) Add(
	ctx context.Context, config proto.TaskConfig, groupName proto.GroupName, id proto.ImageId, uri proto.ImageURI, nameOrTag proto.ImageTag, desc proto.ImageDesc,
) (retId string, err error) {
	var (
		req  Map
		path string
	)

	if fg.isPrivate {
		ret := &FaceGroupPrivateResp{}
		req = Map{"image": Map{"id": id, "uri": uri, "tag": nameOrTag, "desc": desc}, "params": Map{"reject_bad_face": config.RejectBadFace}}
		path = fmt.Sprintf("/v1/face/groups/%s/add", groupName)
		err = fg.CallService(ctx, path, req, ret)
		if err == nil {
			retId = ret.Id
		}
	} else {
		ret := &FaceGroupResp{}
		req = Map{
			"data": []Map{
				Map{
					"uri":       uri,
					"attribute": Map{"id": id, "name": nameOrTag, "desc": desc, "mode": config.Mode, "reject_bad_face": config.RejectBadFace},
				},
			}}
		path = fmt.Sprintf("/v1/face/group/%s/add", groupName)
		err = fg.CallService(ctx, path, req, ret)

		//公有云的error可能在返回结果中
		if err == nil {
			if len(ret.Errors) > 0 && ret.Errors[0] != nil {
				err = httputil.NewError(ret.Errors[0].Code, ret.Errors[0].Message)
			} else if len(ret.Ids) > 0 {
				retId = ret.Ids[0]
			}
		}
	}

	return
}

func (fg *FaceGroup) CreateGroup(ctx context.Context, groupName string) error {
	var (
		req  Map
		path string
	)
	if fg.isPrivate {
		req = Map{"config": map[string]int{"capacity": 100000000}}
		path = "/v1/face/groups/" + groupName
	} else {
		path = "/v1/face/group/" + groupName + "/new"
	}

	err := fg.CallService(ctx, path, req, nil)
	return err
}

func (fg *FaceGroup) CallService(ctx context.Context, path string, req, ret interface{}) error {
	var (
		client = ahttp.NewQiniuStubRPCClient(fg.uid, fg.utype, fg.timeout)
	)
	return callRetry(ctx,
		func(ctx context.Context) error {
			return client.CallWithJson(ctx, ret, "POST", fg.host+path, req)
		})
}

func callRetry(ctx context.Context, f func(context.Context) error) error {
	return ahttp.CallWithRetry(ctx, []int{530}, []func(context.Context) error{f, f})
}

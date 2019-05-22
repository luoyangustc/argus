package facec

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestService_PostImagesFacegather(t *testing.T) {
	cleanDB(t)
	ctx := getMockCtx(t)
	ctx.Exec(`
auth qinutest |authstub -uid 1 -utype 1|
post http://argus.ataraxia.ai.local/v1/face/cluster/gather
auth qinutest
json '{
  "euid": "FsSpWs",
  "items": [
    "http://oohbr1ly0.bkt.clouddn.com/test/xxxxnotexists.jpeg",
    "http://q.hi-hi.cn/black.png",
    "http://q.hi-hi.cn/baidu.png",
    "http://oayqr8eyk.qnssl.com/lena.jpg",
    "http://oayqr8eyk.qnssl.com/Oscars.jpg"
  ]
}'
ret 200
echo $(resp.body)
json '{
  "message": "success",
  "modelv": "1",
  "result": {
    "fail": [
      {
        "code": 40,
        "item": "http://oohbr1ly0.bkt.clouddn.com/test/xxxxnotexists.jpeg",
        "message": "download failed"
      }
    ]
  }
}'
`)
}

func TestService_GetImagesFacegroups(t *testing.T) {
	assert := assert.New(t)
	ctx := getMockCtx(t)
	// TODO: 清空，然后插入数据库记录，检查是否返回结果是否正确
	ctx.Exec(`
auth qinutest |authstub -uid 1 -utype 1|
get http://argus.ataraxia.ai.local/v1/face/cluster?euid=FsSpWs
auth qinutest
ret 200
echo $(resp.body)
json '{
	"message": $(message)
}'
equal $(message) 'success'
`)
	assert.Equal(ctx.GetVar("resp.body.message").Data, "success")
	assert.Equal(ctx.GetVar("message").Data, "success")
}

func TestService_PostImagesFaceGroupsAdjust(t *testing.T) {
	ctx := getMockCtx(t)
	ctx.Exec(`
auth qinutest |authstub -uid 1 -utype 1|
post http://argus.ataraxia.ai.local/v1/face/cluster/adjust
auth qinutest
json '{
  "euid": "FsSpWs",
  "from_group_id": 10001,
  "to_group_id": 10002,
  "items": [
    "kodo://z0/my-bucket/Audrey_Hepburn1.jpg",
    "kodo://z0/my-bucket/Audrey_Hepburn2.jpg",
    "..."
  ]
}'
ret 400
`)
}

func TestService_PostImagesFaceGroupsMerge(t *testing.T) {
	ctx := getMockCtx(t)
	ctx.Exec(`
auth qinutest |authstub -uid 1 -utype 1|
post http://argus.ataraxia.ai.local/v1/face/cluster/merge
auth qinutest
json '{
  "euid": "FsSpWs",
  "to_group_id": 10001,
  "groups": [
    10001,
    10002
  ]
}'
ret 400
`)
}

func TestService_GetImagesFacegroups_(t *testing.T) {
	ctx := getMockCtx(t)
	ctx.Exec(`
auth qinutest |authstub -uid 1 -utype 1|
get http://argus.ataraxia.ai.local/v1/face/cluster/10001?euid=xxxx
auth qinutest
# 找不到group
ret 400 
echo $(resp.body)
equal $(resp.body.error) 'group not exists'
`)
}

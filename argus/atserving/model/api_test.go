package model

import (
	"context"
	"encoding/json"
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestSTRING(t *testing.T) {
	assert.Equal(t, "xxx", fmt.Sprintf("%#v", STRING("xxx")))
}

func TestUnmarshalTaskRequest(t *testing.T) {

	var ctx = context.Background()
	{
		bs, _ := json.Marshal(EvalRequest{Data: Resource{URI: "xx"}})
		v, err := UnmarshalTaskRequest(ctx, bs)
		assert.NoError(t, err)
		assert.NotNil(t, v)
		assert.Equal(t, STRING("xx"), v.(EvalRequest).Data.URI)
	}
	{
		bs, _ := json.Marshal(GroupEvalRequest{Data: []Resource{{URI: "xx"}}})
		v, err := UnmarshalTaskRequest(ctx, bs)
		assert.NoError(t, err)
		assert.NotNil(t, v)
		assert.Equal(t, STRING("xx"), v.(GroupEvalRequest).Data[0].URI)
	}
	{
		bs, _ := json.Marshal(
			struct {
				Data string `json:"data"`
			}{Data: "xx"},
		)
		_, err := UnmarshalTaskRequest(ctx, bs)
		assert.Equal(t, ErrParseTaskRequest, err)
	}
	{
		bs, _ := json.Marshal(
			struct {
				URI string `json:"uri"`
			}{URI: "xx"},
		)
		_, err := UnmarshalTaskRequest(ctx, bs)
		assert.Equal(t, ErrParseTaskRequest, err)
	}

}

func TestUnmarshalBatchTaskRequest(t *testing.T) {

	var ctx = context.Background()
	{
		bs, _ := json.Marshal(
			[]interface{}{
				EvalRequest{Data: Resource{URI: "xx1"}},
				EvalRequest{Data: Resource{URI: "xx2"}},
				GroupEvalRequest{Data: []Resource{{URI: "yy1"}, {URI: "yy2"}}},
				GroupEvalRequest{Data: []Resource{{URI: "zz1"}, {URI: "zz2"}}},
			})
		v, err := UnmarshalBatchTaskRequest(ctx, bs)
		assert.NoError(t, err)
		assert.NotNil(t, v)
		assert.Equal(t, 4, len(v))
		assert.Equal(t, STRING("xx1"), v[0].(EvalRequest).Data.URI)
		assert.Equal(t, STRING("xx2"), v[1].(EvalRequest).Data.URI)
		assert.Equal(t, STRING("yy1"), v[2].(GroupEvalRequest).Data[0].URI)
		assert.Equal(t, STRING("yy2"), v[2].(GroupEvalRequest).Data[1].URI)
		assert.Equal(t, STRING("zz1"), v[3].(GroupEvalRequest).Data[0].URI)
		assert.Equal(t, STRING("zz2"), v[3].(GroupEvalRequest).Data[1].URI)
	}
	{
		bs, _ := json.Marshal(
			[]interface{}{
				struct {
					Data string `json:"data"`
				}{Data: "xx"},
			},
		)
		_, err := UnmarshalBatchTaskRequest(ctx, bs)
		assert.Equal(t, ErrParseTaskRequest, err)
	}
	{
		bs, _ := json.Marshal(
			[]interface{}{
				struct {
					URI string `json:"uri"`
				}{URI: "xx"},
			},
		)
		_, err := UnmarshalBatchTaskRequest(ctx, bs)
		assert.Equal(t, ErrParseTaskRequest, err)
	}

}

package util

import (
	"encoding/json"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"gopkg.in/mgo.v2/bson"
)

func TestJSON(t *testing.T) {
	tests := map[time.Duration]string{
		time.Second: "1000",
		time.Minute: "60000",
		time.Hour:   "3600000",
	}

	for d, ms := range tests {
		b, e := json.Marshal(Duration(d))
		assert.Nil(t, e)
		assert.Equal(t, ms, string(b))

		var du Duration
		e = json.Unmarshal([]byte(ms), &du)
		assert.Nil(t, e)
		assert.Equal(t, Duration(d), du)
	}
}

func TestBSON(t *testing.T) {
	type testT struct {
		Duration Duration `bson:"duration"`
	}

	tests := []time.Duration{
		time.Second,
		time.Minute,
		time.Hour,
	}

	for _, di := range tests {
		b, e := bson.Marshal(testT{Duration: Duration(di)})
		assert.Nil(t, e)

		var out testT
		e = bson.Unmarshal(b, &out)
		assert.Nil(t, e)
		assert.Equal(t, Duration(di), out.Duration)
	}
}

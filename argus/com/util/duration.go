package util

import (
	"encoding/json"
	"time"

	"gopkg.in/mgo.v2/bson"
)

// Duration is just time.Duration.
// It will be marshalled into milli seconds for JSON and BSON, to communicate with other languages.
type Duration time.Duration

// MarshalJSON implements JSON Marshaler interface
// divided by 1E6 to represent it in milli second
func (d Duration) MarshalJSON() ([]byte, error) {
	return json.Marshal(int64(d / 1e6))
}

// UnmarshalJSON implements JSON Unmarshaler interface
// multipled by 1E6 to transfer back to nano second
func (d *Duration) UnmarshalJSON(data []byte) error {
	var ms int64
	if e := json.Unmarshal(data, &ms); e != nil {
		return e
	}
	*d = Duration(ms * 1e6)
	return nil
}

// GetBSON implements BSON Getter interface. It marshals duration into milli seconds.
func (d Duration) GetBSON() (interface{}, error) {
	return int64(d / 1e6), nil
}

// SetBSON implements BSON Setter interface. It unmarshals duration from milli seconds.
func (d *Duration) SetBSON(raw bson.Raw) error {
	var i int64
	if e := raw.Unmarshal(&i); e != nil {
		return e
	}
	*d = Duration(i * 1e6)
	return nil
}
